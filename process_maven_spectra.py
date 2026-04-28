from __future__ import annotations
"""
Single-time MAVEN spectrum processing.

This module sits between raw files and higher-level science products:
- input: one SWE PAD file + one MAG file
- output: one directional spectrum summary around one selected time

The core idea is:
1. load the electron pitch-angle distribution (PAD)
2. find the target time
3. separate electrons moving roughly parallel / anti-parallel to B
4. attach the nearest magnetic-field sample from MAG

method to use this code :python process_maven_spectra.py --time 2024-11-07T02:15:00

"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

import cdflib
import matplotlib.pyplot as plt
import numpy as np

from download_maven_data import DEFAULT_DATA_ROOT, parse_filename, parse_iso_timestamp


@dataclass(frozen=True)
class SpectrumResult:
    """Compact output object written to JSON by the single-time pipeline."""
    target_time: str
    swe_time: str
    mag_time: str
    pad_file: str
    mag_file: str
    magnetic_field_nT: list[float]
    forward_pitch_max_deg: float
    backward_pitch_min_deg: float
    energy_eV: list[float]
    forward_flux: list[float]
    backward_flux: list[float]
    forward_pitch_bins_deg: list[float]
    backward_pitch_bins_deg: list[float]


def unix_seconds_from_cdf_epoch(epoch_values: np.ndarray) -> np.ndarray:
    """Convert CDF epoch values into UNIX seconds."""
    datetimes = cdflib.cdfepoch.to_datetime(epoch_values)
    return datetimes.astype("datetime64[ns]").astype(np.int64) / 1e9


def unix_seconds_from_numeric_time(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size == 0:
        return flat
    if np.nanmedian(flat) > 1e12:
        return flat / 1000.0
    return flat


def pick_first_variable(cdf: cdflib.CDF, names: list[str]) -> np.ndarray | None:
    """Return the first variable from `names` that exists in the CDF file."""
    info = cdf.cdf_info()
    available = set(info.zVariables) | set(info.rVariables)
    for name in names:
        if name in available:
            return np.asarray(cdf.varget(name))
    return None


def load_pad_data(path: Path) -> dict:
    """Load and normalize one SWE PAD file.

    Different MAVEN products can use different variable names and axis orders.
    This function standardizes them into one internal structure:
    - `times`
    - `energy`
    - `pitch`
    - `flux` with axis order `(time, pitch, energy)`
    """
    cdf = cdflib.CDF(str(path))
    time_values = pick_first_variable(cdf, ["epoch", "time_unix", "time_met"])
    if time_values is None:
        raise KeyError(f"No usable time variable was found in {path.name}.")

    if time_values.dtype.kind in {"i", "u"} and np.nanmedian(time_values) > 1e12:
        times = unix_seconds_from_cdf_epoch(time_values)
    else:
        times = unix_seconds_from_numeric_time(time_values)

    energy = pick_first_variable(cdf, ["energy", "g_engy", "en_label"])
    if energy is None:
        raise KeyError(f"No energy variable was found in {path.name}.")
    energy = np.asarray(energy, dtype=float).reshape(-1)
    pitch_index = pick_first_variable(cdf, ["pindex", "pa_label"])
    pitch_variable = pick_first_variable(cdf, ["pitch_angle", "pa", "pitch", "g_pa"])

    info = cdf.cdf_info()
    variable_names = list(info.zVariables) + list(info.rVariables)
    flux = None
    flux_name = None
    for name in variable_names:
        lowered = name.lower()
        data = np.asarray(cdf.varget(name))
        if data.ndim < 3:
            continue
        if "flux" not in lowered and "diff" not in lowered:
            continue
        flux = data.astype(float)
        flux_name = name
        break

    if flux is None:
        raise KeyError(f"No 3D flux variable was found in {path.name}.")

    time_axis = find_axis_by_length(flux.shape, len(times), "time", flux_name)
    pitch_axis_size = infer_pitch_axis_size(flux.shape, pitch_index)
    pitch_axis = find_axis_by_length(flux.shape, pitch_axis_size, "pitch", flux_name)
    energy_axis = find_axis_by_length(flux.shape, len(energy), "energy", flux_name)

    reordered = np.moveaxis(flux, (time_axis, pitch_axis, energy_axis), (0, 1, 2))
    pitch = normalize_pitch_variable(pitch_variable, flux.shape, (time_axis, pitch_axis, energy_axis))
    return {
        "times": times,
        "energy": energy,
        "pitch": pitch,
        "flux": reordered,
        "flux_name": flux_name,
    }


def find_axis_by_length(shape: tuple[int, ...], size: int, label: str, variable_name: str) -> int:
    """Guess which axis corresponds to time/pitch/energy from array shape."""
    matches = [index for index, axis_size in enumerate(shape) if axis_size == size]
    if not matches:
        raise ValueError(f"Could not map {label} axis for {variable_name}. shape={shape}, expected size={size}.")
    return matches[-1] if label == "energy" else matches[0]


def infer_pitch_axis_size(flux_shape: tuple[int, ...], pitch_index: np.ndarray | None) -> int:
    if pitch_index is not None:
        return int(np.asarray(pitch_index).reshape(-1).shape[0])

    candidate_sizes = [size for size in flux_shape if size not in {flux_shape[0], flux_shape[-1]}]
    if candidate_sizes:
        return candidate_sizes[0]
    raise ValueError(f"Could not infer pitch-axis size from shape {flux_shape}.")


def normalize_pitch_variable(
    pitch_variable: np.ndarray | None,
    flux_shape: tuple[int, ...],
    source_axes: tuple[int, int, int],
) -> np.ndarray:
    """Normalize pitch-angle data so downstream code can index it consistently."""
    if pitch_variable is None:
        raise KeyError("No pitch-angle variable was found. Expected one of pitch_angle/pa/pitch/g_pa.")

    pitch_array = np.asarray(pitch_variable, dtype=float)
    if pitch_array.ndim == 1:
        return pitch_array.reshape(-1)
    if pitch_array.shape == flux_shape:
        return np.moveaxis(pitch_array, source_axes, (0, 1, 2))
    raise ValueError(f"Unsupported pitch-angle shape {pitch_array.shape}.")


def parse_odl_record_columns(lines: list[str]) -> list[str] | None:
    """Parse the ODL-style MAG header and reconstruct column names.

    MAVEN MAG `.sts` files describe the tabular payload in a nested ODL header.
    We flatten vector/scalar records into names like `POSN.X` and `OB_B.Z`.
    """
    record_start = None
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if record_start is None and stripped.startswith("OBJECT") and stripped.endswith("RECORD"):
            record_start = index
            break

    if record_start is None:
        return None

    stack: list[dict] = [{"type": "RECORD", "name": "RECORD"}]
    columns: list[str] = []
    for raw_line in lines[record_start + 1 :]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("OBJECT"):
            _, value = stripped.split("=", 1)
            stack.append({"type": value.strip().upper(), "name": ""})
            continue
        if stripped.startswith("NAME") and "=" in stripped and stack:
            _, value = stripped.split("=", 1)
            stack[-1]["name"] = value.strip()
            continue
        if stripped.startswith("END_OBJECT") and stack:
            completed = stack.pop()
            if completed["type"] == "SCALAR":
                parents = [item["name"] for item in stack if item["type"] == "VECTOR" and item["name"]]
                if parents:
                    columns.append(f"{parents[-1]}.{completed['name']}")
                else:
                    columns.append(completed["name"])
            if completed["type"] == "RECORD":
                break
            if not stack:
                break

    return columns or None


def parse_mag_sts(path: Path) -> dict:
    """Parse the numeric body of a MAVEN MAG STS file."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        raw_lines = handle.readlines()

    columns = parse_odl_record_columns(raw_lines)
    rows: list[list[float]] = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^[+\-.\d]", line):
            rows.append([float(item) for item in line.split()])

    if not columns or not rows:
        raise ValueError(f"Failed to parse MAG STS file {path.name}.")

    matrix = np.asarray(rows, dtype=float)
    if matrix.shape[1] != len(columns):
        raise ValueError(f"Column count mismatch in {path.name}: parsed {len(columns)}, data {matrix.shape[1]}.")

    return {"columns": columns, "data": matrix}


def build_mag_times(columns: list[str], data: np.ndarray) -> np.ndarray:
    """Rebuild UNIX timestamps from the split MAG TIME.* columns."""
    needed = ["TIME.YEAR", "TIME.DOY", "TIME.HOUR", "TIME.MIN", "TIME.SEC", "TIME.MSEC"]
    index = {name: columns.index(name) for name in needed}
    timestamps = []
    for row in data:
        base = datetime.strptime(f"{int(row[index['TIME.YEAR']])} {int(row[index['TIME.DOY']]):03d}", "%Y %j").replace(
            tzinfo=timezone.utc
        )
        timestamp = base.replace(
            hour=int(row[index["TIME.HOUR"]]),
            minute=int(row[index["TIME.MIN"]]),
            second=int(row[index["TIME.SEC"]]),
            microsecond=int(row[index["TIME.MSEC"]]) * 1000,
        )
        timestamps.append(timestamp.timestamp())
    return np.asarray(timestamps, dtype=float)


def locate_nearest_index(times: np.ndarray, target_time: datetime) -> int:
    target_unix = target_time.timestamp()
    return int(np.argmin(np.abs(times - target_unix)))


def format_unix_time(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="seconds")


def compute_directional_spectra(
    pad_data: dict,
    target_time: datetime,
    forward_pitch_max_deg: float = 30.0,
    backward_pitch_min_deg: float = 150.0,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    time_index = locate_nearest_index(pad_data["times"], target_time)
    flux_at_time = np.asarray(pad_data["flux"][time_index], dtype=float)
    pitch = np.asarray(pad_data["pitch"], dtype=float)

    if pitch.ndim == 1:
        forward_mask = pitch < forward_pitch_max_deg
        backward_mask = pitch > backward_pitch_min_deg
        if not np.any(forward_mask):
            raise ValueError(f"No pitch-angle bins satisfy pitch < {forward_pitch_max_deg:g} degrees.")
        if not np.any(backward_mask):
            raise ValueError(f"No pitch-angle bins satisfy pitch > {backward_pitch_min_deg:g} degrees.")
        forward_flux = np.nanmean(flux_at_time[forward_mask, :], axis=0)
        backward_flux = np.nanmean(flux_at_time[backward_mask, :], axis=0)
        return forward_flux, backward_flux, time_index, pitch[forward_mask], pitch[backward_mask]

    pitch_at_time = pitch[time_index]
    forward_flux = np.full(flux_at_time.shape[1], np.nan, dtype=float)
    backward_flux = np.full(flux_at_time.shape[1], np.nan, dtype=float)
    forward_bins: list[float] = []
    backward_bins: list[float] = []
    for energy_index in range(flux_at_time.shape[1]):
        pitch_column = pitch_at_time[:, energy_index]
        forward_mask = pitch_column < forward_pitch_max_deg
        backward_mask = pitch_column > backward_pitch_min_deg
        if np.any(forward_mask):
            forward_flux[energy_index] = np.nanmean(flux_at_time[forward_mask, energy_index])
            forward_bins.extend(pitch_column[forward_mask].tolist())
        if np.any(backward_mask):
            backward_flux[energy_index] = np.nanmean(flux_at_time[backward_mask, energy_index])
            backward_bins.extend(pitch_column[backward_mask].tolist())

    if np.all(np.isnan(forward_flux)):
        raise ValueError(f"No pitch-angle bins satisfy pitch < {forward_pitch_max_deg:g} degrees.")
    if np.all(np.isnan(backward_flux)):
        raise ValueError(f"No pitch-angle bins satisfy pitch > {backward_pitch_min_deg:g} degrees.")

    return (
        forward_flux,
        backward_flux,
        time_index,
        np.unique(np.round(np.asarray(forward_bins, dtype=float), 3)),
        np.unique(np.round(np.asarray(backward_bins, dtype=float), 3)),
    )


def nearest_mag_vector(path: Path, target_time: datetime) -> tuple[np.ndarray, str]:
    parsed = parse_mag_sts(path)
    times = build_mag_times(parsed["columns"], parsed["data"])
    index = locate_nearest_index(times, target_time)
    columns = parsed["columns"]
    data = parsed["data"]
    vector = np.array(
        [
            data[index, columns.index("OB_B.X")],
            data[index, columns.index("OB_B.Y")],
            data[index, columns.index("OB_B.Z")],
        ],
        dtype=float,
    )
    return vector, format_unix_time(times[index])


def build_output_dir(target_time: datetime, output_root: Path) -> Path:
    directory = output_root / target_time.strftime("%Y%m%dT%H%M%S")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_spectra(
    energy: np.ndarray,
    forward_flux: np.ndarray,
    backward_flux: np.ndarray,
    output_path: Path,
    forward_pitch_max_deg: float = 30.0,
    backward_pitch_min_deg: float = 150.0,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.loglog(
        energy,
        forward_flux,
        marker="o",
        markersize=3,
        linewidth=1.2,
        label=f"Pitch < {forward_pitch_max_deg:g} deg",
    )
    plt.loglog(
        energy,
        backward_flux,
        marker="s",
        markersize=3,
        linewidth=1.2,
        label=f"Pitch > {backward_pitch_min_deg:g} deg",
    )
    plt.xlabel("Energy (eV)")
    plt.ylabel("Differential energy flux")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def process_target_time(
    target_time: datetime,
    pad_file: Path,
    mag_file: Path,
    output_root: Path,
    forward_pitch_max_deg: float = 30.0,
    backward_pitch_min_deg: float = 150.0,
) -> SpectrumResult:
    pad_data = load_pad_data(pad_file)
    forward_flux, backward_flux, pad_index, forward_bins, backward_bins = compute_directional_spectra(
        pad_data,
        target_time,
        forward_pitch_max_deg=forward_pitch_max_deg,
        backward_pitch_min_deg=backward_pitch_min_deg,
    )
    magnetic_field, mag_time = nearest_mag_vector(mag_file, target_time)
    output_dir = build_output_dir(target_time, output_root)

    plot_path = output_dir / "directional_electron_spectra.png"
    plot_spectra(
        pad_data["energy"],
        forward_flux,
        backward_flux,
        plot_path,
        forward_pitch_max_deg=forward_pitch_max_deg,
        backward_pitch_min_deg=backward_pitch_min_deg,
    )

    result = SpectrumResult(
        target_time=target_time.isoformat(timespec="seconds"),
        swe_time=format_unix_time(pad_data["times"][pad_index]),
        mag_time=mag_time,
        pad_file=str(pad_file),
        mag_file=str(mag_file),
        magnetic_field_nT=magnetic_field.tolist(),
        forward_pitch_max_deg=float(forward_pitch_max_deg),
        backward_pitch_min_deg=float(backward_pitch_min_deg),
        energy_eV=pad_data["energy"].tolist(),
        forward_flux=forward_flux.tolist(),
        backward_flux=backward_flux.tolist(),
        forward_pitch_bins_deg=forward_bins.tolist(),
        backward_pitch_bins_deg=backward_bins.tolist(),
    )

    (output_dir / "spectrum_summary.json").write_text(
        json.dumps(result.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def infer_daily_file(
    data_root: Path,
    instrument: str,
    datatype_alias: str,
    day: datetime,
    extension: str,
) -> Path:
    day_code = day.strftime("%Y%m%d")
    for candidate in data_root.rglob(f"mvn_{instrument}_*"):
        if not candidate.is_file():
            continue
        parsed = parse_filename(candidate.name)
        if not parsed:
            continue
        if parsed["instrument"] != instrument:
            continue
        if parsed["extension"] != extension:
            continue
        if f"{parsed['year']}{parsed['month']}{parsed['day']}" != day_code:
            continue
        if datatype_alias not in parsed["description"].lower():
            continue
        return candidate
    raise FileNotFoundError(f"Could not find {instrument}/{datatype_alias} for {day_code} under {data_root}.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process MAVEN PAD and MAG files for a target timestamp.")
    parser.add_argument("--time", required=True, help="Target timestamp, for example 2024-11-07T12:00:00.")
    parser.add_argument("--pad-file", help="Path to the SWE PAD CDF file.")
    parser.add_argument("--mag-file", help="Path to the MAG STS file.")
    parser.add_argument(
        "--forward-pitch-max",
        type=float,
        default=30.0,
        help="Upper pitch-angle bound, in degrees, used for the forward spectrum.",
    )
    parser.add_argument(
        "--backward-pitch-min",
        type=float,
        default=150.0,
        help="Lower pitch-angle bound, in degrees, used for the backward spectrum.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for downloaded data.")
    parser.add_argument(
        "--output-root",
        default=str(Path("outputs") / "maven_spectra"),
        help="Directory used to store figures and summary files.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    target_time = parse_iso_timestamp(args.time)
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    day = target_time

    pad_file = Path(args.pad_file).expanduser().resolve() if args.pad_file else infer_daily_file(
        data_root=data_root,
        instrument="swe",
        datatype_alias="svypad",
        day=day,
        extension="cdf",
    )
    mag_file = Path(args.mag_file).expanduser().resolve() if args.mag_file else infer_daily_file(
        data_root=data_root,
        instrument="mag",
        datatype_alias="ss1s",
        day=day,
        extension="sts",
    )

    result = process_target_time(
        target_time=target_time,
        pad_file=pad_file,
        mag_file=mag_file,
        output_root=output_root,
        forward_pitch_max_deg=args.forward_pitch_max,
        backward_pitch_min_deg=args.backward_pitch_min,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
