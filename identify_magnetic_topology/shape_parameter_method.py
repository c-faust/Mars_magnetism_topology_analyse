from __future__ import annotations
"""
Compute MAVEN SWE shape parameters from directional df-E spectra.

For each SWE PAD sample in a requested interval, this script:
- separates the spectrum into parallel and anti-parallel directions,
- applies an LPW spacecraft-potential energy correction,
- computes df = Delta log10(F),
- interpolates df onto a template energy grid,
- sums the absolute difference relative to the template to form a shape parameter,
- maps parallel and anti-parallel spectra into toward/away directions using MAG geometry,
- plots the toward and away shape parameters versus time.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from process_maven_spectra import (
    DEFAULT_SCPOT_MIN_FLAG,
    compute_directional_spectra,
    infer_daily_file,
    load_lpw_spacecraft_potential,
    load_pad_data,
    locate_nearest_index,
)
from identify_magnetic_topology.magnetic_field_direction import (
    load_magnetic_geometry_interval,
    map_parallel_antiparallel_to_toward_away,
    nearest_magnetic_field_direction,
)


DEFAULT_TEMPLATE_PATH = REPO_ROOT / "obtain_the_template" / "combined_template" / "derivative_template_10_100eV.csv"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "identify_magnetic_topology" / "shape_parameter_method"
DEFAULT_MAX_LPW_DELTA_SECONDS = 60.0
DEFAULT_MAX_MAG_DELTA_SECONDS = 60.0
DEFAULT_SHAPE_ENERGY_RANGE_EV = (20.0, 80.0)


def iter_utc_days(start: datetime, end: datetime) -> list[datetime]:
    first = datetime.combine(start.date(), time.min, tzinfo=timezone.utc)
    last = datetime.combine(end.date(), time.min, tzinfo=timezone.utc)
    days = []
    current = first
    while current <= last:
        days.append(current)
        current += timedelta(days=1)
    return days


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_template(path: Path) -> tuple[np.ndarray, np.ndarray]:
    energies: list[float] = []
    values: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "energy_eV" not in fieldnames:
            raise ValueError(f"{path} must contain an energy_eV column.")
        if "mean_df" in fieldnames:
            value_column = "mean_df"
        elif "mean_dlogflux_dlogenergy" in fieldnames:
            value_column = "mean_dlogflux_dlogenergy"
        else:
            raise ValueError(f"{path} must contain mean_df or mean_dlogflux_dlogenergy.")
        for row in reader:
            energy = parse_float(row.get("energy_eV"))
            value = parse_float(row.get(value_column))
            if np.isfinite(energy) and np.isfinite(value) and energy > 0.0:
                energies.append(energy)
                values.append(value)

    if len(energies) < 2:
        raise ValueError(f"Template {path} does not contain at least two usable energy points.")
    energy_array = np.asarray(energies, dtype=float)
    value_array = np.asarray(values, dtype=float)
    order = np.argsort(energy_array)
    return energy_array[order], value_array[order]


def load_lpw_interval(
    data_root: Path | tuple[Path, ...] | list[Path],
    start: datetime,
    end: datetime,
    min_flag: float,
) -> dict[str, np.ndarray] | None:
    times: list[np.ndarray] = []
    potentials: list[np.ndarray] = []
    flags: list[np.ndarray] = []
    for day in iter_utc_days(start, end):
        try:
            path = infer_daily_file(data_root, "lpw", "mrgscpot", day, "cdf")
            lpw = load_lpw_spacecraft_potential(path, min_flag=min_flag)
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            print(f"[shape_parameter] skip LPW day {day.date()}: {exc}", flush=True)
            continue
        times.append(np.asarray(lpw["times"], dtype=float))
        potentials.append(np.asarray(lpw["spacecraft_potential"], dtype=float))
        flags.append(np.asarray(lpw["flag"], dtype=float))

    if not times:
        return None
    merged_times = np.concatenate(times)
    order = np.argsort(merged_times)
    return {
        "times": merged_times[order],
        "spacecraft_potential": np.concatenate(potentials)[order],
        "flag": np.concatenate(flags)[order],
    }


def nearest_lpw_sample(
    lpw: dict[str, np.ndarray],
    target_time: datetime,
    max_delta_seconds: float,
) -> tuple[float, float] | None:
    times = np.asarray(lpw.get("times", []), dtype=float)
    if times.size == 0:
        return None
    index = locate_nearest_index(times, target_time)
    delta = abs(float(times[index]) - target_time.timestamp())
    if delta > max_delta_seconds:
        return None
    potential = float(np.asarray(lpw["spacecraft_potential"], dtype=float)[index])
    if not np.isfinite(potential):
        return None
    return potential, delta


def corrected_energy_axis(energy_eV: np.ndarray, spacecraft_potential_V: float) -> np.ndarray:
    return np.asarray(energy_eV, dtype=float) - float(spacecraft_potential_V)


def df_spectrum(energy_eV: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    energy = np.asarray(energy_eV, dtype=float)
    values = np.asarray(flux, dtype=float)
    usable = np.isfinite(energy) & np.isfinite(values) & (energy > 0.0) & (values > 0.0)
    if np.count_nonzero(usable) < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    energy = energy[usable]
    values = values[usable]
    order = np.argsort(energy)
    energy = energy[order]
    values = values[order]
    unique_energy, unique_indices = np.unique(energy, return_index=True)
    if unique_energy.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    unique_flux = values[unique_indices]
    df = np.gradient(np.log10(unique_flux), edge_order=1)
    return unique_energy, df


def interpolate_flux_to_energy(
    source_energy_eV: np.ndarray,
    source_flux: np.ndarray,
    target_energy_eV: np.ndarray,
) -> np.ndarray:
    source_energy = np.asarray(source_energy_eV, dtype=float)
    source_values = np.asarray(source_flux, dtype=float)
    target_energy = np.asarray(target_energy_eV, dtype=float)
    usable = (
        np.isfinite(source_energy)
        & np.isfinite(source_values)
        & (source_energy > 0.0)
        & (source_values > 0.0)
    )
    if np.count_nonzero(usable) < 2:
        return np.full_like(target_energy, np.nan, dtype=float)

    source_energy = source_energy[usable]
    source_values = source_values[usable]
    order = np.argsort(source_energy)
    source_energy = source_energy[order]
    source_values = source_values[order]
    unique_energy, unique_indices = np.unique(source_energy, return_index=True)
    if unique_energy.size < 2:
        return np.full_like(target_energy, np.nan, dtype=float)

    unique_flux = source_values[unique_indices]
    interpolated = np.interp(target_energy, unique_energy, unique_flux, left=np.nan, right=np.nan)
    return interpolated


def smoothed_flux_for_record(records: list[dict], index: int, flux_key: str, window_points: int) -> tuple[np.ndarray, int]:
    record = records[index]
    target_energy = np.asarray(record["energy_eV"], dtype=float)
    spectra = []
    half_window = max(0, int(window_points) // 2)

    for neighbor_index in range(index - half_window, index + half_window + 1):
        if neighbor_index < 0 or neighbor_index >= len(records):
            continue
        neighbor = records[neighbor_index]
        if "energy_eV" not in neighbor or flux_key not in neighbor:
            continue

        neighbor_energy = np.asarray(neighbor["energy_eV"], dtype=float)
        neighbor_flux = np.asarray(neighbor[flux_key], dtype=float)
        if neighbor_energy.shape == target_energy.shape and np.allclose(neighbor_energy, target_energy, rtol=1e-6, atol=1e-9, equal_nan=False):
            spectra.append(neighbor_flux)
        else:
            spectra.append(interpolate_flux_to_energy(neighbor_energy, neighbor_flux, target_energy))

    if not spectra:
        return np.full_like(target_energy, np.nan, dtype=float), 0
    stacked = np.vstack(spectra)
    with np.errstate(invalid="ignore"):
        smoothed = np.nanmean(stacked, axis=0)
    return smoothed, len(spectra)


def shape_parameter(
    spectrum_energy_eV: np.ndarray,
    spectrum_df: np.ndarray,
    template_energy_eV: np.ndarray,
    template_df: np.ndarray,
    difference_mode: str,
    energy_range_eV: tuple[float, float],
) -> float:
    if spectrum_energy_eV.size < 2 or spectrum_df.size < 2:
        return float("nan")
    usable = (
        np.isfinite(spectrum_energy_eV)
        & np.isfinite(spectrum_df)
        & (spectrum_energy_eV > 0.0)
    )
    if np.count_nonzero(usable) < 2:
        return float("nan")

    energy = spectrum_energy_eV[usable]
    df = spectrum_df[usable]
    order = np.argsort(energy)
    energy = energy[order]
    df = df[order]
    low, high = energy_range_eV
    in_range = (
        (template_energy_eV >= energy[0])
        & (template_energy_eV <= energy[-1])
        & (template_energy_eV >= low)
        & (template_energy_eV <= high)
    )
    if not np.any(in_range):
        return float("nan")

    interp_df = np.interp(template_energy_eV[in_range], energy, df)
    diff = interp_df - template_df[in_range]
    if difference_mode == "absolute":
        diff = np.abs(diff)
    elif difference_mode == "squared":
        diff = diff * diff
    elif difference_mode != "signed":
        raise ValueError(f"Unsupported difference mode: {difference_mode}")
    return float(np.nansum(diff))


def append_nan_row(
    rows: list[dict],
    target_time: datetime,
    reason: str,
    smoothing_sample_count: int = 0,
) -> None:
    rows.append(
        {
            "time_unix": target_time.timestamp(),
            "time_utc": target_time.isoformat(timespec="seconds"),
            "parallel_shape_parameter": float("nan"),
            "antiparallel_shape_parameter": float("nan"),
            "towards_shape_parameter": float("nan"),
            "toward_shape_parameter": float("nan"),
            "away_shape_parameter": float("nan"),
            "spacecraft_potential_V": float("nan"),
            "lpw_delta_seconds": float("nan"),
            "lpw_available": False,
            "energy_correction_applied": False,
            "energy_correction_potential_V": 0.0,
            "mag_time_utc": "",
            "mag_delta_seconds": float("nan"),
            "field_direction": "",
            "field_angle_deg": float("nan"),
            "dot_b_r": float("nan"),
            "toward_source_direction": "",
            "away_source_direction": "",
            "smoothing_sample_count": smoothing_sample_count,
            "status": reason,
        }
    )


def compute_shape_parameters(
    start: datetime,
    end: datetime,
    data_root: Path | tuple[Path, ...] | list[Path],
    template_energy_eV: np.ndarray,
    template_df: np.ndarray,
    cadence_seconds: float,
    max_lpw_delta_seconds: float,
    max_mag_delta_seconds: float,
    spacecraft_potential_min_flag: float,
    forward_pitch_max_deg: float,
    backward_pitch_min_deg: float,
    difference_mode: str,
    shape_energy_range_eV: tuple[float, float],
    spectral_smoothing_window_points: int,
) -> tuple[list[dict], dict]:
    lpw = load_lpw_interval(data_root, start, end, spacecraft_potential_min_flag)
    if lpw is None:
        print(
            "[shape_parameter] no usable LPW spacecraft-potential samples were found; "
            "continuing without spacecraft-potential energy correction.",
            flush=True,
        )
    magnetic_geometry = load_magnetic_geometry_interval(data_root, start, end)
    if magnetic_geometry is None:
        raise FileNotFoundError("No usable MAG sunstate-1sec samples were found in the requested interval.")

    sample_records: list[dict] = []
    skip_counts: dict[str, int] = defaultdict(int)
    next_allowed_unix = start.timestamp()

    for day in iter_utc_days(start, end):
        try:
            pad_file = infer_daily_file(data_root, "swe", "svypad", day, "cdf")
            pad_data = load_pad_data(pad_file)
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            print(f"[shape_parameter] skip SWE day {day.date()}: {exc}", flush=True)
            skip_counts["missing_swe_file"] += 1
            continue

        pad_times = np.asarray(pad_data["times"], dtype=float)
        indices = np.where((pad_times >= start.timestamp()) & (pad_times <= end.timestamp()))[0]
        for index in indices:
            sample_unix = float(pad_times[index])
            if cadence_seconds > 0.0 and sample_unix < next_allowed_unix:
                continue
            if cadence_seconds > 0.0:
                next_allowed_unix = sample_unix + cadence_seconds

            sample_time = datetime.fromtimestamp(sample_unix, tz=timezone.utc)
            try:
                parallel_flux, antiparallel_flux, _, _, _ = compute_directional_spectra(
                    pad_data,
                    sample_time,
                    forward_pitch_max_deg=forward_pitch_max_deg,
                    backward_pitch_min_deg=backward_pitch_min_deg,
                )
            except (KeyError, ValueError, IndexError) as exc:
                print(f"[shape_parameter] directional spectrum failed at {sample_time.isoformat(timespec='seconds')}: {exc}", flush=True)
                skip_counts["directional_spectrum_error"] += 1
                sample_records.append(
                    {
                        "time_unix": sample_unix,
                        "time_utc": sample_time.isoformat(timespec="seconds"),
                        "status": "directional_spectrum_error",
                    }
                )
                continue

            lpw_sample = None if lpw is None else nearest_lpw_sample(lpw, sample_time, max_lpw_delta_seconds)
            record = {
                "time_unix": sample_unix,
                "time_utc": sample_time.isoformat(timespec="seconds"),
                "energy_eV": np.asarray(pad_data["energy"], dtype=float),
                "parallel_flux": np.asarray(parallel_flux, dtype=float),
                "antiparallel_flux": np.asarray(antiparallel_flux, dtype=float),
                "status": "raw_spectrum_ok",
            }
            if lpw_sample is None:
                skip_counts["missing_lpw_sample"] += 1
                record["spacecraft_potential_V"] = float("nan")
                record["lpw_delta_seconds"] = float("nan")
                record["lpw_available"] = False
                record["energy_correction_potential_V"] = 0.0
                record["status"] = "missing_lpw_no_energy_correction"
            else:
                spacecraft_potential, lpw_delta = lpw_sample
                record["spacecraft_potential_V"] = spacecraft_potential
                record["lpw_delta_seconds"] = lpw_delta
                record["lpw_available"] = True
                record["energy_correction_potential_V"] = spacecraft_potential
            sample_records.append(record)

    sample_records.sort(key=lambda item: float(item["time_unix"]))

    rows: list[dict] = []
    for index, record in enumerate(sample_records):
        sample_unix = float(record["time_unix"])
        sample_time = datetime.fromtimestamp(sample_unix, tz=timezone.utc)
        if "energy_eV" not in record:
            append_nan_row(rows, sample_time, str(record.get("status", "directional_spectrum_error")), smoothing_sample_count=0)
            continue

        spacecraft_potential = float(record.get("spacecraft_potential_V", float("nan")))
        lpw_delta = float(record.get("lpw_delta_seconds", float("nan")))
        lpw_available = bool(record.get("lpw_available", False))
        correction_potential = float(record.get("energy_correction_potential_V", 0.0))
        corrected_energy = corrected_energy_axis(record["energy_eV"], correction_potential)
        smoothed_parallel_flux, parallel_smoothing_count = smoothed_flux_for_record(
            sample_records,
            index,
            "parallel_flux",
            spectral_smoothing_window_points,
        )
        smoothed_antiparallel_flux, antiparallel_smoothing_count = smoothed_flux_for_record(
            sample_records,
            index,
            "antiparallel_flux",
            spectral_smoothing_window_points,
        )
        smoothing_count = min(parallel_smoothing_count, antiparallel_smoothing_count)
        parallel_energy, parallel_df = df_spectrum(corrected_energy, smoothed_parallel_flux)
        antiparallel_energy, antiparallel_df = df_spectrum(corrected_energy, smoothed_antiparallel_flux)
        parallel_shape = shape_parameter(
            parallel_energy,
            parallel_df,
            template_energy_eV,
            template_df,
            difference_mode,
            shape_energy_range_eV,
        )
        antiparallel_shape = shape_parameter(
            antiparallel_energy,
            antiparallel_df,
            template_energy_eV,
            template_df,
            difference_mode,
            shape_energy_range_eV,
        )
        mag_direction = nearest_magnetic_field_direction(magnetic_geometry, sample_time, max_mag_delta_seconds)
        if mag_direction is None:
            toward_shape = float("nan")
            away_shape = float("nan")
            mag_time_utc = ""
            mag_delta_seconds = float("nan")
            field_direction = "missing_mag_sample"
            field_angle_deg = float("nan")
            dot_b_r = float("nan")
            toward_source_direction = "undefined"
            away_source_direction = "undefined"
            skip_counts["missing_mag_sample"] += 1
        else:
            toward_shape, away_shape, toward_source_direction, away_source_direction = map_parallel_antiparallel_to_toward_away(
                parallel_shape,
                antiparallel_shape,
                mag_direction.field_direction,
            )
            mag_time_utc = mag_direction.time_utc
            mag_delta_seconds = abs(float(mag_direction.time_unix) - sample_unix)
            field_direction = mag_direction.field_direction
            field_angle_deg = mag_direction.angle_deg
            dot_b_r = mag_direction.dot_b_r

        status = "ok"
        if not np.isfinite(parallel_shape) or not np.isfinite(antiparallel_shape):
            skip_counts["shape_parameter_nan"] += 1
            status = "partial_or_nan"
        if not np.isfinite(toward_shape) or not np.isfinite(away_shape):
            if status == "ok":
                status = "toward_away_nan"
        if not lpw_available:
            status = f"{status}_missing_lpw_no_energy_correction"

        rows.append(
            {
                "time_unix": sample_unix,
                "time_utc": sample_time.isoformat(timespec="seconds"),
                "parallel_shape_parameter": parallel_shape,
                "antiparallel_shape_parameter": antiparallel_shape,
                "towards_shape_parameter": toward_shape,
                "toward_shape_parameter": toward_shape,
                "away_shape_parameter": away_shape,
                "spacecraft_potential_V": spacecraft_potential,
                "lpw_delta_seconds": lpw_delta,
                "lpw_available": lpw_available,
                "energy_correction_applied": lpw_available,
                "energy_correction_potential_V": correction_potential,
                "mag_time_utc": mag_time_utc,
                "mag_delta_seconds": mag_delta_seconds,
                "field_direction": field_direction,
                "field_angle_deg": field_angle_deg,
                "dot_b_r": dot_b_r,
                "toward_source_direction": toward_source_direction,
                "away_source_direction": away_source_direction,
                "smoothing_sample_count": smoothing_count,
                "status": status,
            }
        )

    return rows, dict(skip_counts)


def write_shape_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time_unix",
                "time_utc",
                "parallel_shape_parameter",
                "antiparallel_shape_parameter",
                "towards_shape_parameter",
                "toward_shape_parameter",
                "away_shape_parameter",
                "spacecraft_potential_V",
                "lpw_delta_seconds",
                "lpw_available",
                "energy_correction_applied",
                "energy_correction_potential_V",
                "mag_time_utc",
                "mag_delta_seconds",
                "field_direction",
                "field_angle_deg",
                "dot_b_r",
                "toward_source_direction",
                "away_source_direction",
                "smoothing_sample_count",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_shape_parameters(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = [datetime.fromtimestamp(float(row["time_unix"]), tz=timezone.utc) for row in rows]
    toward = np.asarray([row["toward_shape_parameter"] for row in rows], dtype=float)
    away = np.asarray([row["away_shape_parameter"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    if rows:
        ax.plot(times, toward, marker="o", markersize=3, linewidth=1.2, label="towards")
        ax.plot(times, away, marker="s", markersize=3, linewidth=1.2, label="away")
    else:
        ax.text(0.5, 0.5, "No shape-parameter samples", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0.0, color="0.35", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("UTC")
    ax.set_ylabel("Shape parameter")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def default_output_dir(output_root: Path, start: datetime, end: datetime) -> Path:
    return output_root / f"{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute MAVEN magnetic-topology shape parameters from SWE PAD spectra.")
    parser.add_argument("--start", required=True, help="UTC interval start, for example 2024-11-07T02:00:00.")
    parser.add_argument("--end", required=True, help="UTC interval end, for example 2024-11-07T02:30:00.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE_PATH), help="Template CSV with energy_eV and mean_df.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for MAVEN data.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory.")
    parser.add_argument(
        "--cadence-seconds",
        type=float,
        default=0.0,
        help="Optional thinning cadence. Use 0 to process every SWE PAD time in the interval.",
    )
    parser.add_argument("--forward-pitch-max", type=float, default=30.0)
    parser.add_argument("--backward-pitch-min", type=float, default=150.0)
    parser.add_argument("--spacecraft-potential-min-flag", type=float, default=DEFAULT_SCPOT_MIN_FLAG)
    parser.add_argument("--max-lpw-delta-seconds", type=float, default=DEFAULT_MAX_LPW_DELTA_SECONDS)
    parser.add_argument(
        "--max-mag-delta-seconds",
        type=float,
        default=DEFAULT_MAX_MAG_DELTA_SECONDS,
        help="Maximum allowed time offset to the nearest MAG sunstate-1sec sample.",
    )
    parser.add_argument(
        "--difference-mode",
        choices=("signed", "absolute", "squared"),
        default="absolute",
        help="How to sum df-template differences into the shape parameter.",
    )
    parser.add_argument(
        "--shape-energy-range",
        nargs=2,
        type=float,
        default=DEFAULT_SHAPE_ENERGY_RANGE_EV,
        metavar=("LOW_EV", "HIGH_EV"),
        help="Energy range used in the shape-parameter sum.",
    )
    parser.add_argument(
        "--spectral-smoothing-window",
        type=int,
        default=5,
        help="Centered moving-average window, in SWE samples, applied to directional flux spectra before df and shape-parameter calculation. Use 1 for no smoothing.",
    )
    parser.add_argument(
        "--no-spectral-smoothing",
        action="store_true",
        help="Disable time smoothing of the directional flux spectra before df and shape-parameter calculation.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    start = parse_iso_timestamp(args.start).astimezone(timezone.utc)
    end = parse_iso_timestamp(args.end).astimezone(timezone.utc)
    if end <= start:
        raise ValueError("--end must be later than --start.")
    shape_energy_range = (float(args.shape_energy_range[0]), float(args.shape_energy_range[1]))
    if not (0.0 < shape_energy_range[0] < shape_energy_range[1]):
        raise ValueError("--shape-energy-range must satisfy 0 < LOW_EV < HIGH_EV.")
    spectral_smoothing_window = int(args.spectral_smoothing_window)
    if spectral_smoothing_window < 1:
        raise ValueError("--spectral-smoothing-window must be at least 1.")
    if spectral_smoothing_window % 2 == 0:
        spectral_smoothing_window += 1
    spectral_smoothing_enabled = not bool(args.no_spectral_smoothing) and spectral_smoothing_window > 1
    spectral_smoothing_window_points = spectral_smoothing_window if spectral_smoothing_enabled else 1

    template_path = Path(args.template).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = default_output_dir(Path(args.output_root).expanduser().resolve(), start, end)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_energy, template_df = load_template(template_path)
    rows, skip_counts = compute_shape_parameters(
        start=start,
        end=end,
        data_root=data_root,
        template_energy_eV=template_energy,
        template_df=template_df,
        cadence_seconds=max(0.0, float(args.cadence_seconds)),
        max_lpw_delta_seconds=float(args.max_lpw_delta_seconds),
        max_mag_delta_seconds=float(args.max_mag_delta_seconds),
        spacecraft_potential_min_flag=float(args.spacecraft_potential_min_flag),
        forward_pitch_max_deg=float(args.forward_pitch_max),
        backward_pitch_min_deg=float(args.backward_pitch_min),
        difference_mode=args.difference_mode,
        shape_energy_range_eV=shape_energy_range,
        spectral_smoothing_window_points=spectral_smoothing_window_points,
    )

    csv_path = output_dir / "shape_parameters.csv"
    png_path = output_dir / "shape_parameters.png"
    write_shape_csv(csv_path, rows)
    plot_shape_parameters(png_path, rows)

    summary = {
        "start": start.isoformat(timespec="seconds"),
        "end": end.isoformat(timespec="seconds"),
        "template": str(template_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "sample_count": len(rows),
        "finite_parallel_count": int(np.count_nonzero(np.isfinite([row["parallel_shape_parameter"] for row in rows]))),
        "finite_antiparallel_count": int(np.count_nonzero(np.isfinite([row["antiparallel_shape_parameter"] for row in rows]))),
        "finite_towards_count": int(np.count_nonzero(np.isfinite([row["towards_shape_parameter"] for row in rows]))),
        "finite_toward_count": int(np.count_nonzero(np.isfinite([row["toward_shape_parameter"] for row in rows]))),
        "finite_away_count": int(np.count_nonzero(np.isfinite([row["away_shape_parameter"] for row in rows]))),
        "difference_mode": args.difference_mode,
        "shape_energy_range_eV": list(shape_energy_range),
        "spectral_smoothing": {
            "enabled": spectral_smoothing_enabled,
            "window_points": spectral_smoothing_window_points,
            "method": "centered nanmean over directional flux spectra before spacecraft-potential energy correction, df calculation, and shape-parameter calculation",
        },
        "energy_correction": "When LPW is available: corrected_energy_eV = measured_energy_eV - spacecraft_potential_V. When LPW is missing: continue with correction potential 0 V and mark lpw_available=false / energy_correction_applied=false.",
        "magnetic_direction_rule": "MAG SS/MSO geometry is used. If angle(B, radial position) > 90 deg, B points toward the surface and parallel maps to toward; if angle < 90 deg, B points away from the surface and antiparallel maps to toward.",
        "max_mag_delta_seconds": float(args.max_mag_delta_seconds),
        "skip_counts": skip_counts,
        "outputs": {
            "csv": str(csv_path),
            "plot": str(png_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
