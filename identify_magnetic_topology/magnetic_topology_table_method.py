from __future__ import annotations
"""
Combine SWE shape parameters and PAD scores into magnetic-topology classes.

The workflow follows the seven-topology lookup table:
- shape_parameter_method.py separates photoelectron-like (Phe) and
  solar-wind/backscattered-like (SWe) spectra for toward/away directions.
- PAD_score_method.py identifies loss-cone / no-loss-cone PAD behavior.
- This script aligns both outputs in time, computes the away/toward flux ratio
  for 35-60 eV, and applies the table rules to produce topology vs. time.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from process_maven_spectra import compute_directional_spectra, infer_daily_file, load_pad_data
from identify_magnetic_topology.magnetic_field_direction import map_parallel_antiparallel_to_toward_away
from identify_magnetic_topology.PAD_score_method import (
    DEFAULT_ENERGY_RANGE_EV as DEFAULT_PAD_ENERGY_RANGE_EV,
    DEFAULT_MAX_MAG_DELTA_SECONDS,
    PitchAngleBands,
    classify_pad_timeseries,
    plot_pad_score_time_series,
    plot_time_series_classification,
)
from identify_magnetic_topology.shape_parameter_method import (
    DEFAULT_MAX_LPW_DELTA_SECONDS,
    DEFAULT_SCPOT_MIN_FLAG,
    DEFAULT_SHAPE_ENERGY_RANGE_EV,
    DEFAULT_TEMPLATE_PATH,
    compute_shape_parameters,
    load_template,
    plot_shape_parameters,
    write_shape_csv,
)
from region_id.classify_region_id import (
    RegionClassifierConfig,
    classify_interval as classify_region_interval,
    plot_region_ids,
    write_region_csv,
    write_summary_json as write_region_summary_json,
)


DEFAULT_OUTPUT_ROOT = Path("outputs") / "identify_magnetic_topology" / "magnetic_topology_based_on_Xu2019"
DEFAULT_RATIO_ENERGY_RANGE_EV = (35.0, 60.0)
DEFAULT_LOSS_CONE_PAD_SCORE_THRESHOLD = -3.0
DEFAULT_ELECTRON_VOID_ENERGY_EV = 40.0
DEFAULT_ELECTRON_VOID_FLUX_THRESHOLD = 1.0e5
DEFAULT_MAX_REGION_ID_DELTA_SECONDS = 2.0
DEFAULT_REGION_ID_BOUNDARY_MARGIN_KM = 100.0


def format_unix_time(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")


def default_output_dir(output_root: Path, start: datetime, end: datetime) -> Path:
    return output_root / f"{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}"


def finite_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def shape_class(shape_parameter: float, photoelectron_threshold: float) -> str:
    value = finite_float(shape_parameter)
    if not np.isfinite(value):
        return "unknown"
    return "Phe" if value <= photoelectron_threshold else "SWe"


def pad_lc_class_from_score(score: float, loss_cone_threshold: float) -> str:
    value = finite_float(score)
    if not np.isfinite(value):
        return "unknown"
    return "LC" if value < loss_cone_threshold else "No LC"


def band_mean_flux(energy_eV: np.ndarray, flux: np.ndarray, energy_range_eV: tuple[float, float]) -> float:
    energy = np.asarray(energy_eV, dtype=float)
    values = np.asarray(flux, dtype=float)
    selected = np.isfinite(energy) & (energy >= energy_range_eV[0]) & (energy <= energy_range_eV[1])
    usable = selected & np.isfinite(values)
    if not np.any(usable):
        return float("nan")
    return float(np.nanmean(values[usable]))


def compute_at_ratio_for_shape_rows(
    shape_rows: list[dict],
    data_root: Path,
    energy_range_eV: tuple[float, float],
    forward_pitch_max_deg: float,
    backward_pitch_min_deg: float,
    electron_void_energy_eV: float,
    electron_void_flux_threshold: float,
) -> dict[float, dict]:
    pad_cache: dict[str, dict | None] = {}
    ratio_by_time: dict[float, dict] = {}
    for row in shape_rows:
        sample_unix = finite_float(row.get("time_unix"))
        if not np.isfinite(sample_unix):
            continue
        sample_time = datetime.fromtimestamp(sample_unix, tz=timezone.utc)
        day_key = sample_time.strftime("%Y%m%d")
        if day_key not in pad_cache:
            try:
                pad_file = infer_daily_file(data_root, "swe", "svypad", sample_time, "cdf")
                pad_data = load_pad_data(pad_file)
                pad_data["source_file"] = str(pad_file)
                pad_cache[day_key] = pad_data
            except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
                print(f"[topology-table] skip A/T ratio day {day_key}: {exc}", flush=True)
                pad_cache[day_key] = None

        pad_data = pad_cache.get(day_key)
        if pad_data is None:
            ratio_by_time[sample_unix] = {"status": "missing_swe_file"}
            continue

        try:
            parallel_flux, antiparallel_flux, time_index, _, _ = compute_directional_spectra(
                pad_data,
                sample_time,
                forward_pitch_max_deg=forward_pitch_max_deg,
                backward_pitch_min_deg=backward_pitch_min_deg,
            )
        except (KeyError, ValueError, IndexError) as exc:
            ratio_by_time[sample_unix] = {"status": f"directional_spectrum_error:{exc}"}
            continue

        parallel_band_flux = band_mean_flux(np.asarray(pad_data["energy"], dtype=float), parallel_flux, energy_range_eV)
        antiparallel_band_flux = band_mean_flux(np.asarray(pad_data["energy"], dtype=float), antiparallel_flux, energy_range_eV)
        energy = np.asarray(pad_data["energy"], dtype=float)
        finite_energy_indices = np.where(np.isfinite(energy))[0]
        if finite_energy_indices.size:
            void_energy_index = finite_energy_indices[
                np.argmin(np.abs(energy[finite_energy_indices] - electron_void_energy_eV))
            ]
            void_energy_actual = float(energy[void_energy_index])
            flux_at_void_energy = np.asarray(pad_data["flux"][time_index], dtype=float)[:, void_energy_index]
            flux_40eV = (
                float(np.nanmean(flux_at_void_energy))
                if np.any(np.isfinite(flux_at_void_energy))
                else float("nan")
            )
        else:
            void_energy_actual = float("nan")
            flux_40eV = float("nan")
        electron_void = bool(np.isfinite(flux_40eV) and flux_40eV < electron_void_flux_threshold)

        toward_flux, away_flux, toward_source, away_source = map_parallel_antiparallel_to_toward_away(
            parallel_band_flux,
            antiparallel_band_flux,
            str(row.get("field_direction", "")),
        )
        ratio = float("nan")
        if np.isfinite(toward_flux) and toward_flux > 0.0 and np.isfinite(away_flux):
            ratio = float(away_flux / toward_flux)

        pad_times = np.asarray(pad_data["times"], dtype=float)
        ratio_by_time[sample_unix] = {
            "at_ratio_35_60eV": ratio,
            "toward_flux_35_60eV": toward_flux,
            "away_flux_35_60eV": away_flux,
            "toward_ratio_source_direction": toward_source,
            "away_ratio_source_direction": away_source,
            "ratio_swe_time_utc": format_unix_time(pad_times[time_index]),
            "ratio_swe_delta_seconds": abs(float(pad_times[time_index]) - sample_unix),
            "ratio_energy_low_eV": energy_range_eV[0],
            "ratio_energy_high_eV": energy_range_eV[1],
            "electron_void": electron_void,
            "electron_void_target_energy_eV": electron_void_energy_eV,
            "electron_void_actual_energy_eV": void_energy_actual,
            "electron_void_flux": flux_40eV,
            "electron_void_flux_threshold": electron_void_flux_threshold,
            "ratio_source_file": str(pad_data.get("source_file", "")),
            "status": "ok" if np.isfinite(ratio) else "ratio_nan",
        }
    return ratio_by_time


def nearest_pad_row(pad_df: pd.DataFrame, time_unix: float, max_delta_seconds: float) -> tuple[pd.Series | None, float]:
    if pad_df.empty or not np.isfinite(time_unix):
        return None, float("nan")
    times = np.asarray(pad_df["time_unix"], dtype=float)
    if times.size == 0:
        return None, float("nan")
    index = int(np.nanargmin(np.abs(times - time_unix)))
    delta = abs(float(times[index]) - time_unix)
    if delta > max_delta_seconds:
        return None, delta
    return pad_df.iloc[index], delta


def nearest_region_row(
    region_df: pd.DataFrame | None,
    time_unix: float,
    max_delta_seconds: float,
) -> tuple[pd.Series | None, float]:
    if region_df is None or region_df.empty or not np.isfinite(time_unix):
        return None, float("nan")
    times = np.asarray(region_df["time_unix"], dtype=float)
    finite = np.isfinite(times)
    if not np.any(finite):
        return None, float("nan")
    finite_indices = np.where(finite)[0]
    sorted_times = times[finite]
    insertion = int(np.searchsorted(sorted_times, time_unix, side="left"))
    candidates = [
        item
        for item in (insertion - 1, insertion)
        if 0 <= item < sorted_times.size
    ]
    sorted_index = min(
        candidates,
        key=lambda item: abs(float(sorted_times[item]) - time_unix),
    )
    index = int(finite_indices[sorted_index])
    delta = abs(float(times[index]) - time_unix)
    if delta > max_delta_seconds:
        return None, delta
    return region_df.iloc[index], delta


def topology_from_table(
    away_shape_class: str,
    toward_shape_class: str,
    away_pad: str,
    toward_pad: str,
    ratio: float,
    void: bool,
) -> tuple[str, str, str]:
    if void:
        return "C-V", "4", "Superthermal electron void: the electron flux near 40 eV is below the configured absolute threshold."

    if away_shape_class == "Phe" and toward_shape_class == "Phe":
        if np.isfinite(ratio) and 0.2 < ratio < 5.0:
            return "C-D", "1", "A/T ratio is within 0.2 < r < 5 for Phe/Phe spectra."
        if np.isfinite(ratio) and (ratio > 5.0 or ratio < 0.2):
            return "C-X", "2a", "A/T ratio is outside 0.2 < r < 5 for Phe/Phe spectra."

    if away_shape_class == "Phe" and toward_shape_class == "SWe" and away_pad == "No LC" and toward_pad == "LC":
        return "C-X", "2b", "Away is Phe with no LC; toward is SWe with LC."

    if away_shape_class == "SWe" and toward_shape_class == "Phe" and away_pad == "LC" and toward_pad == "No LC":
        return "C-X", "2c", "Away is SWe with LC; toward is Phe with no LC."

    if away_pad == "LC" and toward_pad == "LC":
        return "C-T", "3", "Both away and toward PADs show loss cones."

    if away_shape_class == "Phe" and toward_shape_class == "SWe" and toward_pad == "No LC":
        return "O-D", "5a", "Away is Phe; toward is SWe with no LC."

    if away_shape_class == "SWe" and toward_shape_class == "Phe" and away_pad == "No LC":
        return "O-D", "5b", "Away is SWe with no LC; toward is Phe."

    if away_shape_class == "SWe" and toward_shape_class == "SWe" and away_pad == "LC" and toward_pad == "No LC":
        return "O-N", "6", "Both spectra are SWe; away PAD has LC and toward PAD has no LC."

    if away_shape_class == "SWe" and toward_shape_class == "SWe" and away_pad == "No LC":
        return "DP", "7a", "Both spectra are SWe and away PAD has no LC."

    return "unknown", "", "No lookup-table row matched the available shape/PAD/ratio values."


def build_topology_dataframe(
    shape_rows: list[dict],
    pad_df: pd.DataFrame,
    ratio_by_time: dict[float, dict],
    photoelectron_shape_threshold: float,
    max_pad_delta_seconds: float,
    loss_cone_pad_score_threshold: float,
    region_df: pd.DataFrame | None = None,
    max_region_delta_seconds: float = DEFAULT_MAX_REGION_ID_DELTA_SECONDS,
) -> pd.DataFrame:
    if region_df is not None and not region_df.empty:
        region_df = region_df.sort_values("time_unix", kind="stable").reset_index(
            drop=True
        )
    output_rows = []
    for row in shape_rows:
        sample_unix = finite_float(row.get("time_unix"))
        if not np.isfinite(sample_unix):
            continue
        pad_row, pad_delta = nearest_pad_row(pad_df, sample_unix, max_pad_delta_seconds)
        ratio_info = ratio_by_time.get(sample_unix, {})
        away_shape_value = finite_float(row.get("away_shape_parameter"))
        toward_shape_value = finite_float(row.get("toward_shape_parameter"))
        away_shape = shape_class(away_shape_value, photoelectron_shape_threshold)
        toward_shape = shape_class(toward_shape_value, photoelectron_shape_threshold)

        if pad_row is None:
            away_pad = "unknown"
            toward_pad = "unknown"
            pad_time_utc = ""
            pad_valid = False
            pad_reason = "missing_nearest_pad_score"
            toward_pad_score = float("nan")
            away_pad_score = float("nan")
        else:
            toward_pad_score = finite_float(pad_row.get("toward_pad_score"))
            away_pad_score = finite_float(pad_row.get("away_pad_score"))
            away_pad = pad_lc_class_from_score(away_pad_score, loss_cone_pad_score_threshold)
            toward_pad = pad_lc_class_from_score(toward_pad_score, loss_cone_pad_score_threshold)
            pad_time_utc = str(pad_row.get("time", ""))
            pad_valid = bool(pad_row.get("valid", False))
            pad_reason = str(pad_row.get("reason", ""))

        void = bool(ratio_info.get("electron_void", False))
        ratio = finite_float(ratio_info.get("at_ratio_35_60eV"))
        topology, subcase, reason = topology_from_table(
            away_shape,
            toward_shape,
            away_pad,
            toward_pad,
            ratio,
            void,
        )
        table_topology = topology
        table_subcase = subcase
        table_reason = reason

        region_row, region_delta = nearest_region_row(
            region_df,
            sample_unix,
            max_region_delta_seconds,
        )
        if region_row is None:
            region_id_value = float("nan")
            region_name = ""
            region_time_utc = ""
            region_confidence = float("nan")
            region_reason = "missing_nearest_region_id"
            region_geometry_only = False
            region_valid = False
        else:
            raw_region_id = finite_float(region_row.get("region_id"))
            region_id_value = (
                int(raw_region_id) if np.isfinite(raw_region_id) else float("nan")
            )
            region_name = str(region_row.get("region_name", ""))
            region_time_utc = str(region_row.get("time_utc", ""))
            region_confidence = finite_float(region_row.get("confidence"))
            region_reason = str(region_row.get("reason", ""))
            region_geometry_only = bool(region_row.get("geometry_only", False))
            region_valid = np.isfinite(raw_region_id)

        if region_id_value in {0, 1}:
            topology = "DP"
            subcase = "7b"
            reason = (
                f"region_id={region_id_value} ({region_name}) is assigned "
                "draped topology DP before the shape/PAD lookup table."
            )
            topology_source = "region_id_0_1_override"
        else:
            topology_source = "xu2019_shape_pad_table"

        output_rows.append(
            {
                "time_unix": sample_unix,
                "time_utc": str(row.get("time_utc", format_unix_time(sample_unix))),
                "topology": topology,
                "topology_label": "draped DP" if topology == "DP" else topology,
                "topology_subcase": subcase,
                "topology_reason": reason,
                "valid_topology": topology != "unknown",
                "topology_source": topology_source,
                "table_topology": table_topology,
                "table_topology_subcase": table_subcase,
                "table_topology_reason": table_reason,
                "region_id": region_id_value,
                "region_name": region_name,
                "region_id_time_utc": region_time_utc,
                "region_id_delta_seconds": region_delta,
                "region_id_confidence": region_confidence,
                "region_id_reason": region_reason,
                "region_id_geometry_only": region_geometry_only,
                "region_id_valid": bool(region_valid),
                "away_shape_class": away_shape,
                "toward_shape_class": toward_shape,
                "away_shape_parameter": away_shape_value,
                "toward_shape_parameter": toward_shape_value,
                "photoelectron_shape_threshold": photoelectron_shape_threshold,
                "away_pad_lc": away_pad,
                "toward_pad_lc": toward_pad,
                "away_pad_score": away_pad_score,
                "toward_pad_score": toward_pad_score,
                "loss_cone_pad_score_threshold": loss_cone_pad_score_threshold,
                "void": bool(void),
                "superthermal_electron_void": bool(void),
                "electron_void_target_energy_eV": finite_float(ratio_info.get("electron_void_target_energy_eV")),
                "electron_void_actual_energy_eV": finite_float(ratio_info.get("electron_void_actual_energy_eV")),
                "electron_void_flux": finite_float(ratio_info.get("electron_void_flux")),
                "electron_void_flux_threshold": finite_float(ratio_info.get("electron_void_flux_threshold")),
                "at_ratio_35_60eV": ratio,
                "away_flux_35_60eV": finite_float(ratio_info.get("away_flux_35_60eV")),
                "toward_flux_35_60eV": finite_float(ratio_info.get("toward_flux_35_60eV")),
                "pad_score_time_utc": pad_time_utc,
                "pad_score_delta_seconds": pad_delta,
                "pad_score_valid": pad_valid,
                "pad_score_reason": pad_reason,
                "shape_status": str(row.get("status", "")),
                "field_direction": str(row.get("field_direction", "")),
                "field_angle_deg": finite_float(row.get("field_angle_deg")),
                "spacecraft_potential_V": finite_float(row.get("spacecraft_potential_V")),
                "lpw_available": bool(row.get("lpw_available", False)),
                "energy_correction_applied": bool(row.get("energy_correction_applied", False)),
                "energy_correction_potential_V": finite_float(row.get("energy_correction_potential_V")),
                "ratio_status": str(ratio_info.get("status", "")),
                "ratio_swe_time_utc": str(ratio_info.get("ratio_swe_time_utc", "")),
                "ratio_swe_delta_seconds": finite_float(ratio_info.get("ratio_swe_delta_seconds")),
                "toward_ratio_source_direction": str(ratio_info.get("toward_ratio_source_direction", "")),
                "away_ratio_source_direction": str(ratio_info.get("away_ratio_source_direction", "")),
            }
        )
    return pd.DataFrame(output_rows)


def plot_topology_timeseries(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 3.5))
    if df.empty:
        ax.text(0.5, 0.5, "No topology samples", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        order = ["C-D", "C-X", "C-T", "C-V", "O-D", "O-N", "DP", "unknown"]
        y_map = {name: index for index, name in enumerate(order)}
        times = [parse_iso_timestamp(value) for value in df["time_utc"]]
        y = [y_map.get(str(value), y_map["unknown"]) for value in df["topology"]]
        ax.scatter(times, y, c=y, cmap="tab10", s=32, marker="s")
        ax.set_yticks(list(y_map.values()), list(y_map.keys()))
        ax.set_xlabel("UTC")
        ax.set_ylabel("Topology")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Determine MAVEN magnetic topology from shape parameters and PAD scores.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--photoelectron-shape-threshold",
        type=float,
        default=1.0,
        help="Shape parameter <= this value is classified as Phe; larger values are SWe.",
    )
    parser.add_argument("--cadence-seconds", type=float, default=0.0)
    parser.add_argument("--forward-pitch-max", type=float, default=30.0)
    parser.add_argument("--backward-pitch-min", type=float, default=150.0)
    parser.add_argument("--spacecraft-potential-min-flag", type=float, default=DEFAULT_SCPOT_MIN_FLAG)
    parser.add_argument("--max-lpw-delta-seconds", type=float, default=DEFAULT_MAX_LPW_DELTA_SECONDS)
    parser.add_argument("--max-mag-delta-seconds", type=float, default=DEFAULT_MAX_MAG_DELTA_SECONDS)
    parser.add_argument("--difference-mode", choices=("signed", "absolute", "squared"), default="absolute")
    parser.add_argument("--shape-energy-range", nargs=2, type=float, default=DEFAULT_SHAPE_ENERGY_RANGE_EV)
    parser.add_argument("--spectral-smoothing-window", type=int, default=5)
    parser.add_argument("--no-spectral-smoothing", action="store_true")
    parser.add_argument("--pad-energy-range", nargs=2, type=float, default=DEFAULT_PAD_ENERGY_RANGE_EV)
    parser.add_argument("--pad-energy-method", choices=("sum", "mean"), default="mean")
    parser.add_argument("--pad-group-size", type=int, default=4)
    parser.add_argument("--pad-keep-partial", action="store_true")
    parser.add_argument("--pad-threshold-sigma", type=float, default=2.0)
    parser.add_argument(
        "--loss-cone-pad-score-threshold",
        type=float,
        default=DEFAULT_LOSS_CONE_PAD_SCORE_THRESHOLD,
        help="A directional PAD score below this threshold is classified as LC.",
    )
    parser.add_argument("--max-pad-delta-seconds", type=float, default=6.0)
    parser.add_argument("--parallel-low", nargs=2, type=float, default=(0.0, 30.0))
    parser.add_argument("--perpendicular", nargs=2, type=float, default=(85.0, 95.0))
    parser.add_argument("--antiparallel-high", nargs=2, type=float, default=(150.0, 180.0))
    parser.add_argument("--ratio-energy-range", nargs=2, type=float, default=DEFAULT_RATIO_ENERGY_RANGE_EV)
    parser.add_argument(
        "--electron-void-energy",
        type=float,
        default=DEFAULT_ELECTRON_VOID_ENERGY_EV,
        help="Target electron energy used for the superthermal electron void test.",
    )
    parser.add_argument(
        "--electron-void-flux-threshold",
        type=float,
        default=DEFAULT_ELECTRON_VOID_FLUX_THRESHOLD,
        help="Flux below this absolute threshold is classified as a superthermal electron void.",
    )
    parser.add_argument(
        "--region-id-boundary-margin-km",
        type=float,
        default=DEFAULT_REGION_ID_BOUNDARY_MARGIN_KM,
        help="Bow-shock and MPB Unknown buffer used by the region_id classifier.",
    )
    parser.add_argument(
        "--max-region-id-delta-seconds",
        type=float,
        default=DEFAULT_MAX_REGION_ID_DELTA_SECONDS,
        help="Maximum allowed time difference when attaching region_id to a topology row.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    start = parse_iso_timestamp(args.start).astimezone(timezone.utc)
    end = parse_iso_timestamp(args.end).astimezone(timezone.utc)
    if end <= start:
        raise ValueError("--end must be later than --start.")

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = default_output_dir(Path(args.output_root).expanduser().resolve(), start, end)
    output_dir.mkdir(parents=True, exist_ok=True)

    shape_energy_range = (float(args.shape_energy_range[0]), float(args.shape_energy_range[1]))
    pad_energy_range = (float(args.pad_energy_range[0]), float(args.pad_energy_range[1]))
    ratio_energy_range = (float(args.ratio_energy_range[0]), float(args.ratio_energy_range[1]))
    spectral_smoothing_window = int(args.spectral_smoothing_window)
    if spectral_smoothing_window < 1:
        raise ValueError("--spectral-smoothing-window must be at least 1.")
    if spectral_smoothing_window % 2 == 0:
        spectral_smoothing_window += 1
    spectral_smoothing_enabled = not bool(args.no_spectral_smoothing) and spectral_smoothing_window > 1
    spectral_smoothing_points = spectral_smoothing_window if spectral_smoothing_enabled else 1

    template_path = Path(args.template).expanduser().resolve()
    template_energy, template_df = load_template(template_path)
    shape_rows, shape_skip_counts = compute_shape_parameters(
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
        spectral_smoothing_window_points=spectral_smoothing_points,
    )

    region_rows: list[dict] = []
    region_metadata: dict = {}
    region_target_times = [
        finite_float(row.get("time_unix"))
        for row in shape_rows
        if np.isfinite(finite_float(row.get("time_unix")))
    ]
    if region_target_times:
        region_rows, region_metadata = classify_region_interval(
            start=start,
            end=end,
            data_root=data_root,
            config=RegionClassifierConfig(
                boundary_margin_km=float(args.region_id_boundary_margin_km)
            ),
            target_times_unix=region_target_times,
        )
    region_df = pd.DataFrame(region_rows)
    region_dir = output_dir / "region_id"
    region_csv = write_region_csv(
        region_dir / "region_id_timeseries.csv",
        region_rows,
    )
    region_summary = write_region_summary_json(
        region_dir / "region_id_summary.json",
        region_metadata,
    )
    region_plot = (
        plot_region_ids(
            region_dir / "region_id_timeseries.png",
            region_rows,
        )
        if region_rows
        else None
    )

    shape_dir = output_dir / "shape_parameter_method"
    shape_csv = shape_dir / "shape_parameters.csv"
    shape_plot = shape_dir / "shape_parameters.png"
    write_shape_csv(shape_csv, shape_rows)
    plot_shape_parameters(shape_plot, shape_rows)

    bands = PitchAngleBands(
        parallel_low=tuple(float(v) for v in args.parallel_low),
        perpendicular=tuple(float(v) for v in args.perpendicular),
        antiparallel_high=tuple(float(v) for v in args.antiparallel_high),
    )
    pad_df = classify_pad_timeseries(
        start=start,
        end=end,
        data_root=data_root,
        energy_range_eV=pad_energy_range,
        energy_method=args.pad_energy_method,
        group_size=int(args.pad_group_size),
        keep_partial=bool(args.pad_keep_partial),
        threshold_sigma=float(args.pad_threshold_sigma),
        max_mag_delta_seconds=float(args.max_mag_delta_seconds),
        bands=bands,
    )
    pad_dir = output_dir / "PAD_score_method"
    pad_dir.mkdir(parents=True, exist_ok=True)
    pad_csv = pad_dir / "pad_score_classification.csv"
    pad_class_plot = pad_dir / "pad_score_classification.png"
    pad_score_plot = pad_dir / "pad_score_time_series.png"
    pad_df.to_csv(pad_csv, index=False)
    plot_time_series_classification(pad_df, pad_class_plot)
    plot_pad_score_time_series(pad_df, pad_score_plot, threshold_sigma=float(args.pad_threshold_sigma))

    ratio_by_time = compute_at_ratio_for_shape_rows(
        shape_rows,
        data_root=data_root,
        energy_range_eV=ratio_energy_range,
        forward_pitch_max_deg=float(args.forward_pitch_max),
        backward_pitch_min_deg=float(args.backward_pitch_min),
        electron_void_energy_eV=float(args.electron_void_energy),
        electron_void_flux_threshold=float(args.electron_void_flux_threshold),
    )
    topology_df = build_topology_dataframe(
        shape_rows,
        pad_df,
        ratio_by_time,
        photoelectron_shape_threshold=float(args.photoelectron_shape_threshold),
        max_pad_delta_seconds=float(args.max_pad_delta_seconds),
        loss_cone_pad_score_threshold=float(args.loss_cone_pad_score_threshold),
        region_df=region_df,
        max_region_delta_seconds=float(args.max_region_id_delta_seconds),
    )

    topology_csv = output_dir / "magnetic_topology_classification.csv"
    topology_plot = output_dir / "magnetic_topology_timeseries.png"
    topology_df.to_csv(topology_csv, index=False)
    plot_topology_timeseries(topology_df, topology_plot)

    summary = {
        "start": start.isoformat(timespec="seconds"),
        "end": end.isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "template": str(template_path),
        "photoelectron_shape_threshold": float(args.photoelectron_shape_threshold),
        "shape_energy_range_eV": list(shape_energy_range),
        "pad_energy_range_eV": list(pad_energy_range),
        "ratio_energy_range_eV": list(ratio_energy_range),
        "loss_cone_pad_score_threshold": float(args.loss_cone_pad_score_threshold),
        "loss_cone_rule": "A direction is classified as LC when its PAD score is strictly below loss_cone_pad_score_threshold; otherwise a finite score is No LC.",
        "electron_void_energy_eV": float(args.electron_void_energy),
        "electron_void_flux_threshold": float(args.electron_void_flux_threshold),
        "electron_void_rule": "The pitch-angle-mean differential energy flux at the available SWE energy channel nearest electron_void_energy_eV is below electron_void_flux_threshold.",
        "spectral_smoothing": {
            "enabled": spectral_smoothing_enabled,
            "window_points": spectral_smoothing_points,
        },
        "lookup_table_note": "A=away, T=toward, LC=loss cone, Phe=photoelectron-like shape, SWe=solar-wind/backscattered-like shape. region_id 0 or 1 overrides the shape/PAD result as draped DP case 7b.",
        "region_id_rule": {
            "override_region_ids": [0, 1],
            "topology": "DP",
            "topology_label": "draped DP",
            "topology_subcase": "7b",
            "boundary_margin_km": float(args.region_id_boundary_margin_km),
            "max_delta_seconds": float(args.max_region_id_delta_seconds),
            "metadata": region_metadata,
        },
        "shape_skip_counts": shape_skip_counts,
        "rows": int(len(topology_df)),
        "valid_topology_rows": int(topology_df["valid_topology"].sum()) if not topology_df.empty else 0,
        "topology_counts": topology_df["topology"].value_counts(dropna=False).to_dict() if not topology_df.empty else {},
        "region_id_override_rows": (
            int((topology_df["topology_source"] == "region_id_0_1_override").sum())
            if not topology_df.empty
            else 0
        ),
        "outputs": {
            "topology_csv": str(topology_csv),
            "topology_plot": str(topology_plot),
            "shape_csv": str(shape_csv),
            "shape_plot": str(shape_plot),
            "pad_csv": str(pad_csv),
            "pad_classification_plot": str(pad_class_plot),
            "pad_score_plot": str(pad_score_plot),
            "region_id_csv": str(region_csv),
            "region_id_plot": "" if region_plot is None else str(region_plot),
            "region_id_summary": str(region_summary),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
