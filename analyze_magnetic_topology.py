from __future__ import annotations
"""
Interval-based magnetic topology analysis.

This is the main science workflow in the repository.

High-level flow:
1. Resolve or download the required daily SWE / MAG / STATIC files
2. Sample a time interval at `step_seconds`
3. Extract directional electron spectra around each sample
4. Apply rule-based topology classification
5. Build a richer JSON product for later visualization in the HTML viewer
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cdflib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from download_maven_data import DEFAULT_DATA_ROOT, download_product_for_day, build_session, parse_iso_timestamp
from mars_crustal_model import (
    DEFAULT_MODEL_ROOT,
    ensure_morschhauser_coefficients,
    evaluate_morschhauser_field_mso,
    load_morschhauser_coefficients,
)
from process_maven_spectra import (
    build_mag_times,
    format_unix_time,
    infer_daily_file,
    load_pad_data,
    locate_nearest_index,
    pick_first_variable,
    parse_mag_sts,
    unix_seconds_from_cdf_epoch,
    unix_seconds_from_numeric_time,
)


# User-editable configuration.
CONFIG = {
    "start_time": "2024-11-07T00:00:00",
    "end_time": "2024-11-08T00:00:00",
    "step_seconds": 1,
    "auto_download_missing_data": True,
    "data_root": str(DEFAULT_DATA_ROOT),
    "model_root": str(DEFAULT_MODEL_ROOT),
    "output_root": str(Path("outputs") / "magnetic_topology"),
}


# Topology criteria are centralized here so the rules are explicit and easy to adjust.
TOPOLOGY_RULES = {
    "pitch_angle": {
        "parallel_max_deg": 60.0,
        "anti_parallel_min_deg": 120.0,
    },
    "knee": {
        "pre_band_eV": (30.0, 55.0),
        "knee_band_eV": (50.0, 70.0),
        "post_band_eV": (70.0, 120.0),
        "min_slope_change": 0.6,
        "min_flux_ratio": 1.15,
    },
    "auger_peak": {
        "left_band_eV": (120.0, 180.0),
        "center_band_eV": (180.0, 320.0),
        "right_band_eV": (320.0, 420.0),
        "min_peak_ratio": 1.20,
    },
    "dropoff": {
        "reference_band_eV": (250.0, 450.0),
        "high_band_eV": (550.0, 900.0),
        "min_flux_ratio": 1.8,
    },
    "classification": {
        "closed_min_score_per_direction": 2,
        "open_max_score_per_direction": 1,
    },
}


MARS_RADIUS_KM = 3389.5
TOPOLOGY_COLORS = {
    "closed": "#d62728",
    "open": "#1f77b4",
    "ambiguous": "#7f7f7f",
}


@dataclass(frozen=True)
class DirectionFeatures:
    """Per-direction feature flags used by the topology rules."""
    knee_present: bool
    auger_present: bool
    dropoff_present: bool
    feature_score: int


@dataclass(frozen=True)
class TopologySample:
    """One classified sample along the orbit track."""
    target_time: str
    swe_time: str
    mag_time_ss: str
    mag_time_pc: str
    topology: str
    forward_features: DirectionFeatures
    backward_features: DirectionFeatures
    magnetic_field_nT: list[float]
    energy_eV: list[float]
    forward_flux: list[float]
    backward_flux: list[float]
    altitude_km: float
    altitude_rm: float
    position_km: list[float]
    position_rm: list[float]
    positions_by_frame_km: dict[str, list[float]]
    positions_by_frame_rm: dict[str, list[float]]


def sanitize_for_json(value):
    """Convert NumPy-heavy nested objects into strict JSON-safe values.

    Browsers reject `NaN`/`Inf`, so this function is a guardrail before writing
    `topology_summary.json`.
    """
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def iter_days(start: datetime, end: datetime) -> list[date]:
    """Expand one interval into the list of UTC dates it spans."""
    current = start.date()
    last = end.date()
    days: list[date] = []
    while current <= last:
        days.append(current)
        current += timedelta(days=1)
    return days


def finite_positive(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array) & (array > 0.0)]


def band_mask(energy: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return (energy >= lower) & (energy <= upper)


def median_flux(energy: np.ndarray, flux: np.ndarray, lower: float, upper: float) -> float:
    masked = finite_positive(flux[band_mask(energy, lower, upper)])
    if masked.size == 0:
        return float("nan")
    return float(np.nanmedian(masked))


def log_slope(energy: np.ndarray, flux: np.ndarray, lower: float, upper: float) -> float:
    mask = band_mask(energy, lower, upper)
    selected_energy = np.asarray(energy[mask], dtype=float)
    selected_flux = np.asarray(flux[mask], dtype=float)
    usable_mask = np.isfinite(selected_energy) & np.isfinite(selected_flux) & (selected_energy > 0.0) & (selected_flux > 0.0)
    if np.count_nonzero(usable_mask) < 3:
        return float("nan")
    x = np.log10(selected_energy[usable_mask])
    y = np.log10(selected_flux[usable_mask])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def detect_knee(energy: np.ndarray, flux: np.ndarray) -> bool:
    """Detect the photoelectron knee around the configured energy bands."""
    rules = TOPOLOGY_RULES["knee"]
    pre_lower, pre_upper = rules["pre_band_eV"]
    knee_lower, knee_upper = rules["knee_band_eV"]
    post_lower, post_upper = rules["post_band_eV"]
    pre_slope = log_slope(energy, flux, pre_lower, pre_upper)
    post_slope = log_slope(energy, flux, post_lower, post_upper)
    knee_flux = median_flux(energy, flux, knee_lower, knee_upper)
    post_flux = median_flux(energy, flux, post_lower, post_upper)
    if np.isnan(pre_slope) or np.isnan(post_slope) or np.isnan(knee_flux) or np.isnan(post_flux):
        return False
    return (post_slope < pre_slope - rules["min_slope_change"]) and (knee_flux > post_flux * rules["min_flux_ratio"])


def detect_auger_peak(energy: np.ndarray, flux: np.ndarray) -> bool:
    """Detect an Auger-like enhancement in the configured energy band."""
    rules = TOPOLOGY_RULES["auger_peak"]
    center = finite_positive(flux[band_mask(energy, *rules["center_band_eV"])])
    left = finite_positive(flux[band_mask(energy, *rules["left_band_eV"])])
    right = finite_positive(flux[band_mask(energy, *rules["right_band_eV"])])
    if center.size < 2 or left.size == 0 or right.size == 0:
        return False
    center_peak = float(np.nanmax(center))
    shoulder_level = float(np.nanmax(np.concatenate([left, right])))
    return center_peak > shoulder_level * rules["min_peak_ratio"]


def detect_dropoff(energy: np.ndarray, flux: np.ndarray) -> bool:
    """Detect high-energy flux suppression relative to a mid-energy reference."""
    rules = TOPOLOGY_RULES["dropoff"]
    mid_flux = median_flux(energy, flux, *rules["reference_band_eV"])
    high_flux = median_flux(energy, flux, *rules["high_band_eV"])
    if np.isnan(mid_flux) or np.isnan(high_flux) or high_flux <= 0.0:
        return False
    return (mid_flux / high_flux) >= rules["min_flux_ratio"]


def extract_directional_flux(pad_data: dict, time_index: int) -> tuple[np.ndarray, np.ndarray]:
    flux_at_time = np.asarray(pad_data["flux"][time_index], dtype=float)
    pitch = np.asarray(pad_data["pitch"], dtype=float)
    pitch_rules = TOPOLOGY_RULES["pitch_angle"]

    if pitch.ndim == 1:
        forward_mask = pitch < pitch_rules["parallel_max_deg"]
        backward_mask = pitch > pitch_rules["anti_parallel_min_deg"]
        return (
            np.nanmean(flux_at_time[forward_mask, :], axis=0),
            np.nanmean(flux_at_time[backward_mask, :], axis=0),
        )

    pitch_at_time = pitch[time_index]
    forward_flux = np.full(flux_at_time.shape[1], np.nan, dtype=float)
    backward_flux = np.full(flux_at_time.shape[1], np.nan, dtype=float)
    for energy_index in range(flux_at_time.shape[1]):
        pitch_column = pitch_at_time[:, energy_index]
        forward_mask = pitch_column < pitch_rules["parallel_max_deg"]
        backward_mask = pitch_column > pitch_rules["anti_parallel_min_deg"]
        if np.any(forward_mask):
            forward_values = flux_at_time[forward_mask, energy_index]
            if np.any(np.isfinite(forward_values)):
                forward_flux[energy_index] = np.nanmean(forward_values)
        if np.any(backward_mask):
            backward_values = flux_at_time[backward_mask, energy_index]
            if np.any(np.isfinite(backward_values)):
                backward_flux[energy_index] = np.nanmean(backward_values)
    return forward_flux, backward_flux


def analyze_direction(energy: np.ndarray, flux: np.ndarray) -> DirectionFeatures:
    knee_present = detect_knee(energy, flux)
    auger_present = detect_auger_peak(energy, flux)
    dropoff_present = detect_dropoff(energy, flux)
    feature_score = int(knee_present) + int(auger_present) + int(dropoff_present)
    return DirectionFeatures(
        knee_present=knee_present,
        auger_present=auger_present,
        dropoff_present=dropoff_present,
        feature_score=feature_score,
    )


def classify_topology(forward: DirectionFeatures, backward: DirectionFeatures) -> str:
    rules = TOPOLOGY_RULES["classification"]
    if (
        forward.feature_score >= rules["closed_min_score_per_direction"]
        and backward.feature_score >= rules["closed_min_score_per_direction"]
    ):
        return "closed"
    if (
        forward.feature_score <= rules["open_max_score_per_direction"]
        and backward.feature_score <= rules["open_max_score_per_direction"]
    ):
        return "open"
    return "ambiguous"


def load_mag_day(path: Path) -> dict:
    parsed = parse_mag_sts(path)
    times = build_mag_times(parsed["columns"], parsed["data"])
    columns = parsed["columns"]
    data = parsed["data"]
    return {
        "path": path,
        "times": times,
        "data": data,
        "columns": columns,
        "b_indices": [columns.index("OB_B.X"), columns.index("OB_B.Y"), columns.index("OB_B.Z")],
        "pos_indices": [columns.index("POSN.X"), columns.index("POSN.Y"), columns.index("POSN.Z")],
    }


def normalize_cdf_times(cdf: cdflib.CDF, path: Path) -> np.ndarray:
    time_values = pick_first_variable(cdf, ["epoch", "time_unix", "time_met"])
    if time_values is None:
        raise KeyError(f"No usable time variable was found in {path.name}.")
    if time_values.dtype.kind in {"i", "u"} and np.nanmedian(time_values) > 1e12:
        return unix_seconds_from_cdf_epoch(time_values)
    return unix_seconds_from_numeric_time(time_values)


def infer_axis_index(shape: tuple[int, ...], expected_length: int, label: str, variable_name: str) -> int:
    matches = [idx for idx, axis_size in enumerate(shape) if axis_size == expected_length]
    if not matches:
        raise ValueError(f"Could not map {label} axis for {variable_name}. shape={shape}, expected size={expected_length}.")
    return matches[0]


def median_valid(values: np.ndarray, axis: int) -> np.ndarray:
    return np.nanmedian(np.asarray(values, dtype=float), axis=axis)


def average_positive_flux(values: np.ndarray, axis: int) -> np.ndarray:
    flux = np.asarray(values, dtype=float)
    valid = np.isfinite(flux) & (flux > 0.0)
    summed = np.sum(np.where(valid, flux, 0.0), axis=axis)
    counts = np.sum(valid, axis=axis)
    averaged = np.divide(summed, counts, out=np.zeros_like(summed, dtype=float), where=counts > 0)
    return np.nan_to_num(averaged, nan=0.0, posinf=0.0, neginf=0.0)


def load_static_context(path: Path, start: datetime, end: datetime) -> dict | None:
    cdf = cdflib.CDF(str(path))
    times = normalize_cdf_times(cdf, path)
    time_mask = (times >= start.timestamp()) & (times <= end.timestamp())
    if not np.any(time_mask):
        return None

    flux = pick_first_variable(cdf, ["eflux"])
    energy = pick_first_variable(cdf, ["energy"])
    mass_arr = pick_first_variable(cdf, ["mass_arr"])
    if flux is None or energy is None or mass_arr is None:
        return None

    flux = np.asarray(flux, dtype=float)
    energy = np.asarray(energy, dtype=float)
    mass_arr = np.asarray(mass_arr, dtype=float)
    if flux.ndim != 3 or energy.ndim != 3 or mass_arr.ndim != 3:
        return None

    # STATIC c6-32e64m stores eflux with shape (time, mass_bin, energy_bin).
    # The coordinate arrays have shape (mass_bin, energy_bin, sweep_table), so we
    # collapse over the unused dimensions to derive nominal axes.
    energy_axis_values = np.nanmedian(energy, axis=(0, 2))
    mass_axis_values = np.nanmedian(mass_arr, axis=(1, 2))
    selected_flux = flux[time_mask]

    energy_spectrogram = average_positive_flux(selected_flux, axis=1)
    mass_spectrogram = average_positive_flux(selected_flux, axis=2)
    selected_times = times[time_mask]

    return {
        "times_unix": selected_times.tolist(),
        "energy_eV": np.asarray(energy_axis_values, dtype=float).tolist(),
        "mass_amu": np.asarray(mass_axis_values, dtype=float).tolist(),
        "energy_eflux": energy_spectrogram.tolist(),
        "mass_eflux": mass_spectrogram.tolist(),
        "source_file": str(path),
    }


def derive_pitch_bins(pad_data: dict, indices: np.ndarray, band_mask_energy: np.ndarray) -> np.ndarray:
    pitch = np.asarray(pad_data["pitch"], dtype=float)
    if pitch.ndim == 1:
        return pitch
    band_pitch = pitch[indices][:, :, band_mask_energy]
    return np.nanmedian(band_pitch, axis=(0, 2))


def build_swe_context(pad_data: dict, start: datetime, end: datetime) -> dict | None:
    times = np.asarray(pad_data["times"], dtype=float)
    time_mask = (times >= start.timestamp()) & (times <= end.timestamp())
    if not np.any(time_mask):
        return None

    indices = np.where(time_mask)[0]
    flux = np.asarray(pad_data["flux"], dtype=float)[indices]
    energy = np.asarray(pad_data["energy"], dtype=float)
    band_mask_energy = (energy >= 111.0) & (energy <= 140.0)
    if not np.any(band_mask_energy):
        band_mask_energy = (energy >= 100.0) & (energy <= 150.0)

    omni_spectrum = np.nanmean(flux, axis=1)
    band_flux = flux[:, :, band_mask_energy]
    if band_flux.shape[2] == 0:
        pad_band = np.full((flux.shape[0], flux.shape[1]), np.nan, dtype=float)
    else:
        valid_counts = np.sum(np.isfinite(band_flux), axis=2)
        band_sum = np.nansum(band_flux, axis=2)
        pad_band = np.divide(
            band_sum,
            valid_counts,
            out=np.full_like(band_sum, np.nan, dtype=float),
            where=valid_counts > 0,
        )
    pitch_bins = derive_pitch_bins(pad_data, indices, band_mask_energy)

    return {
        "times_unix": times[indices].tolist(),
        "energy_eV": energy.tolist(),
        "pitch_deg": np.asarray(pitch_bins, dtype=float).tolist(),
        "omni_eflux": omni_spectrum.tolist(),
        "pad_111_140_eflux": pad_band.tolist(),
    }


def build_mag_context(mag_data_ss: dict, start: datetime, end: datetime) -> dict | None:
    times = np.asarray(mag_data_ss["times"], dtype=float)
    time_mask = (times >= start.timestamp()) & (times <= end.timestamp())
    if not np.any(time_mask):
        return None
    selected = np.asarray(mag_data_ss["data"], dtype=float)[time_mask]
    bx = selected[:, mag_data_ss["b_indices"][0]]
    by = selected[:, mag_data_ss["b_indices"][1]]
    bz = selected[:, mag_data_ss["b_indices"][2]]
    bmag = np.sqrt(bx**2 + by**2 + bz**2)
    return {
        "times_unix": times[time_mask].tolist(),
        "bx_nT": bx.tolist(),
        "by_nT": by.tolist(),
        "bz_nT": bz.tolist(),
        "bmag_nT": bmag.tolist(),
        "source_file": str(mag_data_ss["path"]),
    }


def build_model_context(
    mag_data_pc: dict,
    start: datetime,
    end: datetime,
    model_root: Path,
) -> dict:
    times = np.asarray(mag_data_pc["times"], dtype=float)
    time_mask = (times >= start.timestamp()) & (times <= end.timestamp())
    if not np.any(time_mask):
        return {"available": False, "message": "No MAG PC samples were found in the requested window."}

    selected_times = times[time_mask]
    selected_data = np.asarray(mag_data_pc["data"], dtype=float)[time_mask]
    positions = selected_data[:, mag_data_pc["pos_indices"]].astype(float)

    max_points = 720
    stride = max(1, int(np.ceil(len(selected_times) / max_points)))
    sampled_times = selected_times[::stride]
    sampled_positions = positions[::stride]

    try:
        coefficient_path = ensure_morschhauser_coefficients(model_root)
        coeffs = load_morschhauser_coefficients(str(coefficient_path))
    except Exception as exc:
        return {
            "available": False,
            "message": f"Failed to prepare Morschhauser coefficients: {exc}",
        }

    bx_values: list[float] = []
    by_values: list[float] = []
    bz_values: list[float] = []
    for unix_time, position_pc_km in zip(sampled_times, sampled_positions):
        field_mso = evaluate_morschhauser_field_mso(position_pc_km, float(unix_time), coeffs)
        bx_values.append(float(field_mso[0]))
        by_values.append(float(field_mso[1]))
        bz_values.append(float(field_mso[2]))

    return {
        "available": True,
        "model_name": "Morschhauser et al. (2014)",
        "times_unix": sampled_times.tolist(),
        "bx_nT": bx_values,
        "by_nT": by_values,
        "bz_nT": bz_values,
        "source_file": str(coefficient_path),
        "downsample_stride": stride,
    }


def build_context_overview(
    start: datetime,
    end: datetime,
    resolved_files: dict[date, dict[str, Path]],
    model_root: Path,
) -> dict:
    mag_context_parts: list[dict] = []
    swe_context_parts: list[dict] = []
    static_context_parts: list[dict] = []
    model_context_parts: list[dict] = []

    for day in iter_days(start, end):
        files = resolved_files[day]
        day_start = max(start, datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc))
        day_end = min(end, datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc))

        pad_data = load_pad_data(files["pad"])
        swe_context = build_swe_context(pad_data, day_start, day_end)
        if swe_context:
            swe_context_parts.append(swe_context)

        mag_data_ss = load_mag_day(files["mag_ss"])
        mag_context = build_mag_context(mag_data_ss, day_start, day_end)
        if mag_context:
            mag_context_parts.append(mag_context)

        mag_data_pc = load_mag_day(files["mag_pc"])
        model_context = build_model_context(mag_data_pc, day_start, day_end, model_root)
        model_context_parts.append(model_context)

        if "sta_c6" in files:
            static_context = load_static_context(files["sta_c6"], day_start, day_end)
            if static_context:
                static_context_parts.append(static_context)

    def concat_timeseries(parts: list[dict], keys: list[str]) -> dict | None:
        if not parts:
            return None
        merged: dict[str, list] = {key: [] for key in keys}
        static_keys = [key for key in parts[0].keys() if key not in merged]
        for part in parts:
            for key in keys:
                merged[key].extend(part[key])
        for key in static_keys:
            merged[key] = parts[0][key]
        return merged

    return {
        "window_seconds": 600,
        "static": concat_timeseries(
            static_context_parts,
            ["times_unix", "energy_eflux", "mass_eflux"],
        ),
        "mag": concat_timeseries(
            mag_context_parts,
            ["times_unix", "bx_nT", "by_nT", "bz_nT", "bmag_nT"],
        ),
        "swe": concat_timeseries(
            swe_context_parts,
            ["times_unix", "omni_eflux", "pad_111_140_eflux"],
        ),
        "model_b_mso": next((item for item in model_context_parts if item.get("available")), model_context_parts[0] if model_context_parts else {"available": False, "message": "No model context was built."}),
    }


def extract_position_from_mag(target_time: datetime, mag_data: dict) -> tuple[np.ndarray, str]:
    mag_index = locate_nearest_index(mag_data["times"], target_time)
    mag_row = mag_data["data"][mag_index]
    position_km = mag_row[mag_data["pos_indices"]].astype(float)
    return position_km, format_unix_time(mag_data["times"][mag_index])


def sample_from_time(
    target_time: datetime,
    pad_data: dict,
    mag_data_ss: dict,
    mag_data_pc: dict,
    pad_time_index: int,
) -> TopologySample:
    energy = np.asarray(pad_data["energy"], dtype=float)
    forward_flux, backward_flux = extract_directional_flux(pad_data, pad_time_index)
    forward_features = analyze_direction(energy, forward_flux)
    backward_features = analyze_direction(energy, backward_flux)
    topology = classify_topology(forward_features, backward_features)

    mag_index_ss = locate_nearest_index(mag_data_ss["times"], target_time)
    mag_row_ss = mag_data_ss["data"][mag_index_ss]
    magnetic_field = mag_row_ss[mag_data_ss["b_indices"]].astype(float)
    position_km_ss, mag_time_ss = extract_position_from_mag(target_time, mag_data_ss)
    position_km_pc, mag_time_pc = extract_position_from_mag(target_time, mag_data_pc)
    position_rm_ss = position_km_ss / MARS_RADIUS_KM
    position_rm_pc = position_km_pc / MARS_RADIUS_KM
    altitude_km = float(np.linalg.norm(position_km_ss) - MARS_RADIUS_KM)
    altitude_rm = altitude_km / MARS_RADIUS_KM

    return TopologySample(
        target_time=target_time.isoformat(timespec="seconds"),
        swe_time=format_unix_time(pad_data["times"][pad_time_index]),
        mag_time_ss=mag_time_ss,
        mag_time_pc=mag_time_pc,
        topology=topology,
        forward_features=forward_features,
        backward_features=backward_features,
        magnetic_field_nT=magnetic_field.tolist(),
        energy_eV=energy.tolist(),
        forward_flux=forward_flux.tolist(),
        backward_flux=backward_flux.tolist(),
        altitude_km=altitude_km,
        altitude_rm=altitude_rm,
        position_km=position_km_ss.tolist(),
        position_rm=position_rm_ss.tolist(),
        positions_by_frame_km={
            "ss": position_km_ss.tolist(),
            "pc": position_km_pc.tolist(),
        },
        positions_by_frame_rm={
            "ss": position_rm_ss.tolist(),
            "pc": position_rm_pc.tolist(),
        },
    )


def select_time_indices(times: np.ndarray, start: datetime, end: datetime, step_seconds: int) -> list[int]:
    selected: list[int] = []
    last_kept = float("-inf")
    for index, value in enumerate(np.asarray(times, dtype=float)):
        if value < start.timestamp() or value > end.timestamp():
            continue
        if value - last_kept >= step_seconds:
            selected.append(index)
            last_kept = value
    return selected


def build_segments(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0, 2, 2))
    return np.stack([points[:-1], points[1:]], axis=1)


def plot_topology_trajectory(samples: list[TopologySample], output_path: Path) -> None:
    positions = np.asarray([sample.position_rm for sample in samples], dtype=float)
    colors = [TOPOLOGY_COLORS[sample.topology] for sample in samples]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    projections = [
        (positions[:, [0, 1]], "X (R_M)", "Y (R_M)", "MAVEN XY trajectory"),
        (positions[:, [0, 2]], "X (R_M)", "Z (R_M)", "MAVEN XZ trajectory"),
    ]

    for ax, (points, xlabel, ylabel, title) in zip(axes, projections):
        segments = build_segments(points)
        if len(segments):
            segment_colors = [TOPOLOGY_COLORS[sample.topology] for sample in samples[:-1]]
            collection = LineCollection(segments, colors=segment_colors, linewidths=2.0, alpha=0.85)
            ax.add_collection(collection)
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=18, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    for label, color in TOPOLOGY_COLORS.items():
        axes[1].scatter([], [], c=color, s=28, label=label)
    axes[1].legend(loc="best", frameon=False)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_output_dir(start: datetime, end: datetime, output_root: Path) -> Path:
    directory = output_root / f"{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_day_files(day: date, data_root: Path, auto_download_missing_data: bool) -> dict[str, Path]:
    day_dt = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    try:
        return {
            "pad": infer_daily_file(data_root, "swe", "svypad", day_dt, "cdf"),
            "sta_c6": infer_daily_file(data_root, "sta", "32e64m", day_dt, "cdf"),
            "mag_ss": infer_daily_file(data_root, "mag", "ss1s", day_dt, "sts"),
            "mag_pc": infer_daily_file(data_root, "mag", "pc1s", day_dt, "sts"),
        }
    except FileNotFoundError:
        if not auto_download_missing_data:
            raise

    session = build_session()
    from download_maven_data import PIPELINE_PRODUCTS  # local import keeps the top import list shorter

    downloaded: dict[str, Path] = {}
    for spec in PIPELINE_PRODUCTS:
        if spec.instrument == "swe":
            key = "pad"
        elif spec.instrument == "sta":
            key = "sta_c6"
        elif "pc1s" in spec.aliases:
            key = "mag_pc"
        else:
            key = "mag_ss"
        downloaded[key] = download_product_for_day(session=session, spec=spec, day=day, data_root=data_root)
    return downloaded


def resolve_daily_files(
    start: datetime,
    end: datetime,
    data_root: Path,
    pad_file: Path | None,
    mag_file: Path | None,
    auto_download_missing_data: bool,
) -> dict[date, dict[str, Path]]:
    days = iter_days(start, end)
    if len(days) > 1 and (pad_file or mag_file):
        raise ValueError("Explicit --pad-file/--mag-file currently support only single-day intervals.")

    resolved: dict[date, dict[str, Path]] = {}
    for day in days:
        if pad_file or mag_file:
            resolved[day] = {
                "pad": pad_file if pad_file else ensure_day_files(day, data_root, auto_download_missing_data)["pad"],
                "sta_c6": ensure_day_files(day, data_root, auto_download_missing_data)["sta_c6"],
                "mag_ss": mag_file if mag_file else ensure_day_files(day, data_root, auto_download_missing_data)["mag_ss"],
                "mag_pc": ensure_day_files(day, data_root, auto_download_missing_data)["mag_pc"],
            }
        else:
            resolved[day] = ensure_day_files(day, data_root, auto_download_missing_data)
    return resolved


def analyze_interval(
    start: datetime,
    end: datetime,
    data_root: Path,
    model_root: Path,
    output_root: Path,
    step_seconds: int,
    auto_download_missing_data: bool,
    pad_file: Path | None = None,
    mag_file: Path | None = None,
) -> tuple[list[TopologySample], Path]:
    resolved_files = resolve_daily_files(
        start=start,
        end=end,
        data_root=data_root,
        pad_file=pad_file,
        mag_file=mag_file,
        auto_download_missing_data=auto_download_missing_data,
    )
    output_dir = build_output_dir(start, end, output_root)
    all_samples: list[TopologySample] = []

    for day in iter_days(start, end):
        files = resolved_files[day]
        pad_data = load_pad_data(files["pad"])
        mag_data_ss = load_mag_day(files["mag_ss"])
        mag_data_pc = load_mag_day(files["mag_pc"])
        day_start = max(start, datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc))
        day_end = min(end, datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc))
        indices = select_time_indices(pad_data["times"], day_start, day_end, step_seconds)
        for pad_time_index in indices:
            target_time = datetime.fromtimestamp(float(pad_data["times"][pad_time_index]), tz=timezone.utc)
            all_samples.append(sample_from_time(target_time, pad_data, mag_data_ss, mag_data_pc, pad_time_index))

    if not all_samples:
        raise ValueError("No SWE samples were found in the requested interval.")

    figure_path = output_dir / "magnetic_topology_trajectory.png"
    plot_topology_trajectory(all_samples, figure_path)
    context_overview = build_context_overview(start, end, resolved_files, model_root)

    summary = {
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": end.isoformat(timespec="seconds"),
        "step_seconds": step_seconds,
        "auto_download_missing_data": auto_download_missing_data,
        "available_coordinate_frames": {
            "ss": "Sun-State / MSO",
            "pc": "Planetocentric / Mars body-fixed",
        },
        "default_coordinate_frame": "ss",
        "topology_rules": TOPOLOGY_RULES,
        "context_overview": context_overview,
        "counts": {
            "closed": sum(sample.topology == "closed" for sample in all_samples),
            "open": sum(sample.topology == "open" for sample in all_samples),
            "ambiguous": sum(sample.topology == "ambiguous" for sample in all_samples),
        },
        "samples": [asdict(sample) for sample in all_samples],
    }
    summary = sanitize_for_json(summary)
    (output_dir / "topology_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return all_samples, figure_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer magnetic topology from MAVEN electron spectra and plot the spacecraft trajectory.")
    parser.add_argument("--start", help="UTC start time, for example 2024-11-07T02:00:00.")
    parser.add_argument("--end", help="UTC end time, for example 2024-11-07T03:00:00.")
    parser.add_argument("--step-seconds", type=int, help="Minimum spacing between sampled SWE spectra.")
    parser.add_argument("--pad-file", help="Optional SWE PAD CDF file.")
    parser.add_argument("--mag-file", help="Optional MAG STS file.")
    parser.add_argument("--data-root", help="Root directory for MAVEN data.")
    parser.add_argument("--model-root", help="Directory used to store Mars crustal field model coefficients.")
    parser.add_argument("--output-root", help="Directory used to store the trajectory figure and JSON summary.")
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable automatic download when the requested day is missing locally.",
    )
    return parser


def resolve_runtime_config(args: argparse.Namespace) -> dict:
    start_raw = args.start if args.start else CONFIG["start_time"]
    end_raw = args.end if args.end else CONFIG["end_time"]
    step_seconds = args.step_seconds if args.step_seconds is not None else CONFIG["step_seconds"]
    data_root = Path(args.data_root if args.data_root else CONFIG["data_root"]).expanduser().resolve()
    model_root = Path(args.model_root if args.model_root else CONFIG["model_root"]).expanduser().resolve()
    output_root = Path(args.output_root if args.output_root else CONFIG["output_root"]).expanduser().resolve()
    auto_download_missing_data = (
        False if args.no_auto_download else bool(CONFIG["auto_download_missing_data"])
    )

    start = parse_iso_timestamp(start_raw)
    end = parse_iso_timestamp(end_raw)
    if end <= start:
        raise ValueError("end_time/--end must be later than start_time/--start.")
    if step_seconds <= 0:
        raise ValueError("step_seconds/--step-seconds must be positive.")

    return {
        "start": start,
        "end": end,
        "step_seconds": step_seconds,
        "data_root": data_root,
        "model_root": model_root,
        "output_root": output_root,
        "auto_download_missing_data": auto_download_missing_data,
        "pad_file": Path(args.pad_file).expanduser().resolve() if args.pad_file else None,
        "mag_file": Path(args.mag_file).expanduser().resolve() if args.mag_file else None,
    }


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    runtime = resolve_runtime_config(args)

    samples, figure_path = analyze_interval(
        start=runtime["start"],
        end=runtime["end"],
        data_root=runtime["data_root"],
        model_root=runtime["model_root"],
        output_root=runtime["output_root"],
        step_seconds=runtime["step_seconds"],
        auto_download_missing_data=runtime["auto_download_missing_data"],
        pad_file=runtime["pad_file"],
        mag_file=runtime["mag_file"],
    )
    print(
        json.dumps(
            sanitize_for_json(
                {
                "start_time": runtime["start"].isoformat(timespec="seconds"),
                "end_time": runtime["end"].isoformat(timespec="seconds"),
                "step_seconds": runtime["step_seconds"],
                "auto_download_missing_data": runtime["auto_download_missing_data"],
                "topology_rules": TOPOLOGY_RULES,
                "figure_path": str(figure_path),
                "sample_count": len(samples),
                "closed": sum(sample.topology == "closed" for sample in samples),
                "open": sum(sample.topology == "open" for sample in samples),
                "ambiguous": sum(sample.topology == "ambiguous" for sample in samples),
                }
            ),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
