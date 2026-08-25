from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import cdflib
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
from process_maven_spectra import find_axis_by_length, infer_daily_file, infer_pitch_axis_size, load_pad_data
from identify_magnetic_topology.magnetic_field_direction import (
    load_magnetic_geometry_interval,
    nearest_magnetic_field_direction,
)


MARS_RADIUS_KM = 3389.5
DEFAULT_OUTPUT_ROOT = Path("outputs") / "identify_magnetic_topology" / "PAD_score_method"
DEFAULT_ENERGY_RANGE_EV = (100.0, 300.0)
DEFAULT_MAX_MAG_DELTA_SECONDS = 60.0
DEFAULT_FIELD_ALIGNED_WINDOW_DEG = 10.0


@dataclass(frozen=True)
class PitchAngleBands:
    parallel_low: tuple[float, float] = (0.0, 30.0)
    perpendicular: tuple[float, float] = (85.0, 95.0)
    antiparallel_high: tuple[float, float] = (150.0, 180.0)
    intermediate_1: tuple[float, float] = (40.0, 50.0)
    intermediate_2: tuple[float, float] = (130.0, 140.0)


def iter_utc_days(start: datetime, end: datetime) -> list[datetime]:
    first = datetime.combine(start.date(), time.min, tzinfo=timezone.utc)
    last = datetime.combine(end.date(), time.min, tzinfo=timezone.utc)
    days = []
    current = first
    while current <= last:
        days.append(current)
        current += timedelta(days=1)
    return days


def format_unix_time(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")


def reorder_3d_variable_like_flux(
    path: Path,
    variable_name: str,
    times: np.ndarray,
    energy: np.ndarray,
    pitch,
) -> np.ndarray | None:
    """Read a 3D CDF variable and reorder it to (time, pitch, energy)."""
    cdf = cdflib.CDF(str(path))
    try:
        data = np.asarray(cdf.varget(variable_name))
    except Exception:
        return None
    if data.ndim != 3:
        return None

    pitch_index = None
    try:
        pitch_index = np.asarray(cdf.varget("pindex"))
    except Exception:
        pass

    time_axis = find_axis_by_length(data.shape, len(times), "time", variable_name)
    pitch_axis_size = infer_pitch_axis_size(data.shape, pitch_index)
    if np.asarray(pitch).ndim == 1:
        pitch_axis_size = len(np.asarray(pitch).reshape(-1))
    pitch_axis = find_axis_by_length(data.shape, pitch_axis_size, "pitch", variable_name)
    energy_axis = find_axis_by_length(data.shape, len(energy), "energy", variable_name)
    return np.moveaxis(data.astype(float), (time_axis, pitch_axis, energy_axis), (0, 1, 2))


def reorder_counts_like_flux(path: Path, times: np.ndarray, energy: np.ndarray, pitch) -> np.ndarray | None:
    """Read raw counts and reorder them to (time, pitch, energy) when available."""
    cdf = cdflib.CDF(str(path))
    info = cdf.cdf_info()
    names = list(info.zVariables) + list(info.rVariables)
    for name in names:
        if name.lower() != "counts":
            continue
        try:
            return reorder_3d_variable_like_flux(path, name, times, energy, pitch)
        except Exception:
            return None
    return None


def poisson_sigma_from_counts(flux: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Compute flux uncertainty from Poisson counting statistics.

    For calibrated flux F derived from counts N, Poisson statistics give
    sigma_N = sqrt(N), so sigma_F = |F| / sqrt(N).  Zero or negative counts
    do not provide a finite uncertainty and are left as NaN.
    """
    values = np.asarray(flux, dtype=float)
    raw_counts = np.asarray(counts, dtype=float)
    if values.shape != raw_counts.shape:
        raise ValueError(f"counts shape {raw_counts.shape} does not match flux shape {values.shape}.")
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(
            np.isfinite(values) & np.isfinite(raw_counts) & (raw_counts > 0.0),
            np.abs(values) / np.sqrt(raw_counts),
            np.nan,
        )
    return sigma


def reorder_sigma_like_flux(path: Path, times: np.ndarray, energy: np.ndarray, pitch) -> tuple[np.ndarray | None, str]:
    """Try to read product uncertainty and reorder it to sigma with shape (time, pitch, energy)."""
    cdf = cdflib.CDF(str(path))
    info = cdf.cdf_info()
    names = list(info.zVariables) + list(info.rVariables)
    candidates = []
    for name in names:
        lowered = name.lower()
        is_variance = lowered == "variance" or lowered.endswith("_variance") or "variance" in lowered
        is_sigma = any(token in lowered for token in ("sigma", "uncert", "error", "err"))
        if not is_variance and not is_sigma:
            continue
        data = np.asarray(cdf.varget(name))
        if data.ndim == 3:
            priority = 0 if is_variance else 1
            candidates.append((priority, name, data, is_variance))

    if not candidates:
        return None, "missing"
    candidates.sort(key=lambda item: item[0])

    for _, name, data, is_variance in candidates:
        try:
            reordered = reorder_3d_variable_like_flux(path, name, times, energy, pitch)
            if reordered is None:
                continue
            if is_variance:
                reordered = np.where(reordered >= 0.0, np.sqrt(reordered), np.nan)
                return reordered, f"cdf_variance:{name}"
            return reordered, f"cdf_sigma:{name}"
        except Exception:
            continue
    return None, "missing"


def load_pad_with_sigma(path: Path) -> dict:
    pad_data = load_pad_data(path)
    counts = reorder_counts_like_flux(path, np.asarray(pad_data["times"]), np.asarray(pad_data["energy"]), pad_data["pitch"])
    if counts is not None:
        sigma = poisson_sigma_from_counts(np.asarray(pad_data["flux"], dtype=float), counts)
        sigma_source = "counts_poisson:sigma_flux=abs(diff_en_fluxes)/sqrt(counts)"
    else:
        sigma, sigma_source = reorder_sigma_like_flux(
            path,
            np.asarray(pad_data["times"]),
            np.asarray(pad_data["energy"]),
            pad_data["pitch"],
        )
    pad_data["sigma"] = sigma
    pad_data["sigma_available"] = sigma is not None
    pad_data["sigma_source"] = sigma_source
    return pad_data


def pitch_at_time(pitch, time_index: int, energy_index: int | None = None) -> np.ndarray:
    pitch_array = np.asarray(pitch, dtype=float)
    if pitch_array.ndim == 1:
        return pitch_array.reshape(-1)
    if energy_index is None:
        return np.nanmedian(pitch_array[time_index], axis=1)
    return pitch_array[time_index, :, energy_index]


def integrate_energy_band(
    eflux: np.ndarray,
    eflux_sigma: np.ndarray | None,
    energy_eV: np.ndarray,
    energy_range_eV: tuple[float, float] = DEFAULT_ENERGY_RANGE_EV,
    method: str = "sum",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flux = np.asarray(eflux, dtype=float)
    energy = np.asarray(energy_eV, dtype=float)
    selected = np.isfinite(energy) & (energy >= energy_range_eV[0]) & (energy <= energy_range_eV[1])
    if not np.any(selected):
        raise ValueError(f"No SWE PAD energy channels were found in {energy_range_eV[0]:g}-{energy_range_eV[1]:g} eV.")

    band_flux = flux[:, :, selected]
    with np.errstate(invalid="ignore"):
        if method == "mean":
            pad_flux = np.nanmean(band_flux, axis=2)
        elif method == "sum":
            pad_flux = np.nansum(band_flux, axis=2)
        else:
            raise ValueError("method must be 'sum' or 'mean'.")

    if eflux_sigma is None:
        pad_sigma = np.full_like(pad_flux, np.nan, dtype=float)
    else:
        sigma_band = np.asarray(eflux_sigma, dtype=float)[:, :, selected]
        pad_sigma = np.sqrt(np.nansum(sigma_band * sigma_band, axis=2))
        if method == "mean":
            channel_count = max(1, int(np.count_nonzero(selected)))
            pad_sigma = pad_sigma / channel_count
    return pad_flux, pad_sigma, selected


def coadd_pads(
    pad_flux: np.ndarray,
    pad_sigma: np.ndarray,
    times_unix: np.ndarray,
    group_size: int = 4,
    keep_partial: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flux = np.asarray(pad_flux, dtype=float)
    sigma = np.asarray(pad_sigma, dtype=float)
    times = np.asarray(times_unix, dtype=float)
    if group_size <= 0:
        raise ValueError("group_size must be positive.")

    flux_groups = []
    sigma_groups = []
    time_groups = []
    for start in range(0, flux.shape[0], group_size):
        end = min(start + group_size, flux.shape[0])
        if end - start < group_size and not keep_partial:
            break
        flux_chunk = flux[start:end]
        sigma_chunk = sigma[start:end]
        flux_sum = np.nansum(flux_chunk, axis=0)
        flux_valid = np.any(np.isfinite(flux_chunk), axis=0)
        flux_sum = np.where(flux_valid, flux_sum, np.nan)

        sigma_sum_sq = np.nansum(sigma_chunk * sigma_chunk, axis=0)
        sigma_valid = np.any(np.isfinite(sigma_chunk), axis=0)
        sigma_sum = np.where(sigma_valid, np.sqrt(sigma_sum_sq), np.nan)
        flux_groups.append(flux_sum)
        sigma_groups.append(sigma_sum)
        time_groups.append(float(np.nanmean(times[start:end])))

    if not flux_groups:
        return (
            np.empty((0, flux.shape[1]), dtype=float),
            np.empty((0, flux.shape[1]), dtype=float),
            np.empty(0, dtype=float),
        )
    return np.vstack(flux_groups), np.vstack(sigma_groups), np.asarray(time_groups, dtype=float)


def normalized_flux_for_plot(flux: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(flux, dtype=float)
    mean_flux = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
    if not np.isfinite(mean_flux) or mean_flux <= 0.0:
        return values.copy(), False
    return values / mean_flux, True


def mask_range(pitch_angle_deg: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    pitch = np.asarray(pitch_angle_deg, dtype=float)
    return np.isfinite(pitch) & (pitch >= bounds[0]) & (pitch <= bounds[1])


def bin_mean_with_error(flux: np.ndarray, sigma: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(sigma, dtype=float)
    usable = np.asarray(mask, dtype=bool) & np.isfinite(values) & np.isfinite(errors)
    count = int(np.count_nonzero(usable))
    if count == 0:
        return float("nan"), float("nan"), 0
    representative_flux = float(np.nanmean(values[usable]))
    representative_sigma = float(np.sqrt(np.nansum(errors[usable] * errors[usable])) / count)
    return representative_flux, representative_sigma, count


def perpendicular_flux_with_error(
    pitch_angle_deg: np.ndarray,
    flux: np.ndarray,
    sigma: np.ndarray,
    bounds: tuple[float, float],
    target_deg: float = 90.0,
    fallback_width_deg: float = 10.0,
) -> tuple[float, float, int, str, float, float]:
    """Return perpendicular-band mean, or interpolate to 90 deg from adjacent bands."""
    pitch = np.asarray(pitch_angle_deg, dtype=float)
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(sigma, dtype=float)
    in_band = mask_range(pitch, bounds)
    band_flux, band_sigma, band_count = bin_mean_with_error(values, errors, in_band)
    if band_count > 0:
        return band_flux, band_sigma, band_count, "mean", bounds[0], bounds[1]

    usable = np.isfinite(pitch) & np.isfinite(values) & np.isfinite(errors)
    lower_candidates = np.where(
        usable & (pitch >= bounds[0] - fallback_width_deg) & (pitch < bounds[0])
    )[0]
    upper_candidates = np.where(
        usable & (pitch > bounds[1]) & (pitch <= bounds[1] + fallback_width_deg)
    )[0]
    if lower_candidates.size == 0 or upper_candidates.size == 0:
        return float("nan"), float("nan"), 0, "missing", float("nan"), float("nan")

    lower_index = lower_candidates[np.argmax(pitch[lower_candidates])]
    upper_index = upper_candidates[np.argmin(pitch[upper_candidates])]
    lower_pitch = float(pitch[lower_index])
    upper_pitch = float(pitch[upper_index])
    if not (lower_pitch < target_deg < upper_pitch):
        return float("nan"), float("nan"), 0, "missing", lower_pitch, upper_pitch

    upper_weight = (target_deg - lower_pitch) / (upper_pitch - lower_pitch)
    lower_weight = 1.0 - upper_weight
    interpolated_flux = float(lower_weight * values[lower_index] + upper_weight * values[upper_index])
    interpolated_sigma = float(
        np.sqrt((lower_weight * errors[lower_index]) ** 2 + (upper_weight * errors[upper_index]) ** 2)
    )
    return interpolated_flux, interpolated_sigma, 2, "interpolated_90deg", lower_pitch, upper_pitch


def field_aligned_mask(
    pitch_angle_deg: np.ndarray,
    flux: np.ndarray,
    sigma: np.ndarray,
    bounds: tuple[float, float],
    side: str,
    window_deg: float = DEFAULT_FIELD_ALIGNED_WINDOW_DEG,
) -> tuple[np.ndarray, tuple[float, float], int]:
    pitch = np.asarray(pitch_angle_deg, dtype=float)
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(sigma, dtype=float)
    in_bounds = mask_range(pitch, bounds)
    if not np.any(in_bounds):
        return np.zeros(pitch.shape, dtype=bool), (float("nan"), float("nan")), 0

    candidate_edges = np.unique(np.round(pitch[in_bounds], 8))
    candidates: list[tuple[int, float, tuple[float, float], np.ndarray]] = []
    for edge in candidate_edges:
        if side == "low":
            window_bounds = (float(edge), min(bounds[1], float(edge) + window_deg))
            closeness = abs(window_bounds[0] - bounds[0])
        elif side == "high":
            window_bounds = (max(bounds[0], float(edge) - window_deg), float(edge))
            closeness = abs(bounds[1] - window_bounds[1])
        else:
            raise ValueError("side must be 'low' or 'high'.")
        mask = in_bounds & (pitch >= window_bounds[0]) & (pitch <= window_bounds[1])
        valid_count = int(np.count_nonzero(mask & np.isfinite(values) & np.isfinite(errors)))
        candidates.append((valid_count, closeness, window_bounds, mask))

    if not candidates:
        return np.zeros(pitch.shape, dtype=bool), (float("nan"), float("nan")), 0

    candidates.sort(key=lambda item: (-item[0], item[1], item[2][0]))
    count, _, selected_bounds, selected_mask = candidates[0]
    if count == 0:
        return np.zeros(pitch.shape, dtype=bool), selected_bounds, 0
    return selected_mask, selected_bounds, count

def pad_score(
    field_aligned_flux: float,
    field_aligned_sigma: float,
    perpendicular_flux: float,
    perpendicular_sigma: float,
) -> tuple[float, float]:
    required = [field_aligned_flux, field_aligned_sigma, perpendicular_flux, perpendicular_sigma]
    if not all(np.isfinite(required)):
        return float("nan"), float("nan")
    propagated_sigma = float(np.sqrt(field_aligned_sigma**2 + perpendicular_sigma**2))
    if propagated_sigma <= 0.0 or not np.isfinite(propagated_sigma):
        return float("nan"), propagated_sigma
    return float((field_aligned_flux - perpendicular_flux) / propagated_sigma), propagated_sigma


def classify_pad_score(score: float, threshold_sigma: float = 2.0) -> str:
    if not np.isfinite(score):
        return "invalid"
    if score > threshold_sigma:
        return "beam"
    if score < -threshold_sigma:
        return "loss_cone"
    return "isotropic"


def classify_half_pad(
    field_aligned_flux: float,
    field_aligned_sigma: float,
    perpendicular_flux: float,
    perpendicular_sigma: float,
    intermediate_flux: float | None = None,
    intermediate_sigma: float | None = None,
    threshold_sigma: float = 2.0,
) -> str:
    score, _ = pad_score(field_aligned_flux, field_aligned_sigma, perpendicular_flux, perpendicular_sigma)
    return classify_pad_score(score, threshold_sigma=threshold_sigma)


def electron_depletion_flag(
    flux: np.ndarray,
    sigma: np.ndarray,
    min_valid_bins: int = 6,
    threshold_sigma: float = 2.0,
) -> tuple[bool, str, int]:
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(sigma, dtype=float)
    usable = np.isfinite(values) & np.isfinite(errors)
    count = int(np.count_nonzero(usable))
    if count < min_valid_bins:
        return False, "invalid_coverage", count
    return bool(np.all(values[usable] <= threshold_sigma * errors[usable])), "ok", count


def topology_from_shape(pad_shape: str) -> str:
    if pad_shape == "electron_depletion":
        return "closed_like"
    if "__" not in pad_shape:
        return "unknown"
    toward_class, away_class = pad_shape.split("__", 1)
    if toward_class == "loss_cone" and away_class == "loss_cone":
        return "closed_like"
    if "beam" in {toward_class, away_class}:
        return "open_like"
    if toward_class == "isotropic" and away_class == "isotropic":
        return "open_or_closed_ambiguous"
    return "ambiguous"


def classify_full_pad(
    flux: np.ndarray,
    sigma: np.ndarray,
    pitch_angle_deg: np.ndarray,
    downward_side: str,
    bands: PitchAngleBands = PitchAngleBands(),
    threshold_sigma: float = 2.0,
    min_valid_bins: int = 6,
    field_aligned_window_deg: float = DEFAULT_FIELD_ALIGNED_WINDOW_DEG,
) -> dict:
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(sigma, dtype=float)
    pitch = np.asarray(pitch_angle_deg, dtype=float)
    diagnostics: dict[str, float | int | bool | str] = {}

    if values.size != pitch.size or errors.size != pitch.size:
        return {
            "valid": False,
            "pad_shape": "invalid",
            "toward_class": "invalid",
            "away_class": "invalid",
            "topology": "unknown",
            "reason": "shape_mismatch",
            "diagnostics": diagnostics,
        }
    if not np.any(np.isfinite(errors)):
        return {
            "valid": False,
            "pad_shape": "invalid",
            "toward_class": "invalid",
            "away_class": "invalid",
            "topology": "unknown",
            "reason": "missing_sigma",
            "diagnostics": diagnostics,
        }

    depletion, depletion_reason, valid_bin_count = electron_depletion_flag(
        values,
        errors,
        min_valid_bins=min_valid_bins,
        threshold_sigma=threshold_sigma,
    )
    diagnostics["valid_pitch_bins"] = valid_bin_count
    diagnostics["depletion_flag"] = depletion
    if depletion_reason == "invalid_coverage":
        return {
            "valid": False,
            "pad_shape": "invalid",
            "toward_class": "invalid",
            "away_class": "invalid",
            "topology": "unknown",
            "reason": "invalid_coverage",
            "diagnostics": diagnostics,
        }
    if depletion:
        return {
            "valid": True,
            "pad_shape": "electron_depletion",
            "toward_class": "electron_depletion",
            "away_class": "electron_depletion",
            "topology": "closed_like",
            "reason": "all_valid_bins_within_2sigma_of_zero",
            "diagnostics": diagnostics,
        }

    # Xu et al. (2019), section 2.3.2, use the most field-aligned 10-degree
    # interval available inside 0-30 and 150-180 degrees.  Keep the interval
    # selection here (rather than averaging each full 30-degree band) so the
    # public ``field_aligned_window_deg`` parameter controls the calculation.
    low_mask, low_window_bounds, _ = field_aligned_mask(
        pitch,
        values,
        errors,
        bands.parallel_low,
        side="low",
        window_deg=field_aligned_window_deg,
    )
    high_mask, high_window_bounds, _ = field_aligned_mask(
        pitch,
        values,
        errors,
        bands.antiparallel_high,
        side="high",
        window_deg=field_aligned_window_deg,
    )
    low_flux, low_sigma, low_count = bin_mean_with_error(values, errors, low_mask)
    high_flux, high_sigma, high_count = bin_mean_with_error(values, errors, high_mask)
    perp_flux, perp_sigma, perp_count, perp_method, perp_lower_pitch, perp_upper_pitch = perpendicular_flux_with_error(
        pitch,
        values,
        errors,
        bands.perpendicular,
        target_deg=0.5 * (bands.perpendicular[0] + bands.perpendicular[1]),
    )
    low_score, low_score_sigma = pad_score(low_flux, low_sigma, perp_flux, perp_sigma)
    high_score, high_score_sigma = pad_score(high_flux, high_sigma, perp_flux, perp_sigma)
    low_variance = float(low_sigma * low_sigma) if np.isfinite(low_sigma) else float("nan")
    high_variance = float(high_sigma * high_sigma) if np.isfinite(high_sigma) else float("nan")
    perp_variance = float(perp_sigma * perp_sigma) if np.isfinite(perp_sigma) else float("nan")
    low_score_variance = float(low_score_sigma * low_score_sigma) if np.isfinite(low_score_sigma) else float("nan")
    high_score_variance = float(high_score_sigma * high_score_sigma) if np.isfinite(high_score_sigma) else float("nan")

    diagnostics.update(
        {
            "low_flux": low_flux,
            "low_sigma": low_sigma,
            "low_variance": low_variance,
            "high_flux": high_flux,
            "high_sigma": high_sigma,
            "high_variance": high_variance,
            "perpendicular_flux": perp_flux,
            "perpendicular_sigma": perp_sigma,
            "perpendicular_variance": perp_variance,
            "low_pad_score": low_score,
            "high_pad_score": high_score,
            "low_score_sigma": low_score_sigma,
            "high_score_sigma": high_score_sigma,
            "low_score_variance": low_score_variance,
            "high_score_variance": high_score_variance,
            "coverage_low_pa": low_count,
            "coverage_high_pa": high_count,
            "coverage_perpendicular": perp_count,
            "low_fa_window_low_deg": low_window_bounds[0],
            "low_fa_window_high_deg": low_window_bounds[1],
            "high_fa_window_low_deg": high_window_bounds[0],
            "high_fa_window_high_deg": high_window_bounds[1],
            "low_fa_window_valid_count": low_count,
            "high_fa_window_valid_count": high_count,
            "field_aligned_window_deg": field_aligned_window_deg,
            "perpendicular_method": perp_method,
            "perpendicular_interp_lower_pitch_deg": perp_lower_pitch,
            "perpendicular_interp_upper_pitch_deg": perp_upper_pitch,
        }
    )
    if downward_side == "low":
        diagnostics.update(
            {
                "toward_fa_flux": low_flux,
                "toward_fa_sigma": low_sigma,
                "toward_fa_variance": low_variance,
                "toward_score_sigma": low_score_sigma,
                "toward_score_variance": low_score_variance,
                "toward_pad_score": low_score,
                "away_fa_flux": high_flux,
                "away_fa_sigma": high_sigma,
                "away_fa_variance": high_variance,
                "away_score_sigma": high_score_sigma,
                "away_score_variance": high_score_variance,
                "away_pad_score": high_score,
            }
        )
    elif downward_side == "high":
        diagnostics.update(
            {
                "toward_fa_flux": high_flux,
                "toward_fa_sigma": high_sigma,
                "toward_fa_variance": high_variance,
                "toward_score_sigma": high_score_sigma,
                "toward_score_variance": high_score_variance,
                "toward_pad_score": high_score,
                "away_fa_flux": low_flux,
                "away_fa_sigma": low_sigma,
                "away_fa_variance": low_variance,
                "away_score_sigma": low_score_sigma,
                "away_score_variance": low_score_variance,
                "away_pad_score": low_score,
            }
        )
    if low_count == 0 or high_count == 0 or perp_count == 0:
        return {
            "valid": False,
            "pad_shape": "invalid",
            "toward_class": "invalid",
            "away_class": "invalid",
            "topology": "unknown",
            "reason": "invalid_coverage",
            "diagnostics": diagnostics,
        }

    low_class = classify_pad_score(low_score, threshold_sigma=threshold_sigma)
    high_class = classify_pad_score(high_score, threshold_sigma=threshold_sigma)
    if downward_side == "low":
        toward_class, away_class = low_class, high_class
        toward_score, away_score = low_score, high_score
    elif downward_side == "high":
        toward_class, away_class = high_class, low_class
        toward_score, away_score = high_score, low_score
    else:
        return {
            "valid": False,
            "pad_shape": "invalid",
            "toward_class": "invalid",
            "away_class": "invalid",
            "topology": "unknown",
            "reason": "invalid_downward_side",
            "diagnostics": diagnostics,
        }

    pad_shape = f"{toward_class}__{away_class}"
    valid = toward_class != "invalid" and away_class != "invalid"
    return {
        "valid": valid,
        "pad_shape": pad_shape,
        "toward_class": toward_class,
        "away_class": away_class,
        "topology": topology_from_shape(pad_shape),
        "reason": "ok" if valid else "invalid_score",
        "diagnostics": diagnostics,
    }


def altitude_and_sza_from_position(position_km: np.ndarray) -> tuple[float, float]:
    position = np.asarray(position_km, dtype=float)
    radius = float(np.linalg.norm(position))
    if not np.isfinite(radius) or radius <= 0.0:
        return float("nan"), float("nan")
    altitude_km = radius - MARS_RADIUS_KM
    sza_deg = float(np.degrees(np.arccos(np.clip(position[0] / radius, -1.0, 1.0))))
    return altitude_km, sza_deg


def pad_pitch_centers_for_group(pad_data: dict, time_indices: np.ndarray, selected_energy: np.ndarray) -> np.ndarray:
    pitch = np.asarray(pad_data["pitch"], dtype=float)
    if pitch.ndim == 1:
        return pitch.reshape(-1)
    selected = np.where(selected_energy)[0]
    if selected.size == 0:
        return np.nanmedian(pitch[time_indices], axis=(0, 2))
    return np.nanmedian(pitch[np.ix_(time_indices, np.arange(pitch.shape[1]), selected)], axis=(0, 2))


def classify_pad_timeseries(
    start: datetime,
    end: datetime,
    data_root: Path | tuple[Path, ...] | list[Path] = DEFAULT_DATA_ROOT,
    energy_range_eV: tuple[float, float] = DEFAULT_ENERGY_RANGE_EV,
    energy_method: str = "mean",
    group_size: int = 4,
    keep_partial: bool = False,
    threshold_sigma: float = 2.0,
    max_mag_delta_seconds: float = DEFAULT_MAX_MAG_DELTA_SECONDS,
    bands: PitchAngleBands = PitchAngleBands(),
    field_aligned_window_deg: float = DEFAULT_FIELD_ALIGNED_WINDOW_DEG,
) -> pd.DataFrame:
    magnetic_geometry = load_magnetic_geometry_interval(data_root, start, end)
    if magnetic_geometry is None:
        raise FileNotFoundError("No usable MAG sunstate-1sec samples were found in the requested interval.")

    rows = []
    for day in iter_utc_days(start, end):
        try:
            pad_file = infer_daily_file(data_root, "swe", "svypad", day, "cdf")
            pad_data = load_pad_with_sigma(pad_file)
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            print(f"[pad-score] skip SWE day {day.date()}: {exc}", flush=True)
            continue

        times = np.asarray(pad_data["times"], dtype=float)
        time_mask = (times >= start.timestamp()) & (times <= end.timestamp())
        indices = np.where(time_mask)[0]
        if indices.size == 0:
            continue

        daily_flux = np.asarray(pad_data["flux"], dtype=float)[indices]
        daily_sigma = None if pad_data["sigma"] is None else np.asarray(pad_data["sigma"], dtype=float)[indices]
        sigma_source = str(pad_data.get("sigma_source", "missing"))
        daily_times = times[indices]
        pad_flux, pad_sigma, selected_energy = integrate_energy_band(
            daily_flux,
            daily_sigma,
            np.asarray(pad_data["energy"], dtype=float),
            energy_range_eV=energy_range_eV,
            method=energy_method,
        )
        co_flux, co_sigma, co_times = coadd_pads(
            pad_flux,
            pad_sigma,
            daily_times,
            group_size=group_size,
            keep_partial=keep_partial,
        )

        for group_index, unix_time in enumerate(co_times):
            source_start = group_index * group_size
            source_end = min(source_start + group_size, indices.size)
            if source_end - source_start < group_size and not keep_partial:
                continue
            source_indices = indices[source_start:source_end]
            sample_time = datetime.fromtimestamp(float(unix_time), tz=timezone.utc)
            pitch_angle = pad_pitch_centers_for_group(pad_data, source_indices, selected_energy)
            mag_direction = nearest_magnetic_field_direction(magnetic_geometry, sample_time, max_mag_delta_seconds)

            if mag_direction is None:
                result = {
                    "valid": False,
                    "pad_shape": "invalid",
                    "toward_class": "invalid",
                    "away_class": "invalid",
                    "topology": "unknown",
                    "reason": "missing_mag_sample",
                    "diagnostics": {},
                }
                altitude_km, sza_deg = float("nan"), float("nan")
                downward_side = "unknown"
                field_direction = "missing_mag_sample"
            else:
                altitude_km, sza_deg = altitude_and_sza_from_position(mag_direction.position_km)
                if mag_direction.field_direction == "toward_surface":
                    downward_side = "low"
                elif mag_direction.field_direction == "away_from_surface":
                    downward_side = "high"
                else:
                    downward_side = "unknown"
                field_direction = mag_direction.field_direction
                result = classify_full_pad(
                    co_flux[group_index],
                    co_sigma[group_index],
                    pitch_angle,
                    downward_side,
                    bands=bands,
                    threshold_sigma=threshold_sigma,
                    field_aligned_window_deg=field_aligned_window_deg,
                )

            diagnostics = result.get("diagnostics", {})
            normalized_flux, normalization_valid = normalized_flux_for_plot(co_flux[group_index])
            mean_flux = float(np.nanmean(co_flux[group_index])) if np.any(np.isfinite(co_flux[group_index])) else float("nan")
            rows.append(
                {
                    "time": format_unix_time(unix_time),
                    "time_unix": float(unix_time),
                    "altitude_km": altitude_km,
                    "sza_deg": sza_deg,
                    "valid": bool(result["valid"]),
                    "pad_shape": result["pad_shape"],
                    "toward_class": result["toward_class"],
                    "away_class": result["away_class"],
                    "topology": result["topology"],
                    "toward_pad_score": diagnostics.get("toward_pad_score", float("nan")),
                    "away_pad_score": diagnostics.get("away_pad_score", float("nan")),
                    "low_pad_score": diagnostics.get("low_pad_score", float("nan")),
                    "high_pad_score": diagnostics.get("high_pad_score", float("nan")),
                    "toward_fa_flux": diagnostics.get("toward_fa_flux", float("nan")),
                    "away_fa_flux": diagnostics.get("away_fa_flux", float("nan")),
                    "low_fa_flux": diagnostics.get("low_flux", float("nan")),
                    "high_fa_flux": diagnostics.get("high_flux", float("nan")),
                    "perpendicular_flux": diagnostics.get("perpendicular_flux", float("nan")),
                    "toward_fa_sigma": diagnostics.get("toward_fa_sigma", float("nan")),
                    "away_fa_sigma": diagnostics.get("away_fa_sigma", float("nan")),
                    "low_fa_sigma": diagnostics.get("low_sigma", float("nan")),
                    "high_fa_sigma": diagnostics.get("high_sigma", float("nan")),
                    "perpendicular_sigma": diagnostics.get("perpendicular_sigma", float("nan")),
                    "toward_fa_variance": diagnostics.get("toward_fa_variance", float("nan")),
                    "away_fa_variance": diagnostics.get("away_fa_variance", float("nan")),
                    "low_fa_variance": diagnostics.get("low_variance", float("nan")),
                    "high_fa_variance": diagnostics.get("high_variance", float("nan")),
                    "perpendicular_variance": diagnostics.get("perpendicular_variance", float("nan")),
                    "toward_score_sigma": diagnostics.get("toward_score_sigma", float("nan")),
                    "away_score_sigma": diagnostics.get("away_score_sigma", float("nan")),
                    "toward_score_variance": diagnostics.get("toward_score_variance", float("nan")),
                    "away_score_variance": diagnostics.get("away_score_variance", float("nan")),
                    "low_score_sigma": diagnostics.get("low_score_sigma", float("nan")),
                    "high_score_sigma": diagnostics.get("high_score_sigma", float("nan")),
                    "low_score_variance": diagnostics.get("low_score_variance", float("nan")),
                    "high_score_variance": diagnostics.get("high_score_variance", float("nan")),
                    "mean_flux_100_300": mean_flux,
                    "invalid_flux": not normalization_valid,
                    "normalized_flux_100_300_by_pa": json.dumps(normalized_flux.tolist()),
                    "coverage_low_pa": int(diagnostics.get("coverage_low_pa", 0)),
                    "coverage_high_pa": int(diagnostics.get("coverage_high_pa", 0)),
                    "coverage_perpendicular": int(diagnostics.get("coverage_perpendicular", 0)),
                    "perpendicular_method": diagnostics.get("perpendicular_method", "missing"),
                    "perpendicular_interp_lower_pitch_deg": diagnostics.get(
                        "perpendicular_interp_lower_pitch_deg", float("nan")
                    ),
                    "perpendicular_interp_upper_pitch_deg": diagnostics.get(
                        "perpendicular_interp_upper_pitch_deg", float("nan")
                    ),
                    "low_fa_window_low_deg": diagnostics.get("low_fa_window_low_deg", float("nan")),
                    "low_fa_window_high_deg": diagnostics.get("low_fa_window_high_deg", float("nan")),
                    "high_fa_window_low_deg": diagnostics.get("high_fa_window_low_deg", float("nan")),
                    "high_fa_window_high_deg": diagnostics.get("high_fa_window_high_deg", float("nan")),
                    "depletion_flag": bool(diagnostics.get("depletion_flag", False)),
                    "reason": result["reason"],
                    "sigma_source": sigma_source,
                    "field_direction": field_direction,
                    "downward_side": downward_side,
                    "source_file": str(pad_file),
                }
            )
    return pd.DataFrame(rows)


def plot_time_series_classification(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.text(0.5, 0.5, "No PAD classifications", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    class_order = sorted(str(item) for item in df["pad_shape"].fillna("invalid").unique())
    class_to_y = {label: index for index, label in enumerate(class_order)}
    times = [parse_iso_timestamp(value) for value in df["time"]]
    y = [class_to_y[str(value)] for value in df["pad_shape"].fillna("invalid")]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    scatter = ax.scatter(times, y, c=y, cmap="tab20", s=28, marker="s")
    ax.set_yticks(list(class_to_y.values()), list(class_to_y.keys()))
    ax.set_xlabel("UTC")
    ax.set_ylabel("PAD shape")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pad_score_time_series(df: pd.DataFrame, output_path: Path, threshold_sigma: float = 2.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    if df.empty:
        ax.text(0.5, 0.5, "No PAD scores", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        times = [parse_iso_timestamp(value) for value in df["time"]]
        toward = np.asarray(df.get("toward_pad_score", np.nan), dtype=float)
        away = np.asarray(df.get("away_pad_score", np.nan), dtype=float)
        ax.plot(times, toward, marker="o", markersize=3.5, linewidth=1.3, label="towards")
        ax.plot(times, away, marker="s", markersize=3.5, linewidth=1.3, label="away")
        ax.axhline(0.0, color="0.2", linewidth=0.9, alpha=0.8)
        ax.axhline(threshold_sigma, color="0.45", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.axhline(-threshold_sigma, color="0.45", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_xlabel("UTC")
        ax.set_ylabel("PAD score")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=False)
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def default_output_dir(output_root: Path, start: datetime, end: datetime) -> Path:
    return output_root / f"{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify MAVEN SWE PAD shapes using a Weber-style PAD score method.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--energy-range", nargs=2, type=float, default=DEFAULT_ENERGY_RANGE_EV, metavar=("LOW_EV", "HIGH_EV"))
    parser.add_argument("--energy-method", choices=("sum", "mean"), default="mean")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--keep-partial", action="store_true")
    parser.add_argument("--threshold-sigma", type=float, default=2.0)
    parser.add_argument("--max-mag-delta-seconds", type=float, default=DEFAULT_MAX_MAG_DELTA_SECONDS)
    parser.add_argument("--field-aligned-window", type=float, default=DEFAULT_FIELD_ALIGNED_WINDOW_DEG)
    parser.add_argument("--parallel-low", nargs=2, type=float, default=(0.0, 30.0))
    parser.add_argument("--perpendicular", nargs=2, type=float, default=(85.0, 95.0))
    parser.add_argument("--antiparallel-high", nargs=2, type=float, default=(150.0, 180.0))
    parser.add_argument("--intermediate-1", nargs=2, type=float, default=(40.0, 50.0))
    parser.add_argument("--intermediate-2", nargs=2, type=float, default=(130.0, 140.0))
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    start = parse_iso_timestamp(args.start).astimezone(timezone.utc)
    end = parse_iso_timestamp(args.end).astimezone(timezone.utc)
    if end <= start:
        raise ValueError("--end must be later than --start.")

    energy_range = (float(args.energy_range[0]), float(args.energy_range[1]))
    if not (0.0 < energy_range[0] < energy_range[1]):
        raise ValueError("--energy-range must satisfy 0 < LOW_EV < HIGH_EV.")
    bands = PitchAngleBands(
        parallel_low=tuple(float(v) for v in args.parallel_low),
        perpendicular=tuple(float(v) for v in args.perpendicular),
        antiparallel_high=tuple(float(v) for v in args.antiparallel_high),
        intermediate_1=tuple(float(v) for v in args.intermediate_1),
        intermediate_2=tuple(float(v) for v in args.intermediate_2),
    )

    output_dir = default_output_dir(Path(args.output_root).expanduser().resolve(), start, end)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = classify_pad_timeseries(
        start=start,
        end=end,
        data_root=Path(args.data_root).expanduser().resolve(),
        energy_range_eV=energy_range,
        energy_method=args.energy_method,
        group_size=int(args.group_size),
        keep_partial=bool(args.keep_partial),
        threshold_sigma=float(args.threshold_sigma),
        max_mag_delta_seconds=float(args.max_mag_delta_seconds),
        bands=bands,
        field_aligned_window_deg=float(args.field_aligned_window),
    )

    csv_path = output_dir / "pad_score_classification.csv"
    plot_path = output_dir / "pad_score_classification.png"
    score_plot_path = output_dir / "pad_score_time_series.png"
    df.to_csv(csv_path, index=False)
    plot_time_series_classification(df, plot_path)
    plot_pad_score_time_series(df, score_plot_path, threshold_sigma=float(args.threshold_sigma))

    summary = {
        "start": start.isoformat(timespec="seconds"),
        "end": end.isoformat(timespec="seconds"),
        "energy_range_eV": list(energy_range),
        "energy_method": args.energy_method,
        "group_size": int(args.group_size),
        "keep_partial": bool(args.keep_partial),
        "threshold_sigma": float(args.threshold_sigma),
        "field_aligned_window_deg": float(args.field_aligned_window),
        "score_definition": "PAD score = (fFA - fperp) / sqrt(sigma_FA^2 + sigma_perp^2); fFA is the mean flux over 0-30 deg or 150-180 deg. fperp is the mean flux over 85-95 deg when available; otherwise it is linearly interpolated to 90 deg from the nearest valid bin within 75-85 deg and the nearest valid bin within 95-105 deg.",
        "sigma_source": "Default: Poisson statistics of measured electron fluxes using raw counts, sigma_flux = abs(diff_en_fluxes) / sqrt(counts). If a file has no usable 3D counts variable, fall back to product uncertainty/variance and mark each row's sigma_source accordingly. Sigma is then propagated through energy averaging, 8 s coadding, pitch-bin averaging, and fFA-fperp subtraction.",
        "field_aligned_window_selection": "Disabled by current logic: field-aligned flux uses the full configured low/high PA ranges. The --field-aligned-window argument is retained for CLI compatibility but is not used.",
        "pitch_angle_bands": bands.__dict__,
        "rows": int(len(df)),
        "valid_rows": int(df["valid"].sum()) if not df.empty else 0,
        "outputs": {"csv": str(csv_path), "classification_plot": str(plot_path), "score_plot": str(score_plot_path)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
