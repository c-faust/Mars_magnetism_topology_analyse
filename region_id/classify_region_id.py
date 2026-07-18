from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bow_shock.models import (
    DEFAULT_MODEL_NAME,
    MARS_RADIUS_KM,
    AxisymmetricConicModel,
    evaluate_position,
    get_model,
)
from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from identify_magnetic_topology.magnetic_field_direction import (
    load_magnetic_geometry_interval,
)
from region_id.data_features import (
    load_static_features_interval,
    load_swe_features_interval,
    nearest_feature_index,
)


REGION_NAMES = {
    0: "Unknown",
    1: "Solar wind",
    2: "Magnetosheath",
    3: "Ionosphere",
    4: "Magnetic lobes",
}

# Average MPB surface from Vignes et al. (2000), expressed in MSO coordinates.
VIGNES_2000_MPB = AxisymmetricConicModel(
    name="vignes2000_mpb",
    display_name="Vignes et al. (2000) MPB",
    focus_x_rm=0.78,
    eccentricity=0.90,
    semilatus_rectum_rm=0.96,
    source_doi="10.1029/1999GL010703",
    source_url="https://doi.org/10.1029/1999GL010703",
    notes="Average magnetic-pileup-boundary conic used as an inner geometric guide.",
)


@dataclass(frozen=True)
class RegionClassifierConfig:
    cadence_seconds: float = 10.0
    max_mag_delta_seconds: float = 2.0
    max_swe_delta_seconds: float = 6.0
    max_static_delta_seconds: float = 12.0
    boundary_margin_km: float = 100.0
    ionosphere_max_altitude_km: float = 400.0
    heavy_ion_fraction_threshold: float = 0.45
    cold_heavy_ion_max_energy_eV: float = 10.0
    lobe_min_sza_deg: float = 100.0
    lobe_min_altitude_km: float = 400.0
    lobe_min_b_nT: float = 5.0
    lobe_max_b_nT: float = 25.0
    lobe_max_b_relative_std: float = 0.25
    lobe_max_direction_dispersion_deg: float = 25.0
    lobe_min_tail_alignment: float = 0.50
    magnetic_window_seconds: float = 60.0
    current_sheet_flank_window_seconds: float = 20.0
    current_sheet_center_half_window_seconds: float = 10.0
    current_sheet_min_rotation_deg: float = 90.0
    current_sheet_b_dip_ratio: float = 0.70
    photoelectron_ratio_threshold: float = 1.20
    electron_void_target_energy_eV: float = 40.0
    electron_void_flux_threshold: float = 1.0e5


CSV_FIELDS = [
    "time_unix",
    "time_utc",
    "region_id",
    "region_name",
    "confidence",
    "reason",
    "geometry_only",
    "mag_invalid_reason",
    "x_mso_km",
    "y_mso_km",
    "z_mso_km",
    "x_mso_rm",
    "y_mso_rm",
    "z_mso_rm",
    "altitude_km",
    "sza_deg",
    "bow_model",
    "bow_location",
    "bow_radial_offset_km",
    "mpb_location",
    "mpb_radial_offset_km",
    "bx_nT",
    "by_nT",
    "bz_nT",
    "b_nT",
    "b_window_median_nT",
    "b_to_window_median_ratio",
    "b_relative_std",
    "b_direction_dispersion_deg",
    "b_tail_alignment",
    "component_reversal",
    "current_sheet_signature",
    "current_sheet_rotation_deg",
    "current_sheet_dip_ratio",
    "current_sheet_center_min_b_nT",
    "current_sheet_flank_median_b_nT",
    "photoelectron_present",
    "photoelectron_ratio",
    "electron_void",
    "electron_flux_target",
    "high_energy_suppression_ratio",
    "planetary_heavy_ion_flux_fraction",
    "heavy_ion_peak_energy_eV",
    "static_valid_bin_count",
    "planetary_heavy_ion_valid_bin_count",
    "total_valid_ion_flux",
    "planetary_heavy_ion_integrated_flux",
    "hplus_flux_fraction",
    "hplus_peak_energy_eV",
    "hplus_log_energy_width",
    "swe_valid",
    "static_valid",
    "mag_time_delta_seconds",
    "swe_time_delta_seconds",
    "static_time_delta_seconds",
]


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _true(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def classify_region_sample(
    features: dict[str, Any],
    config: RegionClassifierConfig | None = None,
) -> tuple[int, float, str]:
    """Classify one sample using geometry first and plasma/MAG support second."""
    cfg = config or RegionClassifierConfig()
    if not _true(features.get("mag_valid")):
        return 0, 0.0, str(
            features.get("mag_invalid_reason") or "missing_mag_position"
        )

    bow_offset = features.get("bow_radial_offset_km")
    if _finite(bow_offset) and abs(float(bow_offset)) <= cfg.boundary_margin_km:
        return 0, 0.35, "near_bow_shock"
    if features.get("bow_location") == "outside":
        return 1, 0.98, "outside_gruesbeck_bow_shock"

    mpb_offset = features.get("mpb_radial_offset_km")
    if _finite(mpb_offset) and abs(float(mpb_offset)) <= cfg.boundary_margin_km:
        return 0, 0.35, "near_magnetic_pileup_boundary"
    if features.get("mpb_location") == "outside":
        return 2, 0.86, "inside_bow_shock_outside_mpb"

    if _true(features.get("current_sheet_signature")):
        return 0, 0.40, "nightside_current_sheet_signature"

    altitude = float(features.get("altitude_km", np.nan))
    heavy_fraction = float(
        features.get("planetary_heavy_ion_flux_fraction", np.nan)
    )
    heavy_peak = float(features.get("heavy_ion_peak_energy_eV", np.nan))
    low_altitude = _finite(altitude) and altitude <= cfg.ionosphere_max_altitude_km
    cold_heavy_ions = (
        _finite(heavy_fraction)
        and heavy_fraction >= cfg.heavy_ion_fraction_threshold
        and _finite(heavy_peak)
        and heavy_peak <= cfg.cold_heavy_ion_max_energy_eV
    )
    photoelectrons = _true(features.get("photoelectron_present"))
    if low_altitude and cold_heavy_ions:
        support = ["low_altitude"]
        support.append("cold_planetary_heavy_ions")
        if photoelectrons:
            support.append("photoelectrons")
        confidence = min(0.99, 0.90 + 0.04 * (len(support) - 1))
        return 3, confidence, ";".join(support)
    if cold_heavy_ions:
        return 3, 0.82, "cold_planetary_heavy_ions"
    if low_altitude and photoelectrons:
        return 3, 0.94, "low_altitude;photoelectrons"
    if low_altitude:
        return 0, 0.30, "low_altitude_without_ionospheric_particle_evidence"

    sza = float(features.get("sza_deg", np.nan))
    b_median = float(features.get("b_window_median_nT", np.nan))
    b_relative_std = float(features.get("b_relative_std", np.nan))
    b_direction_dispersion = float(
        features.get("b_direction_dispersion_deg", np.nan)
    )
    b_tail_alignment = float(features.get("b_tail_alignment", np.nan))
    lobe_geometry = (
        features.get("mpb_location") == "inside"
        and _finite(sza)
        and sza >= cfg.lobe_min_sza_deg
        and _finite(altitude)
        and altitude >= cfg.lobe_min_altitude_km
    )
    stable_lobe_field = (
        _finite(b_median)
        and cfg.lobe_min_b_nT <= b_median <= cfg.lobe_max_b_nT
        and _finite(b_relative_std)
        and b_relative_std <= cfg.lobe_max_b_relative_std
        and _finite(b_direction_dispersion)
        and b_direction_dispersion <= cfg.lobe_max_direction_dispersion_deg
        and _finite(b_tail_alignment)
        and b_tail_alignment >= cfg.lobe_min_tail_alignment
    )
    lobe_particle_exclusion = (
        _true(features.get("electron_void"))
        or features.get("photoelectron_present") is False
    )
    if lobe_geometry and stable_lobe_field and lobe_particle_exclusion:
        support = ["nightside_inside_mpb", "stable_5_25nT_field"]
        if _true(features.get("electron_void")):
            support.append("electron_void")
        if features.get("photoelectron_present") is False:
            support.append("no_photoelectron_peak")
        confidence = min(0.95, 0.78 + 0.05 * (len(support) - 2))
        return 4, confidence, ";".join(support)

    if lobe_geometry and stable_lobe_field:
        return 0, 0.30, "stable_lobe_field_without_particle_exclusion"
    if lobe_geometry:
        return 0, 0.30, "nightside_inner_region_without_stable_lobe_field"
    return 0, 0.25, "inside_mpb_not_resolved_by_available_features"


def _sample_magnetic_context(
    magnetic_geometry: dict,
    target_unix: float,
    config: RegionClassifierConfig,
) -> dict[str, Any]:
    times = np.asarray(magnetic_geometry["times"], dtype=float)
    if times.size == 0:
        return {
            "mag_valid": False,
            "mag_invalid_reason": "missing_mag_sample",
            "mag_time_delta_seconds": float("nan"),
        }
    insertion = int(np.searchsorted(times, target_unix, side="left"))
    candidates = [
        item
        for item in (insertion - 1, insertion)
        if 0 <= item < times.size
    ]
    index = min(candidates, key=lambda item: abs(float(times[item]) - target_unix))
    delta = abs(float(times[index]) - target_unix)
    if delta > config.max_mag_delta_seconds:
        return {
            "mag_valid": False,
            "mag_invalid_reason": "mag_time_mismatch",
            "mag_time_delta_seconds": delta,
        }

    field = np.asarray(magnetic_geometry["magnetic_field_nT"][index], dtype=float)
    position = np.asarray(magnetic_geometry["position_km"][index], dtype=float)
    if not np.all(np.isfinite(position)):
        return {
            "mag_valid": False,
            "mag_invalid_reason": "invalid_position",
            "mag_time_delta_seconds": delta,
        }
    if not np.all(np.isfinite(field)):
        return {
            "mag_valid": False,
            "mag_invalid_reason": "invalid_magnetic_field",
            "mag_time_delta_seconds": delta,
        }
    radius_km = float(np.linalg.norm(position))
    if radius_km <= 0.0:
        return {
            "mag_valid": False,
            "mag_invalid_reason": "invalid_position",
            "mag_time_delta_seconds": delta,
        }

    half_window = config.magnetic_window_seconds / 2.0
    first = int(np.searchsorted(times, target_unix - half_window, side="left"))
    last = int(np.searchsorted(times, target_unix + half_window, side="right"))
    window_fields = np.asarray(
        magnetic_geometry["magnetic_field_nT"][first:last],
        dtype=float,
    )
    window_fields = window_fields[np.all(np.isfinite(window_fields), axis=1)]
    window_norm = np.linalg.norm(window_fields, axis=1)
    window_norm = window_norm[np.isfinite(window_norm)]
    b_norm = float(np.linalg.norm(field))
    b_median = float(np.nanmedian(window_norm)) if window_norm.size else float("nan")
    b_relative_std = (
        float(np.nanstd(window_norm) / b_median)
        if window_norm.size >= 3 and b_median > 0.0
        else float("nan")
    )
    b_to_window_median_ratio = (
        b_norm / b_median if np.isfinite(b_median) and b_median > 0.0 else float("nan")
    )
    nonzero = np.linalg.norm(window_fields, axis=1) > 0.0
    unit_fields = window_fields[nonzero] / np.linalg.norm(
        window_fields[nonzero], axis=1
    )[:, None]
    mean_direction = np.nanmean(unit_fields, axis=0) if unit_fields.size else None
    if mean_direction is not None and np.linalg.norm(mean_direction) > 0.0:
        mean_direction = mean_direction / np.linalg.norm(mean_direction)
        direction_angles = np.degrees(
            np.arccos(np.clip(unit_fields @ mean_direction, -1.0, 1.0))
        )
        b_direction_dispersion = float(np.sqrt(np.nanmean(direction_angles**2)))
    else:
        b_direction_dispersion = float("nan")

    median_window_field = (
        np.nanmedian(window_fields, axis=0)
        if window_fields.size
        else np.full(3, np.nan, dtype=float)
    )
    median_window_norm = float(np.linalg.norm(median_window_field))
    b_tail_alignment = (
        abs(float(median_window_field[0])) / median_window_norm
        if np.isfinite(median_window_norm) and median_window_norm > 0.0
        else float("nan")
    )

    component_reversal = False
    if window_fields.shape[0] >= 3:
        component_min = np.nanmin(window_fields, axis=0)
        component_max = np.nanmax(window_fields, axis=0)
        component_range = component_max - component_min
        dominant = int(np.nanargmax(component_range))
        component_reversal = bool(
            component_min[dominant] < 0.0 < component_max[dominant]
        )

    center_half = config.current_sheet_center_half_window_seconds
    flank_width = config.current_sheet_flank_window_seconds
    pre_first = int(
        np.searchsorted(
            times,
            target_unix - center_half - flank_width,
            side="left",
        )
    )
    pre_last = int(
        np.searchsorted(times, target_unix - center_half, side="left")
    )
    center_first = int(
        np.searchsorted(times, target_unix - center_half, side="left")
    )
    center_last = int(
        np.searchsorted(times, target_unix + center_half, side="right")
    )
    post_first = int(
        np.searchsorted(times, target_unix + center_half, side="right")
    )
    post_last = int(
        np.searchsorted(
            times,
            target_unix + center_half + flank_width,
            side="right",
        )
    )

    def finite_fields(first_index: int, last_index: int) -> np.ndarray:
        values = np.asarray(
            magnetic_geometry["magnetic_field_nT"][first_index:last_index],
            dtype=float,
        )
        return values[np.all(np.isfinite(values), axis=1)]

    pre_fields = finite_fields(pre_first, pre_last)
    center_fields = finite_fields(center_first, center_last)
    post_fields = finite_fields(post_first, post_last)
    current_sheet_rotation = float("nan")
    current_sheet_dip = float("nan")
    center_min_b = float("nan")
    flank_median_b = float("nan")
    if pre_fields.size and center_fields.size and post_fields.size:
        pre_vector = np.nanmedian(pre_fields, axis=0)
        post_vector = np.nanmedian(post_fields, axis=0)
        pre_norm = float(np.linalg.norm(pre_vector))
        post_norm = float(np.linalg.norm(post_vector))
        if pre_norm > 0.0 and post_norm > 0.0:
            current_sheet_rotation = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(np.dot(pre_vector, post_vector))
                            / (pre_norm * post_norm),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        center_norms = np.linalg.norm(center_fields, axis=1)
        flank_norms = np.concatenate(
            (np.linalg.norm(pre_fields, axis=1), np.linalg.norm(post_fields, axis=1))
        )
        center_min_b = float(np.nanmin(center_norms))
        flank_median_b = float(np.nanmedian(flank_norms))
        if flank_median_b > 0.0:
            current_sheet_dip = center_min_b / flank_median_b

    sza_deg = float(
        np.degrees(np.arccos(np.clip(position[0] / radius_km, -1.0, 1.0)))
    )
    current_sheet = bool(
        sza_deg >= config.lobe_min_sza_deg
        and np.isfinite(current_sheet_rotation)
        and current_sheet_rotation >= config.current_sheet_min_rotation_deg
        and np.isfinite(current_sheet_dip)
        and current_sheet_dip <= config.current_sheet_b_dip_ratio
    )
    return {
        "mag_valid": True,
        "mag_invalid_reason": "",
        "position_km": position,
        "field_nT": field,
        "b_nT": b_norm,
        "b_window_median_nT": b_median,
        "b_to_window_median_ratio": b_to_window_median_ratio,
        "b_relative_std": b_relative_std,
        "b_direction_dispersion_deg": b_direction_dispersion,
        "b_tail_alignment": b_tail_alignment,
        "component_reversal": component_reversal,
        "current_sheet_signature": current_sheet,
        "current_sheet_rotation_deg": current_sheet_rotation,
        "current_sheet_dip_ratio": current_sheet_dip,
        "current_sheet_center_min_b_nT": center_min_b,
        "current_sheet_flank_median_b_nT": flank_median_b,
        "altitude_km": radius_km - MARS_RADIUS_KM,
        "sza_deg": sza_deg,
        "mag_time_delta_seconds": delta,
    }


def _copy_nearest_features(
    destination: dict[str, Any],
    feature_data: dict | None,
    target_unix: float,
    max_delta_seconds: float,
    keys: tuple[str, ...],
    delta_key: str,
) -> None:
    nearest = nearest_feature_index(feature_data, target_unix, max_delta_seconds)
    if nearest is None or feature_data is None:
        destination[delta_key] = float("nan")
        return
    index, delta = nearest
    destination[delta_key] = delta
    for key in keys:
        if key not in feature_data:
            continue
        value = np.asarray(feature_data[key])[index]
        destination[key] = value.item() if hasattr(value, "item") else value


def _empty_row(target_unix: float, bow_model_name: str) -> dict[str, Any]:
    row = {field: float("nan") for field in CSV_FIELDS}
    row.update(
        {
            "time_unix": float(target_unix),
            "time_utc": datetime.fromtimestamp(
                target_unix, tz=timezone.utc
            ).isoformat(timespec="seconds"),
            "region_id": 0,
            "region_name": REGION_NAMES[0],
            "confidence": 0.0,
            "reason": "missing_mag_position",
            "geometry_only": False,
            "mag_invalid_reason": "missing_mag_position",
            "bow_model": bow_model_name,
            "bow_location": "",
            "mpb_location": "",
            "current_sheet_signature": False,
            "photoelectron_present": "",
            "electron_void": "",
            "swe_valid": False,
            "static_valid": False,
        }
    )
    return row


def classify_interval(
    start: datetime,
    end: datetime,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    bow_model_name: str = DEFAULT_MODEL_NAME,
    config: RegionClassifierConfig | None = None,
    target_times_unix: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config or RegionClassifierConfig()
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if end <= start:
        raise ValueError("end must be later than start")
    if cfg.cadence_seconds <= 0.0:
        raise ValueError("cadence_seconds must be positive")

    if target_times_unix is None:
        sampling_mode = "fixed_cadence"
        target_times = np.arange(
            start.timestamp(),
            end.timestamp() + 1.0e-6,
            cfg.cadence_seconds,
            dtype=float,
        )
    else:
        sampling_mode = "explicit_times"
        target_times = np.asarray(target_times_unix, dtype=float).reshape(-1)
        if not np.all(np.isfinite(target_times)):
            raise ValueError("target_times_unix must contain only finite values")
        outside_interval = (
            (target_times < start.timestamp() - 1.0e-6)
            | (target_times > end.timestamp() + 1.0e-6)
        )
        if np.any(outside_interval):
            raise ValueError("target_times_unix values must stay within start/end")

    root = Path(data_root).expanduser().resolve()
    bow_model = get_model(bow_model_name)
    magnetic_padding_seconds = max(
        cfg.magnetic_window_seconds / 2.0,
        cfg.current_sheet_center_half_window_seconds
        + cfg.current_sheet_flank_window_seconds,
        cfg.max_mag_delta_seconds,
    )
    magnetic = load_magnetic_geometry_interval(
        root,
        start - timedelta(seconds=magnetic_padding_seconds),
        end + timedelta(seconds=magnetic_padding_seconds),
    )
    if magnetic is None:
        raise FileNotFoundError(
            "No usable MAG sunstate-1sec samples were found in the requested interval."
        )
    swe = load_swe_features_interval(
        root,
        start - timedelta(seconds=cfg.max_swe_delta_seconds),
        end + timedelta(seconds=cfg.max_swe_delta_seconds),
        photoelectron_ratio_threshold=cfg.photoelectron_ratio_threshold,
        electron_void_target_energy_eV=cfg.electron_void_target_energy_eV,
        electron_void_flux_threshold=cfg.electron_void_flux_threshold,
    )
    static = load_static_features_interval(
        root,
        start - timedelta(seconds=cfg.max_static_delta_seconds),
        end + timedelta(seconds=cfg.max_static_delta_seconds),
    )

    rows: list[dict[str, Any]] = []
    for target_unix in target_times:
        row = _empty_row(float(target_unix), bow_model.name)
        magnetic_sample = _sample_magnetic_context(magnetic, target_unix, cfg)
        if not _true(magnetic_sample.get("mag_valid")):
            row.update(magnetic_sample)
            region_id, confidence, reason = classify_region_sample(row, cfg)
            row.update(
                {
                    "region_id": region_id,
                    "region_name": REGION_NAMES[region_id],
                    "confidence": confidence,
                    "reason": reason,
                    "geometry_only": False,
                }
            )
            rows.append(row)
            continue

        position = np.asarray(magnetic_sample.pop("position_km"), dtype=float)
        field = np.asarray(magnetic_sample.pop("field_nT"), dtype=float)
        row.update(magnetic_sample)
        row["x_mso_km"], row["y_mso_km"], row["z_mso_km"] = position
        row["bx_nT"], row["by_nT"], row["bz_nT"] = field
        row["x_mso_rm"], row["y_mso_rm"], row["z_mso_rm"] = (
            position / MARS_RADIUS_KM
        )
        bow = evaluate_position(
            position,
            model=bow_model,
            boundary_tolerance_km=cfg.boundary_margin_km,
        )
        mpb = evaluate_position(
            position,
            model=VIGNES_2000_MPB,
            boundary_tolerance_km=cfg.boundary_margin_km,
        )
        row.update(
            {
                "bow_location": bow.location,
                "bow_radial_offset_km": bow.radial_offset_km,
                "mpb_location": mpb.location,
                "mpb_radial_offset_km": mpb.radial_offset_km,
            }
        )
        _copy_nearest_features(
            row,
            swe,
            target_unix,
            cfg.max_swe_delta_seconds,
            (
                "photoelectron_present",
                "photoelectron_ratio",
                "electron_void",
                "electron_flux_target",
                "high_energy_suppression_ratio",
            ),
            "swe_time_delta_seconds",
        )
        _copy_nearest_features(
            row,
            static,
            target_unix,
            cfg.max_static_delta_seconds,
            (
                "planetary_heavy_ion_flux_fraction",
                "heavy_ion_peak_energy_eV",
                "static_valid_bin_count",
                "planetary_heavy_ion_valid_bin_count",
                "total_valid_ion_flux",
                "planetary_heavy_ion_integrated_flux",
                "hplus_flux_fraction",
                "hplus_peak_energy_eV",
                "hplus_log_energy_width",
            ),
            "static_time_delta_seconds",
        )
        row["swe_valid"] = _finite(row.get("swe_time_delta_seconds"))
        row["static_valid"] = _finite(row.get("static_time_delta_seconds"))
        region_id, confidence, reason = classify_region_sample(row, cfg)
        row.update(
            {
                "region_id": region_id,
                "region_name": REGION_NAMES[region_id],
                "confidence": confidence,
                "reason": reason,
                "geometry_only": region_id in {1, 2},
            }
        )
        rows.append(row)

    reason_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    unknown_count = sum(int(row["region_id"]) == 0 for row in rows)

    metadata = {
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "sample_count": len(rows),
        "sampling_mode": sampling_mode,
        "data_root": str(root),
        "bow_model": bow_model.metadata(),
        "mpb_model": VIGNES_2000_MPB.metadata(),
        "config": asdict(cfg),
        "available_products": {
            "mag": True,
            "swea": swe is not None,
            "static": static is not None,
        },
        "source_files": {
            "mag": magnetic.get("source_files", []),
            "swea": [] if swe is None else swe.get("source_files", []),
            "static": [] if static is None else static.get("source_files", []),
        },
        "region_counts": {
            str(region_id): sum(
                int(row["region_id"]) == region_id for row in rows
            )
            for region_id in REGION_NAMES
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "unknown_count": unknown_count,
        "unknown_fraction": unknown_count / len(rows) if rows else 0.0,
        "geometry_only_count": sum(bool(row["geometry_only"]) for row in rows),
    }
    return rows, metadata


def write_region_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_summary_json(path: str | Path, metadata: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return output


def plot_region_ids(
    path: str | Path,
    rows: list[dict[str, Any]],
    title: str = "MAVEN plasma region",
) -> Path:
    if not rows:
        raise ValueError("At least one classified row is required for plotting.")
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "maven_region_id_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    times = [
        datetime.fromtimestamp(float(row["time_unix"]), tz=timezone.utc)
        for row in rows
    ]
    region_ids = np.asarray([int(row["region_id"]) for row in rows], dtype=int)
    colors = {
        0: "#6B7280",
        1: "#E0A21A",
        2: "#D65A4A",
        3: "#2E7D5B",
        4: "#3568A8",
    }

    fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
    ax.step(times, region_ids, where="mid", color="#252A31", linewidth=1.0)
    for region_id in REGION_NAMES:
        selected = region_ids == region_id
        if np.any(selected):
            selected_times = [time for time, keep in zip(times, selected) if keep]
            ax.scatter(
                selected_times,
                region_ids[selected],
                s=18,
                color=colors[region_id],
                edgecolors="none",
                label=f"{region_id} {REGION_NAMES[region_id]}",
                zorder=3,
            )

    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("region_id")
    ax.set_yticks(list(REGION_NAMES))
    ax.set_yticklabels(
        [f"{region_id}  {REGION_NAMES[region_id]}" for region_id in REGION_NAMES]
    )
    ax.set_ylim(-0.45, 4.45)
    ax.grid(axis="y", color="#D7DADE", linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=5,
        frameon=False,
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return output


def default_output_dir(start: datetime, end: datetime) -> Path:
    interval = (
        f"{start.strftime('%Y%m%dT%H%M%S')}_"
        f"{end.strftime('%Y%m%dT%H%M%S')}"
    )
    return REPO_ROOT / "outputs" / "region_id" / interval


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify MAVEN location into region_id 0-4 and plot region_id versus time."
        )
    )
    parser.add_argument("--start", required=True, help="UTC start time.")
    parser.add_argument("--end", required=True, help="UTC end time.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--bow-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cadence-seconds", type=float, default=10.0)
    parser.add_argument("--boundary-margin-km", type=float, default=100.0)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    start = parse_iso_timestamp(args.start).astimezone(timezone.utc)
    end = parse_iso_timestamp(args.end).astimezone(timezone.utc)
    config = RegionClassifierConfig(
        cadence_seconds=args.cadence_seconds,
        boundary_margin_km=args.boundary_margin_km,
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(start, end)
    )
    rows, metadata = classify_interval(
        start=start,
        end=end,
        data_root=args.data_root,
        bow_model_name=args.bow_model,
        config=config,
    )
    csv_path = write_region_csv(output_dir / "region_id_timeseries.csv", rows)
    plot_path = plot_region_ids(output_dir / "region_id_timeseries.png", rows)
    json_path = write_summary_json(output_dir / "region_id_summary.json", metadata)
    counts = ", ".join(
        f"{region_id}:{metadata['region_counts'][str(region_id)]}"
        for region_id in REGION_NAMES
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {json_path}")
    print(f"Region counts: {counts}")


if __name__ == "__main__":
    main()
