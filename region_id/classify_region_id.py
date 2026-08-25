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
    load_swia_moments_interval,
    load_swia_spectra_interval,
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
    max_swia_delta_seconds: float = 6.0
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
    swia_window_seconds: float = 60.0
    swia_min_valid_samples: int = 5
    swia_min_window_coverage_fraction: float = 0.60
    swia_max_valid_sample_gap_seconds: float = 12.0
    current_sheet_flank_window_seconds: float = 20.0
    current_sheet_center_half_window_seconds: float = 10.0
    current_sheet_min_rotation_deg: float = 90.0
    current_sheet_b_dip_ratio: float = 0.70
    photoelectron_ratio_threshold: float = 1.20
    electron_void_target_energy_eV: float = 40.0
    electron_void_flux_threshold: float = 1.0e5
    electron_depletion_lower_eV: float = 30.0
    electron_depletion_upper_eV: float = 80.0
    electron_depletion_flux_threshold: float = 1.0e5
    electron_depletion_min_valid_bins: int = 3
    electron_depletion_min_low_fraction: float = 0.75
    lobe_max_proton_density_cm3: float = 0.50
    upstream_search_window_seconds: float = 21600.0
    upstream_min_radius_rm: float = 2.50
    upstream_min_x_rm: float = 1.00
    upstream_min_speed_km_s: float = 300.0
    upstream_min_segment_samples: int = 15
    upstream_segment_max_gap_seconds: float = 12.0
    upstream_max_relative_spread: float = 0.35
    solar_wind_min_density_ratio: float = 0.70
    solar_wind_max_density_ratio: float = 1.30
    solar_wind_min_b_ratio: float = 0.70
    solar_wind_max_b_ratio: float = 1.30
    solar_wind_min_speed_ratio: float = 0.85
    solar_wind_max_speed_ratio: float = 1.15
    solar_wind_min_speed_km_s: float = 200.0
    solar_wind_max_thermal_to_bulk_ratio: float = 0.17
    solar_wind_max_b_relative_std: float = 0.15
    magnetosheath_min_density_ratio: float = 1.50
    magnetosheath_min_b_ratio: float = 2.00
    magnetosheath_min_speed_ratio: float = 0.35
    magnetosheath_max_speed_ratio: float = 0.85
    magnetosheath_max_speed_km_s: float = 300.0
    magnetosheath_min_thermal_to_bulk_ratio: float = 0.22
    magnetosheath_min_b_relative_std: float = 0.15
    magnetosheath_min_spectrum_log_width: float = 0.28
    magnetosheath_min_flow_deflection_deg: float = 20.0
    magnetosheath_reference_free_min_evidence: int = 3
    magnetosheath_min_hplus_fraction: float = 0.50
    region_evidence_min_score_separation: float = 0.08
    boundary_geometry_confidence_bonus: float = 0.03


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
    "structure_flags",
    "swia_density_cm3",
    "swia_speed_km_s",
    "swia_vx_mso_km_s",
    "swia_vy_mso_km_s",
    "swia_vz_mso_km_s",
    "swia_temperature_eV",
    "swia_proton_thermal_to_bulk_ratio",
    "swia_flow_deflection_deg",
    "swia_window_valid_sample_count",
    "swia_window_coverage_fraction",
    "swia_window_max_gap_seconds",
    "swia_moment_quality_valid",
    "swia_temperature_valid",
    "swia_spectrum_peak_energy_eV",
    "swia_spectrum_log_energy_width",
    "swia_spectrum_entropy",
    "swia_spectrum_valid_bin_count",
    "swia_spectrum_quality_valid",
    "upstream_reference_valid",
    "upstream_reference_source",
    "upstream_reference_age_seconds",
    "upstream_reference_relative_spread",
    "upstream_density_cm3",
    "upstream_speed_km_s",
    "upstream_b_nT",
    "density_to_upstream_ratio",
    "speed_to_upstream_ratio",
    "b_to_upstream_ratio",
    "solar_wind_normalized_signature",
    "solar_wind_reference_free_signature",
    "solar_wind_signature",
    "magnetosheath_normalized_signature",
    "magnetosheath_reference_free_signature",
    "magnetosheath_evidence_count",
    "magnetosheath_evidence",
    "planetary_ion_contradiction",
    "region_candidate_ids",
    "region_candidate_scores",
    "boundary_geometry_support",
    "boundary_geometry_confidence_bonus",
    "photoelectron_present",
    "photoelectron_ratio",
    "electron_void",
    "electron_flux_target",
    "high_energy_suppression_ratio",
    "electron_depletion_band_median_flux",
    "electron_depletion_low_fraction",
    "electron_depletion_valid_bin_count",
    "multichannel_electron_depletion",
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
    "swia_valid",
    "swia_spectrum_valid",
    "mag_time_delta_seconds",
    "swe_time_delta_seconds",
    "static_time_delta_seconds",
    "swia_time_delta_seconds",
    "swia_spectrum_time_delta_seconds",
]


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _true(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    if not _finite(numerator) or not _finite(denominator) or float(denominator) <= 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


def _sample_swia_context(
    moments: dict | None,
    spectra: dict | None,
    target_unix: float,
    config: RegionClassifierConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "swia_valid": False,
        "swia_temperature_valid": False,
        "swia_spectrum_valid": False,
        "swia_moment_quality_valid": False,
        "swia_spectrum_quality_valid": False,
        "swia_time_delta_seconds": float("nan"),
        "swia_spectrum_time_delta_seconds": float("nan"),
    }
    nearest = nearest_feature_index(moments, target_unix, config.max_swia_delta_seconds)
    if nearest is not None and moments is not None:
        nearest_index, nearest_delta = nearest
        result["swia_time_delta_seconds"] = nearest_delta
        nearest_valid = _true(
            np.asarray(moments["swia_moment_quality_valid"])[nearest_index]
        )
        result["swia_moment_quality_valid"] = nearest_valid
        times = np.asarray(moments["times_unix"], dtype=float)
        half_window = config.swia_window_seconds / 2.0
        first = int(np.searchsorted(times, target_unix - half_window, side="left"))
        last = int(np.searchsorted(times, target_unix + half_window, side="right"))
        window_valid = np.asarray(
            moments["swia_moment_quality_valid"][first:last], dtype=bool
        )
        window_times = times[first:last]
        valid_times = window_times[window_valid]
        all_deltas = np.diff(times)
        cadence_candidates = all_deltas[
            np.isfinite(all_deltas) & (all_deltas > 0.0) & (all_deltas <= 30.0)
        ]
        expected_cadence = (
            float(np.nanmedian(cadence_candidates))
            if cadence_candidates.size
            else 4.0
        )
        expected_count = max(1, int(np.floor(config.swia_window_seconds / expected_cadence)) + 1)
        coverage = min(1.0, float(valid_times.size) / expected_count)
        max_gap = (
            float(np.max(np.diff(valid_times)))
            if valid_times.size >= 2
            else float("inf")
        )
        result.update(
            {
                "swia_window_valid_sample_count": int(valid_times.size),
                "swia_window_coverage_fraction": coverage,
                "swia_window_max_gap_seconds": max_gap,
            }
        )
        context_valid = bool(
            nearest_valid
            and valid_times.size >= config.swia_min_valid_samples
            and coverage >= config.swia_min_window_coverage_fraction
            and max_gap <= config.swia_max_valid_sample_gap_seconds
        )
        if context_valid:
            density = np.asarray(moments["density_cm3"][first:last], dtype=float)[
                window_valid
            ]
            velocity = np.asarray(
                moments["velocity_mso_km_s"][first:last], dtype=float
            )[window_valid]
            speed = np.asarray(moments["speed_km_s"][first:last], dtype=float)[
                window_valid
            ]
            temperature_valid = np.asarray(
                moments["swia_temperature_valid"][first:last], dtype=bool
            ) & window_valid
            temperature = np.asarray(
                moments["temperature_eV"][first:last], dtype=float
            )[temperature_valid]
            median_velocity = np.nanmedian(velocity, axis=0)
            median_speed = float(np.nanmedian(speed))
            median_temperature = (
                float(np.nanmedian(temperature)) if temperature.size else float("nan")
            )
            thermal_ratio = (
                13.841 * np.sqrt(median_temperature) / median_speed
                if np.isfinite(median_temperature)
                and median_temperature > 0.0
                and median_speed > 0.0
                else float("nan")
            )
            flow_deflection = (
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(-float(median_velocity[0]) / median_speed, -1.0, 1.0)
                        )
                    )
                )
                if np.all(np.isfinite(median_velocity)) and median_speed > 0.0
                else float("nan")
            )
            result.update(
                {
                    "swia_valid": True,
                    "swia_density_cm3": float(np.nanmedian(density)),
                    "swia_speed_km_s": median_speed,
                    "swia_vx_mso_km_s": float(median_velocity[0]),
                    "swia_vy_mso_km_s": float(median_velocity[1]),
                    "swia_vz_mso_km_s": float(median_velocity[2]),
                    "swia_temperature_eV": median_temperature,
                    "swia_temperature_valid": bool(temperature.size),
                    "swia_proton_thermal_to_bulk_ratio": float(thermal_ratio),
                    "swia_flow_deflection_deg": flow_deflection,
                }
            )

    spectrum_nearest = nearest_feature_index(
        spectra,
        target_unix,
        config.max_swia_delta_seconds,
    )
    if spectrum_nearest is not None and spectra is not None:
        spectrum_index, spectrum_delta = spectrum_nearest
        spectrum_quality = _true(
            np.asarray(spectra["swia_spectrum_quality_valid"])[spectrum_index]
        )
        result["swia_spectrum_time_delta_seconds"] = spectrum_delta
        result["swia_spectrum_quality_valid"] = spectrum_quality
        for key in (
            "swia_spectrum_peak_energy_eV",
            "swia_spectrum_log_energy_width",
            "swia_spectrum_entropy",
            "swia_spectrum_valid_bin_count",
        ):
            value = np.asarray(spectra[key])[spectrum_index]
            result[key] = value.item() if hasattr(value, "item") else value
        result["swia_spectrum_valid"] = bool(
            spectrum_quality
            and _finite(result.get("swia_spectrum_log_energy_width"))
        )
    return result


def _build_upstream_segments(
    moments: dict | None,
    magnetic: dict,
    bow_model: AxisymmetricConicModel,
    config: RegionClassifierConfig,
) -> list[dict[str, float]]:
    """Find stable, high-confidence upstream intervals without assuming every orbit has one."""
    if moments is None:
        return []
    times = np.asarray(moments["times_unix"], dtype=float)
    quality = np.asarray(moments["swia_moment_quality_valid"], dtype=bool)
    density = np.asarray(moments["density_cm3"], dtype=float)
    speed = np.asarray(moments["speed_km_s"], dtype=float)
    mag_times = np.asarray(magnetic["times"], dtype=float)
    positions = np.asarray(magnetic["position_km"], dtype=float)
    fields = np.asarray(magnetic["magnetic_field_nT"], dtype=float)
    candidate_records: list[tuple[int, float, bool]] = []
    for index in np.where(quality & (speed >= config.upstream_min_speed_km_s))[0]:
        insertion = int(np.searchsorted(mag_times, times[index], side="left"))
        possible = [item for item in (insertion - 1, insertion) if 0 <= item < mag_times.size]
        if not possible:
            continue
        mag_index = min(possible, key=lambda item: abs(float(mag_times[item]) - times[index]))
        if abs(float(mag_times[mag_index]) - times[index]) > config.max_mag_delta_seconds:
            continue
        position = positions[mag_index]
        field = fields[mag_index]
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(field)):
            continue
        radius_rm = float(np.linalg.norm(position) / MARS_RADIUS_KM)
        if (
            radius_rm < config.upstream_min_radius_rm
            or position[0] / MARS_RADIUS_KM < config.upstream_min_x_rm
        ):
            continue
        geometry = evaluate_position(position, model=bow_model, boundary_tolerance_km=0.0)
        # The statistical bow shock is diagnostic here, not a hard upstream
        # gate. Segment quality, fast flow and the measured dayside/far-field
        # position determine whether the reference is usable.
        candidate_records.append(
            (
                int(index),
                float(np.linalg.norm(field)),
                geometry.location == "outside",
            )
        )

    segments: list[list[tuple[int, float, bool]]] = []
    for record in candidate_records:
        if not segments:
            segments.append([record])
            continue
        previous_index = segments[-1][-1][0]
        if times[record[0]] - times[previous_index] <= config.upstream_segment_max_gap_seconds:
            segments[-1].append(record)
        else:
            segments.append([record])

    result: list[dict[str, float]] = []
    for records in segments:
        if len(records) < config.upstream_min_segment_samples:
            continue
        indices = np.asarray([record[0] for record in records], dtype=int)
        b_values = np.asarray([record[1] for record in records], dtype=float)
        bow_outside_fraction = float(
            np.mean([record[2] for record in records])
        )
        medians = np.asarray(
            [np.nanmedian(density[indices]), np.nanmedian(speed[indices]), np.nanmedian(b_values)],
            dtype=float,
        )
        spreads = []
        for values, median in zip((density[indices], speed[indices], b_values), medians):
            spreads.append(
                float(
                    (np.nanpercentile(values, 90) - np.nanpercentile(values, 10))
                    / (2.0 * median)
                )
                if np.isfinite(median) and median > 0.0
                else float("inf")
            )
        relative_spread = float(max(spreads))
        if relative_spread > config.upstream_max_relative_spread:
            continue
        result.append(
            {
                "start_unix": float(times[indices[0]]),
                "end_unix": float(times[indices[-1]]),
                "mid_unix": float((times[indices[0]] + times[indices[-1]]) / 2.0),
                "sample_count": float(indices.size),
                "density_cm3": float(medians[0]),
                "speed_km_s": float(medians[1]),
                "b_nT": float(medians[2]),
                "relative_spread": relative_spread,
                "bow_model_outside_fraction": bow_outside_fraction,
            }
        )
    return result


def _upstream_reference_for_time(
    target_unix: float,
    segments: list[dict[str, float]],
    config: RegionClassifierConfig,
) -> dict[str, Any]:
    unavailable: dict[str, Any] = {
        "upstream_reference_valid": False,
        "upstream_reference_source": "unavailable",
        "upstream_reference_age_seconds": float("nan"),
        "upstream_reference_relative_spread": float("nan"),
    }
    containing = [
        segment
        for segment in segments
        if segment["start_unix"] <= target_unix <= segment["end_unix"]
    ]
    if containing:
        selected = containing[0]
        source = "local_upstream_segment"
        age = 0.0
        combined = [selected]
    else:
        before = [segment for segment in segments if segment["end_unix"] < target_unix]
        after = [segment for segment in segments if segment["start_unix"] > target_unix]
        previous = max(before, key=lambda item: item["end_unix"]) if before else None
        following = min(after, key=lambda item: item["start_unix"]) if after else None
        if (
            previous is not None
            and target_unix - previous["end_unix"]
            > config.upstream_search_window_seconds
        ):
            previous = None
        if (
            following is not None
            and following["start_unix"] - target_unix
            > config.upstream_search_window_seconds
        ):
            following = None
        if previous is not None and following is not None:
            pair_values = (
                (previous["density_cm3"], following["density_cm3"]),
                (previous["speed_km_s"], following["speed_km_s"]),
                (previous["b_nT"], following["b_nT"]),
            )
            disagreement = max(
                abs(first - second) / np.nanmedian([first, second])
                for first, second in pair_values
            )
            if disagreement > config.upstream_max_relative_spread:
                unavailable["upstream_reference_source"] = "inconsistent_bracketing_segments"
                unavailable["upstream_reference_relative_spread"] = float(disagreement)
                return unavailable
            combined = [previous, following]
            source = "bracketing_upstream_segments"
            age = min(target_unix - previous["end_unix"], following["start_unix"] - target_unix)
        elif previous is not None or following is not None:
            selected = previous if previous is not None else following
            assert selected is not None
            combined = [selected]
            source = "nearest_upstream_segment"
            age = min(
                abs(target_unix - selected["start_unix"]),
                abs(target_unix - selected["end_unix"]),
            )
        else:
            return unavailable

    return {
        "upstream_reference_valid": True,
        "upstream_reference_source": source,
        "upstream_reference_age_seconds": float(age),
        "upstream_reference_relative_spread": float(
            max(segment["relative_spread"] for segment in combined)
        ),
        "upstream_density_cm3": float(
            np.nanmedian([segment["density_cm3"] for segment in combined])
        ),
        "upstream_speed_km_s": float(np.nanmedian([segment["speed_km_s"] for segment in combined])),
        "upstream_b_nT": float(np.nanmedian([segment["b_nT"] for segment in combined])),
    }


def _derive_plasma_signatures(
    features: dict[str, Any],
    config: RegionClassifierConfig,
) -> None:
    density_ratio = _safe_ratio(
        features.get("swia_density_cm3"), features.get("upstream_density_cm3")
    )
    speed_ratio = _safe_ratio(
        features.get("swia_speed_km_s"), features.get("upstream_speed_km_s")
    )
    b_ratio = _safe_ratio(
        features.get("b_window_median_nT"), features.get("upstream_b_nT")
    )
    features["density_to_upstream_ratio"] = density_ratio
    features["speed_to_upstream_ratio"] = speed_ratio
    features["b_to_upstream_ratio"] = b_ratio

    normalized_solar_wind = bool(
        _true(features.get("swia_valid"))
        and _true(features.get("upstream_reference_valid"))
        and config.solar_wind_min_density_ratio
        <= density_ratio
        <= config.solar_wind_max_density_ratio
        and config.solar_wind_min_speed_ratio
        <= speed_ratio
        <= config.solar_wind_max_speed_ratio
        and config.solar_wind_min_b_ratio <= b_ratio <= config.solar_wind_max_b_ratio
    )
    absolute_solar_wind = bool(
        _true(features.get("swia_valid"))
        and _finite(features.get("swia_speed_km_s"))
        and float(features["swia_speed_km_s"]) >= config.solar_wind_min_speed_km_s
        and _finite(features.get("swia_proton_thermal_to_bulk_ratio"))
        and float(features["swia_proton_thermal_to_bulk_ratio"])
        <= config.solar_wind_max_thermal_to_bulk_ratio
        and _finite(features.get("b_relative_std"))
        and float(features["b_relative_std"]) <= config.solar_wind_max_b_relative_std
    )
    features["solar_wind_normalized_signature"] = normalized_solar_wind
    features["solar_wind_reference_free_signature"] = absolute_solar_wind
    features["solar_wind_signature"] = normalized_solar_wind or absolute_solar_wind

    normalized_sheath = bool(
        _true(features.get("swia_valid"))
        and _true(features.get("upstream_reference_valid"))
        and density_ratio >= config.magnetosheath_min_density_ratio
        and b_ratio >= config.magnetosheath_min_b_ratio
        and config.magnetosheath_min_speed_ratio
        <= speed_ratio
        <= config.magnetosheath_max_speed_ratio
    )
    evidence: list[str] = []
    if _true(features.get("swia_valid")):
        if (
            _finite(features.get("swia_speed_km_s"))
            and float(features["swia_speed_km_s"])
            <= config.magnetosheath_max_speed_km_s
        ):
            evidence.append("slow_flow")
        if (
            _true(features.get("swia_temperature_valid"))
            and _finite(features.get("swia_proton_thermal_to_bulk_ratio"))
            and float(features["swia_proton_thermal_to_bulk_ratio"])
            >= config.magnetosheath_min_thermal_to_bulk_ratio
        ):
            evidence.append("proton_heating")
        if (
            _finite(features.get("swia_flow_deflection_deg"))
            and float(features["swia_flow_deflection_deg"])
            >= config.magnetosheath_min_flow_deflection_deg
        ):
            evidence.append("flow_deflection")
    if (
        _finite(features.get("b_relative_std"))
        and float(features["b_relative_std"])
        >= config.magnetosheath_min_b_relative_std
    ):
        evidence.append("magnetic_fluctuations")
    if (
        _true(features.get("swia_spectrum_valid"))
        and _finite(features.get("swia_spectrum_log_energy_width"))
        and float(features["swia_spectrum_log_energy_width"])
        >= config.magnetosheath_min_spectrum_log_width
    ):
        evidence.append("broad_ion_spectrum")

    heavy_fraction = features.get("planetary_heavy_ion_flux_fraction")
    hplus_fraction = features.get("hplus_flux_fraction")
    contradiction = bool(
        _true(features.get("static_valid"))
        and _finite(heavy_fraction)
        and float(heavy_fraction) >= config.heavy_ion_fraction_threshold
        and _finite(hplus_fraction)
        and float(hplus_fraction) < config.magnetosheath_min_hplus_fraction
    )
    hplus_support = bool(
        _true(features.get("static_valid"))
        and _finite(hplus_fraction)
        and float(hplus_fraction) >= config.magnetosheath_min_hplus_fraction
    )
    if hplus_support:
        evidence.append("hplus_support")
    primary_count = len([item for item in evidence if item != "hplus_support"])
    reference_free = bool(
        _true(features.get("swia_valid"))
        and primary_count >= config.magnetosheath_reference_free_min_evidence
        and not contradiction
    )
    features["magnetosheath_normalized_signature"] = normalized_sheath
    features["magnetosheath_reference_free_signature"] = reference_free
    features["magnetosheath_evidence_count"] = primary_count
    features["magnetosheath_evidence"] = ";".join(evidence)
    features["planetary_ion_contradiction"] = contradiction


def classify_region_sample(
    features: dict[str, Any],
    config: RegionClassifierConfig | None = None,
) -> tuple[int, float, str]:
    """Classify from in-situ evidence; statistical boundaries only adjust confidence."""
    cfg = config or RegionClassifierConfig()
    features["region_candidate_ids"] = ""
    features["region_candidate_scores"] = ""
    features["boundary_geometry_support"] = ""
    features["boundary_geometry_confidence_bonus"] = 0.0
    if not _true(features.get("mag_valid")):
        return 0, 0.0, str(
            features.get("mag_invalid_reason") or "missing_mag_position"
        )

    # Candidate creation and ranking use in-situ measurements only. The bow-shock
    # and MPB models are deliberately absent from this stage.
    candidates: list[tuple[int, float, str]] = []

    normalized_solar_wind = _true(
        features.get("solar_wind_normalized_signature")
    )
    reference_free_solar_wind = _true(
        features.get("solar_wind_reference_free_signature")
    )
    # Compatibility for external callers that only supply the historical aggregate.
    if (
        _true(features.get("solar_wind_signature"))
        and not normalized_solar_wind
        and not reference_free_solar_wind
    ):
        normalized_solar_wind = _true(features.get("upstream_reference_valid"))
        reference_free_solar_wind = not normalized_solar_wind
    if normalized_solar_wind:
        candidates.append((1, 0.94, "normalized_solar_wind_plasma"))
    elif reference_free_solar_wind:
        candidates.append((1, 0.84, "local_cool_fast_stable_solar_wind_plasma"))

    if _true(features.get("magnetosheath_normalized_signature")):
        source = str(features.get("upstream_reference_source") or "")
        confidence = 0.90 if source == "local_upstream_segment" else 0.84
        support = "normalized_compressed_slow_magnetosheath_plasma"
        if "hplus_support" in str(features.get("magnetosheath_evidence") or ""):
            support += ";hplus_support"
            confidence = min(0.94, confidence + 0.02)
        candidates.append((2, confidence, support))
    elif _true(features.get("magnetosheath_reference_free_signature")):
        evidence_count = int(features.get("magnetosheath_evidence_count", 0))
        confidence = 0.82 if evidence_count >= 4 else 0.76
        candidates.append(
            (
                2,
                confidence,
                "reference_free_magnetosheath:"
                + str(features.get("magnetosheath_evidence") or ""),
            )
        )

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
        support = ["low_altitude", "cold_planetary_heavy_ions"]
        if photoelectrons:
            support.append("photoelectrons")
        confidence = min(0.99, 0.90 + 0.04 * (len(support) - 1))
        candidates.append((3, confidence, ";".join(support)))
    elif cold_heavy_ions:
        candidates.append((3, 0.82, "cold_planetary_heavy_ions"))
    elif low_altitude and photoelectrons:
        candidates.append((3, 0.94, "low_altitude;photoelectrons"))

    sza = float(features.get("sza_deg", np.nan))
    b_median = float(features.get("b_window_median_nT", np.nan))
    b_relative_std = float(features.get("b_relative_std", np.nan))
    b_direction_dispersion = float(
        features.get("b_direction_dispersion_deg", np.nan)
    )
    b_tail_alignment = float(features.get("b_tail_alignment", np.nan))
    lobe_position_context = (
        _finite(sza)
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
    lobe_particle_exclusion = _true(
        features.get("multichannel_electron_depletion")
    )
    proton_density_compatible = bool(
        not _true(features.get("swia_valid"))
        or (
            _finite(features.get("swia_density_cm3"))
            and float(features["swia_density_cm3"]) <= cfg.lobe_max_proton_density_cm3
        )
    )
    if (
        lobe_position_context
        and stable_lobe_field
        and lobe_particle_exclusion
        and proton_density_compatible
    ):
        support = [
            "nightside_high_altitude",
            "stable_5_25nT_field",
            "multichannel_electron_depletion",
        ]
        if _true(features.get("swia_valid")):
            support.append("low_proton_density")
            confidence = 0.88
        else:
            support.append("proton_density_unavailable")
            confidence = 0.82
        candidates.append((4, confidence, ";".join(support)))

    features["region_candidate_ids"] = ";".join(
        str(candidate[0]) for candidate in candidates
    )
    features["region_candidate_scores"] = ";".join(
        f"{candidate[0]}:{candidate[1]:.2f}" for candidate in candidates
    )

    if not candidates:
        if lobe_position_context and stable_lobe_field and lobe_particle_exclusion:
            return 0, 0.35, "lobe_evidence_but_proton_density_too_high"
        if lobe_position_context and stable_lobe_field:
            return 0, 0.30, "stable_lobe_field_without_particle_exclusion"
        if lobe_position_context:
            return 0, 0.30, "nightside_high_altitude_without_stable_lobe_field"
        if low_altitude:
            return 0, 0.30, "low_altitude_without_ionospheric_particle_evidence"
        if _true(features.get("planetary_ion_contradiction")):
            return 0, 0.35, "planetary_ions_contradict_magnetosheath"
        if features.get("bow_location") == "outside":
            return 0, 0.30, "outside_bow_shock_without_solar_wind_plasma_evidence"
        if (
            features.get("bow_location") == "inside"
            and features.get("mpb_location") == "outside"
        ):
            if not _true(features.get("swia_valid")):
                return 0, 0.20, "sheath_geometry_without_valid_swia_plasma"
            return 0, 0.35, "sheath_geometry_without_sufficient_magnetosheath_evidence"
        return 0, 0.25, "unresolved_by_available_observational_features"

    # Statistical boundaries never resolve conflicting in-situ candidates.
    ranked = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
    if (
        len(ranked) > 1
        and ranked[0][1] - ranked[1][1] < cfg.region_evidence_min_score_separation
    ):
        conflict = ";".join(
            f"ID{candidate[0]}={candidate[1]:.2f}" for candidate in ranked
        )
        return 0, 0.40, "conflicting_region_evidence:" + conflict

    region_id, evidence_confidence, evidence_reason = ranked[0]
    bow_offset = features.get("bow_radial_offset_km")
    mpb_offset = features.get("mpb_radial_offset_km")
    bow_is_clear = bool(
        _finite(bow_offset) and abs(float(bow_offset)) > cfg.boundary_margin_km
    )
    mpb_is_clear = bool(
        _finite(mpb_offset) and abs(float(mpb_offset)) > cfg.boundary_margin_km
    )
    geometry_support = ""
    if region_id == 1 and bow_is_clear and features.get("bow_location") == "outside":
        geometry_support = "bow_model_support"
    elif (
        region_id == 2
        and bow_is_clear
        and mpb_is_clear
        and features.get("bow_location") == "inside"
        and features.get("mpb_location") == "outside"
    ):
        geometry_support = "bow_and_mpb_models_support"
    elif (
        region_id in (3, 4)
        and mpb_is_clear
        and features.get("mpb_location") == "inside"
    ):
        geometry_support = "mpb_model_support"

    if geometry_support:
        bonus = max(0.0, min(float(cfg.boundary_geometry_confidence_bonus), 0.05))
        features["boundary_geometry_support"] = geometry_support
        features["boundary_geometry_confidence_bonus"] = bonus
        return (
            region_id,
            min(0.99, evidence_confidence + bonus),
            evidence_reason + ";" + geometry_support,
        )

    near_boundary = bool(
        (_finite(bow_offset) and abs(float(bow_offset)) <= cfg.boundary_margin_km)
        or (_finite(mpb_offset) and abs(float(mpb_offset)) <= cfg.boundary_margin_km)
    )
    boundary_note = (
        "statistical_boundary_nearby_no_bonus"
        if near_boundary
        else "statistical_boundary_models_not_supportive"
    )
    return region_id, evidence_confidence, evidence_reason + ";" + boundary_note


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
            "structure_flags": "",
            "photoelectron_present": "",
            "electron_void": "",
            "multichannel_electron_depletion": False,
            "swe_valid": False,
            "static_valid": False,
            "swia_valid": False,
            "swia_spectrum_valid": False,
            "upstream_reference_valid": False,
            "upstream_reference_source": "unavailable",
            "solar_wind_normalized_signature": False,
            "solar_wind_reference_free_signature": False,
            "solar_wind_signature": False,
            "magnetosheath_normalized_signature": False,
            "magnetosheath_reference_free_signature": False,
            "planetary_ion_contradiction": False,
            "magnetosheath_evidence": "",
            "region_candidate_ids": "",
            "region_candidate_scores": "",
            "boundary_geometry_support": "",
            "boundary_geometry_confidence_bonus": 0.0,
        }
    )
    return row


def classify_interval(
    start: datetime,
    end: datetime,
    data_root: str | Path | tuple[str | Path, ...] | list[str | Path] = DEFAULT_DATA_ROOT,
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
    if cfg.boundary_margin_km < 0.0:
        raise ValueError("boundary_margin_km must be nonnegative")
    if cfg.boundary_geometry_confidence_bonus < 0.0:
        raise ValueError("boundary_geometry_confidence_bonus must be nonnegative")
    if cfg.region_evidence_min_score_separation < 0.0:
        raise ValueError("region_evidence_min_score_separation must be nonnegative")

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

    if isinstance(data_root, (str, Path)):
        roots = (Path(data_root).expanduser().resolve(),)
    else:
        roots = tuple(Path(value).expanduser().resolve() for value in data_root)
    if not roots:
        raise ValueError("At least one MAVEN data root is required")
    root: Path | tuple[Path, ...] = roots[0] if len(roots) == 1 else roots
    bow_model = get_model(bow_model_name)
    local_magnetic_padding_seconds = max(
        cfg.magnetic_window_seconds / 2.0,
        cfg.current_sheet_center_half_window_seconds
        + cfg.current_sheet_flank_window_seconds,
        cfg.max_mag_delta_seconds,
    )
    upstream_padding_seconds = max(
        local_magnetic_padding_seconds,
        cfg.upstream_search_window_seconds,
    )
    magnetic = load_magnetic_geometry_interval(
        root,
        start - timedelta(seconds=upstream_padding_seconds),
        end + timedelta(seconds=upstream_padding_seconds),
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
        electron_depletion_lower_eV=cfg.electron_depletion_lower_eV,
        electron_depletion_upper_eV=cfg.electron_depletion_upper_eV,
        electron_depletion_flux_threshold=cfg.electron_depletion_flux_threshold,
        electron_depletion_min_valid_bins=cfg.electron_depletion_min_valid_bins,
        electron_depletion_min_low_fraction=cfg.electron_depletion_min_low_fraction,
    )
    static = load_static_features_interval(
        root,
        start - timedelta(seconds=cfg.max_static_delta_seconds),
        end + timedelta(seconds=cfg.max_static_delta_seconds),
    )
    swia_moments = load_swia_moments_interval(
        root,
        start - timedelta(seconds=cfg.upstream_search_window_seconds),
        end + timedelta(seconds=cfg.upstream_search_window_seconds),
    )
    swia_spectra = load_swia_spectra_interval(
        root,
        start - timedelta(seconds=cfg.max_swia_delta_seconds),
        end + timedelta(seconds=cfg.max_swia_delta_seconds),
    )
    upstream_segments = _build_upstream_segments(
        swia_moments,
        magnetic,
        bow_model,
        cfg,
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
                "electron_depletion_band_median_flux",
                "electron_depletion_low_fraction",
                "electron_depletion_valid_bin_count",
                "multichannel_electron_depletion",
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
        row.update(_sample_swia_context(swia_moments, swia_spectra, target_unix, cfg))
        row.update(_upstream_reference_for_time(target_unix, upstream_segments, cfg))
        _derive_plasma_signatures(row, cfg)
        row["structure_flags"] = (
            "current_sheet" if _true(row.get("current_sheet_signature")) else ""
        )
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
        "data_root": str(roots[0]) if len(roots) == 1 else [str(path) for path in roots],
        "data_roots": [str(path) for path in roots],
        "bow_model": bow_model.metadata(),
        "mpb_model": VIGNES_2000_MPB.metadata(),
        "config": asdict(cfg),
        "available_products": {
            "mag": True,
            "swea": swe is not None,
            "static": static is not None,
            "swia_moments": swia_moments is not None,
            "swia_spectra": swia_spectra is not None,
        },
        "source_files": {
            "mag": magnetic.get("source_files", []),
            "swea": [] if swe is None else swe.get("source_files", []),
            "static": [] if static is None else static.get("source_files", []),
            "swia_moments": (
                [] if swia_moments is None else swia_moments.get("source_files", [])
            ),
            "swia_spectra": (
                [] if swia_spectra is None else swia_spectra.get("source_files", [])
            ),
        },
        "upstream_reference": {
            "segment_count": len(upstream_segments),
            "segments": upstream_segments,
            "optional": True,
            "note": (
                "A missing upstream segment does not prevent magnetosheath "
                "classification; the reference-free local evidence path is used."
            ),
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
        "method_notes": {
            "geometry": (
                "Bow shock and MPB models cannot create, reject, or rank region "
                "candidates; consistent geometry only adds a small confidence bonus."
            ),
            "swia_temperature": (
                "SWIA onboard proton temperature is auxiliary because alpha "
                "particles and field-of-view coverage can bias it."
            ),
            "static_fraction": (
                "STATIC H+ and heavy-ion fractions are ratios of summed valid "
                "energy-flux array values, used only as support/contradiction."
            ),
            "current_sheet": (
                "Current-sheet detection is retained as a structure flag and does "
                "not overwrite the background region_id."
            ),
        },
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
    parser.add_argument(
        "--boundary-geometry-confidence-bonus",
        type=float,
        default=0.03,
        help="Post-selection bow-shock/MPB confidence bonus (hard-capped at 0.05).",
    )
    parser.add_argument(
        "--region-evidence-min-score-separation",
        type=float,
        default=0.08,
        help="Minimum top-two observational score gap; smaller gaps return ID 0.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    start = parse_iso_timestamp(args.start).astimezone(timezone.utc)
    end = parse_iso_timestamp(args.end).astimezone(timezone.utc)
    config = RegionClassifierConfig(
        cadence_seconds=args.cadence_seconds,
        boundary_margin_km=args.boundary_margin_km,
        boundary_geometry_confidence_bonus=(
            args.boundary_geometry_confidence_bonus
        ),
        region_evidence_min_score_separation=(
            args.region_evidence_min_score_separation
        ),
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
