from __future__ import annotations

import numpy as np

from region_id.classify_region_id import (
    VIGNES_2000_MPB,
    RegionClassifierConfig,
    _build_upstream_segments,
    _derive_plasma_signatures,
    _sample_magnetic_context,
    _upstream_reference_for_time,
    classify_region_sample,
)


def sample(**updates):
    values = {
        "mag_valid": True,
        "bow_location": "inside",
        "bow_radial_offset_km": -500.0,
        "mpb_location": "outside",
        "mpb_radial_offset_km": 500.0,
        "altitude_km": 1000.0,
        "sza_deg": 60.0,
        "b_window_median_nT": 10.0,
        "b_relative_std": 0.10,
        "b_direction_dispersion_deg": 8.0,
        "b_tail_alignment": 0.80,
        "current_sheet_signature": False,
        "solar_wind_normalized_signature": False,
        "solar_wind_reference_free_signature": False,
        "solar_wind_signature": False,
        "magnetosheath_normalized_signature": False,
        "magnetosheath_reference_free_signature": False,
        "magnetosheath_evidence_count": 0,
        "magnetosheath_evidence": "",
        "planetary_ion_contradiction": False,
        "upstream_reference_valid": True,
        "upstream_reference_source": "local_upstream_segment",
        "swia_valid": True,
        "swia_density_cm3": 0.1,
        "photoelectron_present": False,
        "electron_void": False,
        "multichannel_electron_depletion": False,
        "planetary_heavy_ion_flux_fraction": 0.05,
        "heavy_ion_peak_energy_eV": 100.0,
    }
    values.update(updates)
    return values


def test_solar_wind_is_outside_bow_shock():
    region_id, _, _ = classify_region_sample(
        sample(
            bow_location="outside",
            bow_radial_offset_km=500.0,
            solar_wind_signature=True,
        )
    )
    assert region_id == 1


def test_outside_bow_shock_without_plasma_support_is_unknown():
    region_id, _, reason = classify_region_sample(
        sample(bow_location="outside", bow_radial_offset_km=500.0)
    )
    assert region_id == 0
    assert reason == "outside_bow_shock_without_solar_wind_plasma_evidence"


def test_bow_shock_margin_does_not_override_plasma_evidence():
    region_id, confidence, reason = classify_region_sample(
        sample(
            bow_location="on_boundary",
            bow_radial_offset_km=20.0,
            magnetosheath_normalized_signature=True,
        )
    )
    assert region_id == 2
    assert confidence == 0.90
    assert "statistical_boundary_nearby_no_bonus" in reason


def test_specific_mag_invalid_reason_is_preserved():
    region_id, confidence, reason = classify_region_sample(
        sample(
            mag_valid=False,
            mag_invalid_reason="mag_time_mismatch",
        )
    )
    assert region_id == 0
    assert confidence == 0.0
    assert reason == "mag_time_mismatch"


def test_magnetosheath_is_between_bow_shock_and_mpb():
    region_id, confidence, reason = classify_region_sample(
        sample(magnetosheath_normalized_signature=True)
    )
    assert region_id == 2
    assert confidence == 0.93
    assert "bow_and_mpb_models_support" in reason


def test_sheath_geometry_without_swia_is_unknown():
    region_id, _, reason = classify_region_sample(
        sample(
            magnetosheath_normalized_signature=False,
            swia_valid=False,
        )
    )
    assert region_id == 0
    assert reason == "sheath_geometry_without_valid_swia_plasma"


def test_reference_free_magnetosheath_does_not_require_upstream_segment():
    region_id, confidence, reason = classify_region_sample(
        sample(
            magnetosheath_normalized_signature=False,
            magnetosheath_reference_free_signature=True,
            magnetosheath_evidence_count=4,
            magnetosheath_evidence=(
                "slow_flow;proton_heating;magnetic_fluctuations;flow_deflection"
            ),
            upstream_reference_valid=False,
            upstream_reference_source="unavailable",
        )
    )
    assert region_id == 2
    assert confidence == 0.85
    assert "reference_free_magnetosheath" in reason


def test_low_altitude_without_particle_evidence_is_unknown():
    region_id, _, reason = classify_region_sample(
        sample(
            mpb_location="inside",
            mpb_radial_offset_km=-400.0,
            altitude_km=250.0,
        )
    )
    assert region_id == 0
    assert reason == "low_altitude_without_ionospheric_particle_evidence"


def test_low_altitude_with_photoelectrons_is_ionosphere():
    region_id, _, _ = classify_region_sample(
        sample(
            mpb_location="inside",
            mpb_radial_offset_km=-400.0,
            altitude_km=250.0,
            photoelectron_present=True,
        )
    )
    assert region_id == 3


def test_cold_heavy_ions_support_ionosphere_above_altitude_cutoff():
    region_id, _, _ = classify_region_sample(
        sample(
            mpb_location="inside",
            mpb_radial_offset_km=-400.0,
            altitude_km=550.0,
            planetary_heavy_ion_flux_fraction=0.70,
            heavy_ion_peak_energy_eV=8.0,
        )
    )
    assert region_id == 3


def test_magnetosheath_and_ionosphere_evidence_conflict_is_unknown():
    region_id, _, reason = classify_region_sample(
        sample(
            altitude_km=250.0,
            photoelectron_present=True,
            planetary_heavy_ion_flux_fraction=0.70,
            heavy_ion_peak_energy_eV=8.0,
            magnetosheath_normalized_signature=True,
        )
    )
    assert region_id == 0
    assert reason.startswith("conflicting_region_evidence:")


def test_bow_model_disagreement_cannot_reject_solar_wind_evidence():
    features = sample(
        bow_location="inside",
        bow_radial_offset_km=-500.0,
        solar_wind_normalized_signature=True,
        solar_wind_signature=True,
    )
    region_id, confidence, reason = classify_region_sample(features)
    assert region_id == 1
    assert confidence == 0.94
    assert reason.endswith("statistical_boundary_models_not_supportive")
    assert features["boundary_geometry_confidence_bonus"] == 0.0


def test_boundary_bonus_is_hard_capped_as_minor_support():
    features = sample(
        bow_location="outside",
        bow_radial_offset_km=500.0,
        solar_wind_reference_free_signature=True,
        solar_wind_signature=True,
    )
    region_id, confidence, _ = classify_region_sample(
        features,
        RegionClassifierConfig(boundary_geometry_confidence_bonus=0.50),
    )
    assert region_id == 1
    assert confidence == 0.89
    assert features["boundary_geometry_confidence_bonus"] == 0.05


def test_bow_model_disagreement_cannot_reject_magnetosheath_evidence():
    features = sample(
        bow_location="outside",
        bow_radial_offset_km=500.0,
        magnetosheath_normalized_signature=True,
    )
    region_id, confidence, reason = classify_region_sample(features)
    assert region_id == 2
    assert confidence == 0.90
    assert reason.endswith("statistical_boundary_models_not_supportive")


def test_mpb_model_disagreement_cannot_reject_ionosphere_evidence():
    region_id, confidence, reason = classify_region_sample(
        sample(
            mpb_location="outside",
            mpb_radial_offset_km=500.0,
            altitude_km=250.0,
            photoelectron_present=True,
        )
    )
    assert region_id == 3
    assert confidence == 0.94
    assert reason.endswith("statistical_boundary_models_not_supportive")


def test_mpb_model_disagreement_cannot_reject_lobe_evidence():
    region_id, confidence, reason = classify_region_sample(
        sample(
            mpb_location="outside",
            mpb_radial_offset_km=500.0,
            altitude_km=900.0,
            sza_deg=140.0,
            multichannel_electron_depletion=True,
        )
    )
    assert region_id == 4
    assert confidence == 0.88
    assert reason.endswith("statistical_boundary_models_not_supportive")


def test_stable_nightside_inner_field_is_lobe():
    region_id, _, _ = classify_region_sample(
        sample(
            mpb_location="inside",
            mpb_radial_offset_km=-600.0,
            altitude_km=900.0,
            sza_deg=140.0,
            multichannel_electron_depletion=True,
        )
    )
    assert region_id == 4


def test_stable_lobe_field_without_particle_exclusion_is_unknown():
    region_id, _, reason = classify_region_sample(
        sample(
            mpb_location="inside",
            mpb_radial_offset_km=-600.0,
            altitude_km=900.0,
            sza_deg=140.0,
            photoelectron_present="",
        )
    )
    assert region_id == 0
    assert reason == "stable_lobe_field_without_particle_exclusion"


def test_stable_nontail_field_is_not_lobe():
    region_id, _, reason = classify_region_sample(
        sample(
            mpb_location="inside",
            mpb_radial_offset_km=-600.0,
            altitude_km=900.0,
            sza_deg=140.0,
            electron_void=True,
            multichannel_electron_depletion=True,
            b_tail_alignment=0.20,
        )
    )
    assert region_id == 0
    assert reason == "nightside_high_altitude_without_stable_lobe_field"


def test_current_sheet_is_unknown():
    region_id, _, _ = classify_region_sample(
        sample(
            mpb_location="inside",
            mpb_radial_offset_km=-600.0,
            altitude_km=900.0,
            sza_deg=140.0,
            current_sheet_signature=True,
        )
    )
    assert region_id == 0


def test_current_sheet_flag_does_not_override_background_region():
    region_id, _, reason = classify_region_sample(
        sample(
            mpb_location="inside",
            mpb_radial_offset_km=-600.0,
            altitude_km=350.0,
            sza_deg=140.0,
            current_sheet_signature=True,
            planetary_heavy_ion_flux_fraction=0.70,
            heavy_ion_peak_energy_eV=8.0,
        )
    )
    assert region_id == 3
    assert "cold_planetary_heavy_ions" in reason


def test_single_channel_electron_void_is_not_enough_for_lobe():
    region_id, _, reason = classify_region_sample(
        sample(
            mpb_location="inside",
            mpb_radial_offset_km=-600.0,
            altitude_km=900.0,
            sza_deg=140.0,
            electron_void=True,
            multichannel_electron_depletion=False,
        )
    )
    assert region_id == 0
    assert reason == "stable_lobe_field_without_particle_exclusion"


def test_current_sheet_window_detects_offset_field_dip_and_rotation():
    times = np.arange(-40.0, 41.0)
    fields = np.tile(np.array([10.0, 0.0, 0.0]), (times.size, 1))
    fields[times > 5.0, 0] = -10.0
    fields[times == 5.0, 0] = 1.0
    positions = np.tile(np.array([-4500.0, 0.0, 0.0]), (times.size, 1))
    context = _sample_magnetic_context(
        {
            "times": times,
            "magnetic_field_nT": fields,
            "position_km": positions,
        },
        target_unix=0.0,
        config=RegionClassifierConfig(),
    )
    assert context["current_sheet_signature"] is True
    assert context["current_sheet_rotation_deg"] >= 179.0
    assert context["current_sheet_dip_ratio"] <= 0.11


def test_unresolved_dayside_inside_mpb_is_unknown():
    region_id, _, _ = classify_region_sample(
        sample(
            mpb_location="inside",
            mpb_radial_offset_km=-400.0,
            altitude_km=900.0,
            sza_deg=45.0,
        )
    )
    assert region_id == 0


def test_local_sheath_evidence_is_derived_when_upstream_is_unavailable():
    features = sample(
        upstream_reference_valid=False,
        swia_valid=True,
        swia_temperature_valid=True,
        swia_speed_km_s=180.0,
        swia_temperature_eV=80.0,
        swia_proton_thermal_to_bulk_ratio=0.69,
        swia_flow_deflection_deg=35.0,
        b_relative_std=0.25,
        swia_spectrum_valid=True,
        swia_spectrum_log_energy_width=0.34,
        static_valid=False,
    )
    _derive_plasma_signatures(features, RegionClassifierConfig())
    assert features["magnetosheath_reference_free_signature"] is True
    assert features["magnetosheath_normalized_signature"] is False
    assert features["magnetosheath_evidence_count"] == 5


def test_inconsistent_bracketing_upstream_segments_are_rejected():
    segments = [
        {
            "start_unix": 0.0,
            "end_unix": 100.0,
            "mid_unix": 50.0,
            "density_cm3": 1.0,
            "speed_km_s": 400.0,
            "b_nT": 4.0,
            "relative_spread": 0.1,
        },
        {
            "start_unix": 200.0,
            "end_unix": 300.0,
            "mid_unix": 250.0,
            "density_cm3": 3.0,
            "speed_km_s": 400.0,
            "b_nT": 4.0,
            "relative_spread": 0.1,
        },
    ]
    reference = _upstream_reference_for_time(
        150.0,
        segments,
        RegionClassifierConfig(upstream_search_window_seconds=1000.0),
    )
    assert reference["upstream_reference_valid"] is False
    assert reference["upstream_reference_source"] == "inconsistent_bracketing_segments"


def test_statistical_boundary_does_not_gate_upstream_segment_creation():
    times = np.arange(15, dtype=float) * 4.0
    moments = {
        "times_unix": times,
        "swia_moment_quality_valid": np.ones(times.size, dtype=bool),
        "density_cm3": np.ones(times.size, dtype=float),
        "speed_km_s": np.full(times.size, 400.0),
    }
    magnetic = {
        "times": times,
        "position_km": np.tile(np.array([3389.5, 0.0, 0.0]), (times.size, 1)),
        "magnetic_field_nT": np.tile(np.array([5.0, 0.0, 0.0]), (times.size, 1)),
    }
    segments = _build_upstream_segments(
        moments,
        magnetic,
        VIGNES_2000_MPB,
        RegionClassifierConfig(
            upstream_min_radius_rm=0.5,
            upstream_min_x_rm=0.5,
        ),
    )
    assert len(segments) == 1
    assert segments[0]["bow_model_outside_fraction"] == 0.0
