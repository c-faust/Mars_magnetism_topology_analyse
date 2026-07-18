from __future__ import annotations

import numpy as np

from region_id.classify_region_id import (
    RegionClassifierConfig,
    _sample_magnetic_context,
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
        "photoelectron_present": False,
        "electron_void": False,
        "planetary_heavy_ion_flux_fraction": 0.05,
        "heavy_ion_peak_energy_eV": 100.0,
    }
    values.update(updates)
    return values


def test_solar_wind_is_outside_bow_shock():
    region_id, _, _ = classify_region_sample(
        sample(bow_location="outside", bow_radial_offset_km=500.0)
    )
    assert region_id == 1


def test_bow_shock_margin_is_unknown():
    region_id, _, reason = classify_region_sample(
        sample(bow_location="on_boundary", bow_radial_offset_km=20.0)
    )
    assert region_id == 0
    assert reason == "near_bow_shock"


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
    region_id, _, _ = classify_region_sample(sample())
    assert region_id == 2


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


def test_magnetosheath_geometry_overrides_ionosphere_features():
    region_id, _, _ = classify_region_sample(
        sample(
            altitude_km=250.0,
            photoelectron_present=True,
            planetary_heavy_ion_flux_fraction=0.70,
            heavy_ion_peak_energy_eV=8.0,
        )
    )
    assert region_id == 2


def test_stable_nightside_inner_field_is_lobe():
    region_id, _, _ = classify_region_sample(
        sample(
            mpb_location="inside",
            mpb_radial_offset_km=-600.0,
            altitude_km=900.0,
            sza_deg=140.0,
            electron_void=True,
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
            b_tail_alignment=0.20,
        )
    )
    assert region_id == 0
    assert reason == "nightside_inner_region_without_stable_lobe_field"


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


def test_current_sheet_overrides_cold_heavy_ions():
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
    assert region_id == 0
    assert reason == "nightside_current_sheet_signature"


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
