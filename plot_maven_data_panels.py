from __future__ import annotations
"""
Render the magnetic-topology data panels as a static PNG.

This is the Python counterpart of `magnetic_topology_data_panels.html`: it reads
either a `topology_summary.json`-shaped dictionary or the local MAVEN daily data,
picks a target time, and draws the same science context panels around that time.

Panel catalog (use these stable IDs with ``--panels``):
  1  STATIC ion energy spectrogram
  2  STATIC light-ion mass spectrogram (0.5-1.5 amu)
  3  STATIC heavy-ion mass spectrogram (>1.5 amu)
  4  SWEA omnidirectional electron energy spectrogram
  5  MAG field magnitude |B|
  6  MAG field components Bx, By, Bz in MSO
  7  SWEA PAD for energy band 1 (default 20-80 eV)
  8  SWEA PAD for energy band 2 (default 111-140 eV)
  9  Bottom UTC/position/latitude/longitude/altitude annotations
 10  region_id classification versus time

Panel 9 is a coordinate annotation footer and is always placed last.

Examples:
  --panels 1 4 5 6 9
  --panels 4,7,8,10,9
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cdflib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from analyze_magnetic_topology import (
    build_mag_context,
    derive_pitch_bins,
    iter_days,
    load_mag_day,
    resolve_daily_files,
    select_time_indices,
)
from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from process_maven_spectra import load_pad_data


MARS_RADIUS_KM = 3389.5
LINE_COLORS = {"bx": "#cc4338", "by": "#2674c8", "bz": "#3a8a53", "bmag": "#6e5b4f"}
REGION_ID_COLORS = {
    0: "#6B7280",
    1: "#E0A21A",
    2: "#D65A4A",
    3: "#2E7D5B",
    4: "#3568A8",
}
PAD_CMAP = "turbo"
FLUX_CMAP = "inferno"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "maven_data_panels"
DEFAULT_PAD_ENERGY_BANDS_EV = ((20.0, 80.0), (111.0, 140.0))
DEFAULT_PAD_ENERGY_BAND_EV = DEFAULT_PAD_ENERGY_BANDS_EV[0]
STATIC_MASS_SPLIT_AMU = 1.5

# Keep IDs stable: command lines and downstream pipelines may persist them.
PANEL_CATALOG = {
    1: {"key": "static_energy", "name": "STATIC ion energy", "height_ratio": 1.15},
    2: {"key": "static_mass_light", "name": "STATIC light-ion mass", "height_ratio": 1.0},
    3: {"key": "static_mass_heavy", "name": "STATIC heavy-ion mass", "height_ratio": 1.0},
    4: {"key": "swe_energy", "name": "SWEA electron energy", "height_ratio": 1.15},
    5: {"key": "mag_magnitude", "name": "MAG |B|", "height_ratio": 0.8},
    6: {"key": "mag_components", "name": "MAG Bx/By/Bz MSO", "height_ratio": 0.8},
    7: {"key": "swe_pad_band_1", "name": "SWEA PAD band 1", "height_ratio": 1.05},
    8: {"key": "swe_pad_band_2", "name": "SWEA PAD band 2", "height_ratio": 1.05},
    9: {"key": "coordinates", "name": "UTC and spacecraft coordinates", "height_ratio": 0.78},
    10: {"key": "region_id", "name": "region_id classification", "height_ratio": 0.8},
}
DEFAULT_PANEL_IDS = tuple(range(1, 10))
PAD_PANEL_BAND_INDEX = {7: 0, 8: 1}
COORDINATE_PANEL_ID = 9
REGION_ID_PANEL_ID = 10


def log_step(message: str) -> None:
    print(f"[data-panels] {datetime.now().isoformat(timespec='seconds')} | {message}", flush=True)


def unix_to_matplotlib_dates(values: np.ndarray) -> np.ndarray:
    return mdates.date2num([datetime.fromtimestamp(float(value), tz=timezone.utc) for value in values])


def iso_to_unix(value: str) -> float:
    return parse_iso_timestamp(value).timestamp()


def finite_array(values) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=float)


def nearest_sample_index(samples: list[dict], target_time: datetime) -> int:
    if not samples:
        raise ValueError("The data-panel summary does not contain any samples.")
    sample_times = np.asarray([iso_to_unix(sample["target_time"]) for sample in samples], dtype=float)
    return int(np.argmin(np.abs(sample_times - target_time.timestamp())))


def validate_panel_ids(values=None) -> tuple[int, ...]:
    if values is None:
        return DEFAULT_PANEL_IDS

    parsed: list[int] = []
    for value in values:
        tokens = str(value).split(",")
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            try:
                parsed.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Invalid panel ID {token!r}; panel IDs must be integers.") from exc

    if not parsed:
        raise ValueError("At least one panel ID is required.")
    unknown = sorted(set(parsed) - set(PANEL_CATALOG))
    if unknown:
        choices = ", ".join(str(panel_id) for panel_id in PANEL_CATALOG)
        raise ValueError(f"Unknown panel ID(s) {unknown}; choose from: {choices}.")
    duplicates = sorted({panel_id for panel_id in parsed if parsed.count(panel_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate panel ID(s) are not allowed: {duplicates}.")
    ordered = [
        panel_id
        for panel_id in PANEL_CATALOG
        if panel_id in parsed and panel_id != COORDINATE_PANEL_ID
    ]
    if COORDINATE_PANEL_ID in parsed:
        ordered.append(COORDINATE_PANEL_ID)
    return tuple(ordered)


def panel_catalog_help() -> str:
    entries = [
        f"{panel_id}={metadata['name']}"
        for panel_id, metadata in PANEL_CATALOG.items()
    ]
    return "; ".join(entries)


def validate_energy_band(energy_band_eV: tuple[float, float]) -> tuple[float, float]:
    low, high = (float(energy_band_eV[0]), float(energy_band_eV[1]))
    if low <= 0.0 or high <= 0.0 or low >= high:
        raise ValueError("pad-energy-band must satisfy 0 < LOW_EV < HIGH_EV.")
    return low, high


def validate_energy_bands(values) -> tuple[tuple[float, float], ...]:
    if values is None:
        return DEFAULT_PAD_ENERGY_BANDS_EV
    if len(values) and not isinstance(values[0], (int, float)):
        bands = tuple(validate_energy_band(tuple(item)) for item in values)
    else:
        flat = [float(value) for value in values]
        if len(flat) % 2 != 0:
            raise ValueError("pad-energy-bands must be LOW HIGH pairs.")
        bands = tuple(validate_energy_band((flat[index], flat[index + 1])) for index in range(0, len(flat), 2))
    if not bands:
        raise ValueError("At least one PAD energy band is required.")
    return bands


def flatten_energy_bands(energy_bands_eV: tuple[tuple[float, float], ...]) -> list[float]:
    return [value for band in energy_bands_eV for value in band]


def energy_band_label(energy_band_eV: tuple[float, float]) -> str:
    low, high = validate_energy_band(energy_band_eV)
    return f"{low:g}-{high:g} eV"


def default_pad_energy_band(energy: np.ndarray, requested_band_eV: tuple[float, float]) -> tuple[float, float]:
    low, high = validate_energy_band(requested_band_eV)
    band_mask = (energy >= low) & (energy <= high)
    if np.any(band_mask):
        return low, high
    fallback = (100.0, 150.0)
    fallback_mask = (energy >= fallback[0]) & (energy <= fallback[1])
    if np.any(fallback_mask):
        return fallback
    raise ValueError(
        f"No SWE energy bins were found in {energy_band_label((low, high))} "
        f"or fallback {energy_band_label(fallback)}."
    )


def pick_first_cdf_variable(cdf: cdflib.CDF, names: list[str]) -> np.ndarray | None:
    info = cdf.cdf_info()
    available = set(info.zVariables) | set(info.rVariables)
    for name in names:
        if name in available:
            return np.asarray(cdf.varget(name))
    return None


def unix_seconds_from_cdf_epoch(epoch_values: np.ndarray) -> np.ndarray:
    datetimes = cdflib.cdfepoch.to_datetime(epoch_values)
    return datetimes.astype("datetime64[ns]").astype(np.int64) / 1e9


def unix_seconds_from_numeric_time(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size == 0:
        return flat
    if np.nanmedian(flat) > 1e12:
        return flat / 1000.0
    return flat


def normalize_cdf_times(cdf: cdflib.CDF, path: Path) -> np.ndarray:
    time_values = pick_first_cdf_variable(cdf, ["epoch", "time_unix", "time_met"])
    if time_values is None:
        raise KeyError(f"No usable time variable was found in {path.name}.")
    if time_values.dtype.kind in {"i", "u"} and np.nanmedian(time_values) > 1e12:
        return unix_seconds_from_cdf_epoch(time_values)
    return unix_seconds_from_numeric_time(time_values)


def average_positive_flux(values: np.ndarray, axis: int) -> np.ndarray:
    flux = np.asarray(values, dtype=float)
    valid = np.isfinite(flux) & (flux > 0.0)
    summed = np.sum(np.where(valid, flux, 0.0), axis=axis)
    counts = np.sum(valid, axis=axis)
    averaged = np.divide(summed, counts, out=np.zeros_like(summed, dtype=float), where=counts > 0)
    return np.nan_to_num(averaged, nan=0.0, posinf=0.0, neginf=0.0)


def static_axis_key(value: float) -> float:
    return round(float(value), 8)


def infer_static_sweep_indices(sweep_values: np.ndarray, n_sweeps: int) -> np.ndarray:
    indices = np.asarray(sweep_values, dtype=float).reshape(-1)
    result = np.full(indices.shape, -1, dtype=int)
    finite = np.isfinite(indices)
    if not np.any(finite):
        return result
    integer_values = indices[finite].astype(int)
    unique = np.unique(integer_values)
    # The STATIC SIS example uses swp_ind directly as an array index. Keep it
    # zero-based unless a file explicitly contains the one-based upper endpoint.
    if unique.size and unique.min() >= 1 and unique.max() == n_sweeps:
        integer_values = integer_values - 1
    result[finite] = integer_values
    return result


def common_static_axis(axis_sets: list[np.ndarray]) -> np.ndarray:
    values_by_key: dict[float, float] = {}
    for axis in axis_sets:
        for value in np.asarray(axis, dtype=float).reshape(-1):
            if np.isfinite(value) and value > 0.0:
                values_by_key.setdefault(static_axis_key(value), float(value))
    return np.asarray([values_by_key[key] for key in sorted(values_by_key)], dtype=float)


def regrid_static_rows(
    row_values: list[np.ndarray] | np.ndarray,
    row_axes: list[np.ndarray],
    target_axis: np.ndarray,
) -> np.ndarray:
    target_lookup = {static_axis_key(value): index for index, value in enumerate(np.asarray(target_axis, dtype=float))}
    output = np.full((len(row_axes), len(target_axis)), np.nan, dtype=float)
    for row_index, (values, axis) in enumerate(zip(row_values, row_axes)):
        sums = np.zeros(len(target_axis), dtype=float)
        counts = np.zeros(len(target_axis), dtype=int)
        for value, coordinate in zip(np.asarray(values, dtype=float).reshape(-1), np.asarray(axis, dtype=float).reshape(-1)):
            target_index = target_lookup.get(static_axis_key(coordinate))
            if target_index is None or not np.isfinite(value):
                continue
            sums[target_index] += float(value)
            counts[target_index] += 1
        output[row_index] = np.divide(
            sums,
            counts,
            out=np.full(len(target_axis), np.nan, dtype=float),
            where=counts > 0,
        )
    return output


def load_static_context(path: Path, start: datetime, end: datetime) -> dict | None:
    cdf = cdflib.CDF(str(path))
    times = normalize_cdf_times(cdf, path)
    time_mask = (times >= start.timestamp()) & (times <= end.timestamp())
    if not np.any(time_mask):
        return None

    eflux = pick_first_cdf_variable(cdf, ["eflux"])
    energy = pick_first_cdf_variable(cdf, ["energy"])
    mass_arr = pick_first_cdf_variable(cdf, ["mass_arr"])
    swp_ind = pick_first_cdf_variable(cdf, ["swp_ind"])
    if eflux is None or energy is None or mass_arr is None or swp_ind is None:
        return None

    eflux = np.asarray(eflux, dtype=float)
    energy = np.asarray(energy, dtype=float)
    mass_arr = np.asarray(mass_arr, dtype=float)
    if eflux.ndim != 3 or energy.ndim != 3 or mass_arr.ndim != 3:
        return None

    selected_eflux = eflux[time_mask]
    selected_times = times[time_mask]
    sweep_indices = infer_static_sweep_indices(swp_ind, energy.shape[2])[time_mask]
    valid_sweep = (sweep_indices >= 0) & (sweep_indices < energy.shape[2])
    if not np.any(valid_sweep):
        return None

    selected_eflux = selected_eflux[valid_sweep]
    selected_times = selected_times[valid_sweep]
    sweep_indices = sweep_indices[valid_sweep]

    energy_axes_by_sweep = np.asarray(energy[0, :, :], dtype=float)
    mass_axes_by_sweep = np.nanmean(mass_arr, axis=1)
    energy_rows = average_positive_flux(selected_eflux, axis=1)
    mass_rows = average_positive_flux(selected_eflux, axis=2)
    energy_row_axes = [energy_axes_by_sweep[:, sweep_index] for sweep_index in sweep_indices]
    mass_row_axes = [mass_axes_by_sweep[:, sweep_index] for sweep_index in sweep_indices]
    energy_axis_values = common_static_axis(energy_row_axes)
    mass_axis_values = common_static_axis(mass_row_axes)
    energy_spectrogram = regrid_static_rows(energy_rows, energy_row_axes, energy_axis_values)
    mass_spectrogram = regrid_static_rows(mass_rows, mass_row_axes, mass_axis_values)
    light_row_values: list[np.ndarray] = []
    light_row_axes: list[np.ndarray] = []
    heavy_row_values: list[np.ndarray] = []
    heavy_row_axes: list[np.ndarray] = []
    for row, axis in zip(mass_rows, mass_row_axes):
        light_mask = np.asarray(axis, dtype=float) < STATIC_MASS_SPLIT_AMU
        heavy_mask = np.asarray(axis, dtype=float) > STATIC_MASS_SPLIT_AMU
        light_row_values.append(row[light_mask])
        light_row_axes.append(axis[light_mask])
        heavy_row_values.append(row[heavy_mask])
        heavy_row_axes.append(axis[heavy_mask])
    light_mass_axis = common_static_axis(light_row_axes)
    heavy_mass_axis = common_static_axis(heavy_row_axes)
    light_mass_spectrogram = regrid_static_rows(light_row_values, light_row_axes, light_mass_axis)
    heavy_mass_spectrogram = regrid_static_rows(heavy_row_values, heavy_row_axes, heavy_mass_axis)

    return {
        "times_unix": selected_times.tolist(),
        "energy_eV": np.asarray(energy_axis_values, dtype=float).tolist(),
        "energy_eflux": energy_spectrogram.tolist(),
        "energy_eV_by_time": [np.asarray(axis, dtype=float).tolist() for axis in energy_row_axes],
        "energy_eflux_by_time": np.asarray(energy_rows, dtype=float).tolist(),
        "mass_amu": np.asarray(mass_axis_values, dtype=float).tolist(),
        "mass_eflux": mass_spectrogram.tolist(),
        "mass_amu_by_time": [np.asarray(axis, dtype=float).tolist() for axis in mass_row_axes],
        "mass_eflux_by_time": np.asarray(mass_rows, dtype=float).tolist(),
        "mass_amu_0_1p5": np.asarray(light_mass_axis, dtype=float).tolist(),
        "mass_amu_gt_1p5": np.asarray(heavy_mass_axis, dtype=float).tolist(),
        "mass_eflux_0_1p5": light_mass_spectrogram.tolist(),
        "mass_eflux_gt_1p5": heavy_mass_spectrogram.tolist(),
        "mass_amu_0_1p5_by_time": [np.asarray(axis, dtype=float).tolist() for axis in light_row_axes],
        "mass_amu_gt_1p5_by_time": [np.asarray(axis, dtype=float).tolist() for axis in heavy_row_axes],
        "mass_eflux_0_1p5_by_time": [np.asarray(row, dtype=float).tolist() for row in light_row_values],
        "mass_eflux_gt_1p5_by_time": [np.asarray(row, dtype=float).tolist() for row in heavy_row_values],
        "mass_split_amu": STATIC_MASS_SPLIT_AMU,
        "sweep_indices_0_based": sweep_indices.tolist(),
        "source_file": str(path),
    }


def build_swe_context_for_band(
    pad_data: dict,
    start: datetime,
    end: datetime,
    pad_energy_bands_eV: tuple[tuple[float, float], ...] = DEFAULT_PAD_ENERGY_BANDS_EV,
) -> dict | None:
    times = np.asarray(pad_data["times"], dtype=float)
    time_mask = (times >= start.timestamp()) & (times <= end.timestamp())
    if not np.any(time_mask):
        return None

    indices = np.where(time_mask)[0]
    flux = np.asarray(pad_data["flux"], dtype=float)[indices]
    energy = np.asarray(pad_data["energy"], dtype=float)
    omni_spectrum = np.nanmean(flux, axis=1)
    pad_bands = []
    for requested_band in validate_energy_bands(pad_energy_bands_eV):
        actual_band_eV = default_pad_energy_band(energy, requested_band)
        band_mask_energy = (energy >= actual_band_eV[0]) & (energy <= actual_band_eV[1])
        band_flux = flux[:, :, band_mask_energy]
        valid_counts = np.sum(np.isfinite(band_flux), axis=2)
        band_sum = np.nansum(band_flux, axis=2)
        pad_band = np.divide(
            band_sum,
            valid_counts,
            out=np.full_like(band_sum, np.nan, dtype=float),
            where=valid_counts > 0,
        )
        pitch_bins = derive_pitch_bins(pad_data, indices, band_mask_energy)
        pad_bands.append(
            {
                "energy_band_eV": list(actual_band_eV),
                "pitch_deg": np.asarray(pitch_bins, dtype=float).tolist(),
                "eflux": pad_band.tolist(),
            }
        )

    context = {
        "times_unix": times[indices].tolist(),
        "energy_eV": energy.tolist(),
        "omni_eflux": omni_spectrum.tolist(),
        "pad_bands": pad_bands,
    }
    if pad_bands:
        context["pitch_deg"] = pad_bands[0]["pitch_deg"]
        context["pad_eflux"] = pad_bands[0]["eflux"]
        context["pad_energy_band_eV"] = pad_bands[0]["energy_band_eV"]
    return context


def concat_timeseries(parts: list[dict], keys: list[str]) -> dict | None:
    if not parts:
        return None
    merged: dict[str, list] = {key: [] for key in keys}
    static_keys = [key for key in parts[0].keys() if key not in merged]
    for part in parts:
        for key in keys:
            merged[key].extend(part.get(key, []))
    for key in static_keys:
        merged[key] = parts[0][key]
    return merged


def concat_swe_context(parts: list[dict]) -> dict | None:
    if not parts:
        return None
    merged = concat_timeseries(parts, ["times_unix", "omni_eflux"])
    if merged is None:
        return None

    band_count = max((len(part.get("pad_bands") or []) for part in parts), default=0)
    pad_bands = []
    for band_index in range(band_count):
        band_eflux: list = []
        energy_band_eV = None
        pitch_deg = None
        for part in parts:
            bands = part.get("pad_bands") or []
            if band_index >= len(bands):
                continue
            band = bands[band_index]
            energy_band_eV = energy_band_eV or band.get("energy_band_eV")
            pitch_deg = pitch_deg or band.get("pitch_deg")
            band_eflux.extend(band.get("eflux", []))
        pad_bands.append(
            {
                "energy_band_eV": energy_band_eV or [],
                "pitch_deg": pitch_deg or [],
                "eflux": band_eflux,
            }
        )

    merged["pad_bands"] = pad_bands
    if pad_bands:
        merged["pad_eflux"] = pad_bands[0]["eflux"]
        merged["pitch_deg"] = pad_bands[0]["pitch_deg"]
        merged["pad_energy_band_eV"] = pad_bands[0]["energy_band_eV"]
    return merged


def concat_static_context(parts: list[dict]) -> dict | None:
    if not parts:
        return None

    energy_axis = common_static_axis([np.asarray(part.get("energy_eV", []), dtype=float) for part in parts])
    mass_axis = common_static_axis([np.asarray(part.get("mass_amu", []), dtype=float) for part in parts])
    light_mass_axis = common_static_axis([np.asarray(part.get("mass_amu_0_1p5", []), dtype=float) for part in parts])
    heavy_mass_axis = common_static_axis([np.asarray(part.get("mass_amu_gt_1p5", []), dtype=float) for part in parts])
    times: list[float] = []
    energy_parts: list[np.ndarray] = []
    mass_parts: list[np.ndarray] = []
    light_parts: list[np.ndarray] = []
    heavy_parts: list[np.ndarray] = []

    for part in parts:
        part_times = np.asarray(part.get("times_unix", []), dtype=float)
        times.extend(part_times.tolist())
        part_energy_axis = np.asarray(part.get("energy_eV", []), dtype=float)
        part_mass_axis = np.asarray(part.get("mass_amu", []), dtype=float)
        part_light_axis = np.asarray(part.get("mass_amu_0_1p5", []), dtype=float)
        part_heavy_axis = np.asarray(part.get("mass_amu_gt_1p5", []), dtype=float)
        part_energy = np.asarray(part.get("energy_eflux", []), dtype=float)
        part_mass = np.asarray(part.get("mass_eflux", []), dtype=float)
        part_light = np.asarray(part.get("mass_eflux_0_1p5", []), dtype=float)
        part_heavy = np.asarray(part.get("mass_eflux_gt_1p5", []), dtype=float)
        energy_parts.append(regrid_static_rows(part_energy, [part_energy_axis] * len(part_energy), energy_axis))
        mass_parts.append(regrid_static_rows(part_mass, [part_mass_axis] * len(part_mass), mass_axis))
        light_parts.append(regrid_static_rows(part_light, [part_light_axis] * len(part_light), light_mass_axis))
        heavy_parts.append(regrid_static_rows(part_heavy, [part_heavy_axis] * len(part_heavy), heavy_mass_axis))

    return {
        "times_unix": times,
        "energy_eV": energy_axis.tolist(),
        "mass_amu": mass_axis.tolist(),
        "mass_amu_0_1p5": light_mass_axis.tolist(),
        "mass_amu_gt_1p5": heavy_mass_axis.tolist(),
        "energy_eflux": np.vstack(energy_parts).tolist() if energy_parts else [],
        "mass_eflux": np.vstack(mass_parts).tolist() if mass_parts else [],
        "mass_eflux_0_1p5": np.vstack(light_parts).tolist() if light_parts else [],
        "mass_eflux_gt_1p5": np.vstack(heavy_parts).tolist() if heavy_parts else [],
        "energy_eV_by_time": [axis for part in parts for axis in part.get("energy_eV_by_time", [])],
        "energy_eflux_by_time": [row for part in parts for row in part.get("energy_eflux_by_time", [])],
        "mass_amu_by_time": [axis for part in parts for axis in part.get("mass_amu_by_time", [])],
        "mass_eflux_by_time": [row for part in parts for row in part.get("mass_eflux_by_time", [])],
        "mass_amu_0_1p5_by_time": [axis for part in parts for axis in part.get("mass_amu_0_1p5_by_time", [])],
        "mass_amu_gt_1p5_by_time": [axis for part in parts for axis in part.get("mass_amu_gt_1p5_by_time", [])],
        "mass_eflux_0_1p5_by_time": [row for part in parts for row in part.get("mass_eflux_0_1p5_by_time", [])],
        "mass_eflux_gt_1p5_by_time": [row for part in parts for row in part.get("mass_eflux_gt_1p5_by_time", [])],
        "mass_split_amu": STATIC_MASS_SPLIT_AMU,
        "source_file": parts[0].get("source_file"),
    }


def sample_altitude_entries(
    mag_data_ss: dict,
    start: datetime,
    end: datetime,
    step_seconds: int,
) -> list[dict]:
    times = np.asarray(mag_data_ss["times"], dtype=float)
    samples: list[dict] = []
    for index in select_time_indices(times, start, end, step_seconds):
        sample_time = datetime.fromtimestamp(float(times[index]), tz=timezone.utc)
        position_km = np.asarray(mag_data_ss["data"][index, mag_data_ss["pos_indices"]], dtype=float)
        altitude_km = float(np.linalg.norm(position_km) - MARS_RADIUS_KM)
        samples.append(
            {
                "target_time": sample_time.isoformat(timespec="seconds"),
                "topology": "not_computed",
                "altitude_km": altitude_km,
                "altitude_rm": altitude_km / MARS_RADIUS_KM,
                "position_km": position_km.tolist(),
                "position_rm": (position_km / MARS_RADIUS_KM).tolist(),
            }
        )
    return samples


def pc_position_to_lon_lat(position_pc_km: np.ndarray) -> tuple[float, float, float]:
    x, y, z = np.asarray(position_pc_km, dtype=float)
    radius = float(np.linalg.norm([x, y, z]))
    lon = float(np.degrees(np.arctan2(y, x)) % 360.0)
    lat = float(np.degrees(np.arcsin(np.clip(z / max(radius, 1e-9), -1.0, 1.0))))
    return lon, lat, radius


def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    source = np.asarray(source_times, dtype=float)
    targets = np.asarray(target_times, dtype=float)
    if source.size == 0 or targets.size == 0:
        return np.asarray([], dtype=int)
    insertion = np.searchsorted(source, targets)
    left = np.clip(insertion - 1, 0, source.size - 1)
    right = np.clip(insertion, 0, source.size - 1)
    choose_right = np.abs(source[right] - targets) < np.abs(source[left] - targets)
    return np.where(choose_right, right, left)


def build_position_context(mag_data_ss: dict, mag_data_pc: dict, start: datetime, end: datetime) -> dict | None:
    ss_times = np.asarray(mag_data_ss["times"], dtype=float)
    time_mask = (ss_times >= start.timestamp()) & (ss_times <= end.timestamp())
    if not np.any(time_mask):
        return None

    selected_times = ss_times[time_mask]
    ss_positions_km = np.asarray(mag_data_ss["data"], dtype=float)[time_mask][:, mag_data_ss["pos_indices"]]
    ss_positions_rm = ss_positions_km / MARS_RADIUS_KM
    altitude_km = np.linalg.norm(ss_positions_km, axis=1) - MARS_RADIUS_KM

    pc_times = np.asarray(mag_data_pc["times"], dtype=float)
    pc_indices = nearest_indices(pc_times, selected_times)
    if pc_indices.size != selected_times.size:
        return {
            "times_unix": selected_times.tolist(),
            "x_mso_rm": ss_positions_rm[:, 0].tolist(),
            "y_mso_rm": ss_positions_rm[:, 1].tolist(),
            "z_mso_rm": ss_positions_rm[:, 2].tolist(),
            "longitude_deg": [float("nan")] * selected_times.size,
            "latitude_deg": [float("nan")] * selected_times.size,
            "altitude_km": altitude_km.tolist(),
        }
    pc_positions_km = np.asarray(mag_data_pc["data"], dtype=float)[pc_indices][:, mag_data_pc["pos_indices"]]
    lon_lat = np.asarray([pc_position_to_lon_lat(position)[:2] for position in pc_positions_km], dtype=float)

    return {
        "times_unix": selected_times.tolist(),
        "x_mso_rm": ss_positions_rm[:, 0].tolist(),
        "y_mso_rm": ss_positions_rm[:, 1].tolist(),
        "z_mso_rm": ss_positions_rm[:, 2].tolist(),
        "longitude_deg": lon_lat[:, 0].tolist(),
        "latitude_deg": lon_lat[:, 1].tolist(),
        "altitude_km": altitude_km.tolist(),
    }


def build_data_panel_summary_from_data(
    target_time: datetime,
    window_minutes: float,
    step_seconds: int,
    data_root: Path = DEFAULT_DATA_ROOT,
    auto_download_missing_data: bool = False,
    pad_energy_bands_eV: tuple[tuple[float, float], ...] = DEFAULT_PAD_ENERGY_BANDS_EV,
) -> dict:
    if window_minutes <= 0:
        raise ValueError("window-minutes must be positive.")
    if step_seconds <= 0:
        raise ValueError("step-seconds must be positive.")
    pad_energy_bands_eV = validate_energy_bands(pad_energy_bands_eV)

    half_window = timedelta(minutes=window_minutes / 2.0)
    start = target_time - half_window
    end = target_time + half_window
    resolved_files = resolve_daily_files(
        start=start,
        end=end,
        data_root=data_root,
        pad_file=None,
        mag_file=None,
        auto_download_missing_data=auto_download_missing_data,
    )

    static_parts: list[dict] = []
    swe_parts: list[dict] = []
    mag_parts: list[dict] = []
    samples: list[dict] = []
    input_files: dict[str, dict[str, str]] = {}

    for day in iter_days(start, end):
        files = resolved_files[day]
        input_files[day.isoformat()] = {key: str(path) for key, path in files.items()}
        day_start = max(start, datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc))
        day_end = min(end, datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc))

        pad_data = load_pad_data(files["pad"])
        swe_context = build_swe_context_for_band(pad_data, day_start, day_end, pad_energy_bands_eV)
        if swe_context:
            swe_parts.append(swe_context)

        mag_data_ss = load_mag_day(files["mag_ss"])
        mag_context = build_mag_context(mag_data_ss, day_start, day_end)
        if mag_context:
            mag_data_pc = load_mag_day(files["mag_pc"])
            position_context = build_position_context(mag_data_ss, mag_data_pc, day_start, day_end)
            if position_context:
                mag_context.update(
                    {
                        key: value
                        for key, value in position_context.items()
                        if key != "times_unix"
                    }
                )
            mag_parts.append(mag_context)
        samples.extend(sample_altitude_entries(mag_data_ss, day_start, day_end, step_seconds))

        static_context = load_static_context(files["sta_c6"], day_start, day_end)
        if static_context:
            static_parts.append(static_context)

    if not samples:
        raise ValueError("No MAG samples were found in the requested event window.")

    return {
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": end.isoformat(timespec="seconds"),
        "step_seconds": step_seconds,
        "topology_computed": False,
        "source": "local_data",
        "input_files": input_files,
        "context_overview": {
            "window_seconds": window_minutes * 60.0,
            "static": concat_static_context(static_parts),
            "mag": concat_timeseries(
                mag_parts,
                [
                    "times_unix",
                    "bx_nT",
                    "by_nT",
                    "bz_nT",
                    "bmag_nT",
                    "x_mso_rm",
                    "y_mso_rm",
                    "z_mso_rm",
                    "latitude_deg",
                    "longitude_deg",
                    "altitude_km",
                ],
            ),
            "swe": concat_swe_context(swe_parts),
        },
        "samples": samples,
    }


def build_region_id_context(
    start: datetime,
    end: datetime,
    data_root: Path,
    cadence_seconds: float = 10.0,
) -> dict:
    # Imported lazily so existing panel combinations do not initialize the
    # region classifier or its SWEA/STATIC feature extractors.
    from region_id.classify_region_id import (
        RegionClassifierConfig,
        classify_interval,
    )

    rows, metadata = classify_interval(
        start=start,
        end=end,
        data_root=data_root,
        config=RegionClassifierConfig(cadence_seconds=float(cadence_seconds)),
    )
    return {
        "times_unix": [float(row["time_unix"]) for row in rows],
        "region_id": [int(row["region_id"]) for row in rows],
        "region_name": [str(row["region_name"]) for row in rows],
        "confidence": [float(row["confidence"]) for row in rows],
        "reason": [str(row["reason"]) for row in rows],
        "metadata": metadata,
    }


def window_indices(times_unix, center_unix: float, window_seconds: float) -> np.ndarray:
    times = finite_array(times_unix)
    return np.where((times >= center_unix - window_seconds / 2.0) & (times <= center_unix + window_seconds / 2.0))[0]


def axis_edges(values: np.ndarray, log_scale: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if values.size == 1:
        delta = values[0] * 0.05 if log_scale and values[0] > 0 else 0.5
        return np.array([values[0] - delta, values[0] + delta], dtype=float)
    if log_scale:
        safe = np.clip(values, 1e-12, None)
        log_values = np.log10(safe)
        mids = (log_values[:-1] + log_values[1:]) / 2.0
        first = log_values[0] - (mids[0] - log_values[0])
        last = log_values[-1] + (log_values[-1] - mids[-1])
        return 10.0 ** np.concatenate([[first], mids, [last]])
    mids = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def prepare_heatmap(matrix, y_values, log_y: bool = False) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(matrix, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if z.ndim != 2 or y.size == 0:
        return np.empty((0, 0)), y
    if z.shape[1] != y.size and z.shape[0] == y.size:
        z = z.T
    order = np.argsort(y)
    if log_y:
        order = order[y[order] > 0]
    y_sorted = y[order]
    return z[:, order].T, y_sorted


def positive_log_norm(matrix, lower_percentile: float = 2.0, upper_percentile: float = 98.0) -> LogNorm | None:
    values = np.asarray(matrix, dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return None
    vmin = float(np.nanpercentile(positive, lower_percentile))
    vmax = float(np.nanpercentile(positive, upper_percentile))
    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = float(np.nanmin(positive))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(positive))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def resolve_local_source(path_value: str | None, data_root: Path = DEFAULT_DATA_ROOT) -> Path | None:
    if not path_value:
        return None
    source = Path(path_value).expanduser()
    if source.exists():
        return source.resolve()
    matches = sorted(data_root.rglob(source.name))
    return matches[0].resolve() if matches else None


def reload_static_context_for_window(static: dict, center_unix: float, window_seconds: float) -> dict:
    source = resolve_local_source(static.get("source_file"))
    if source is None:
        return static
    start = datetime.fromtimestamp(center_unix - window_seconds / 2.0, tz=timezone.utc)
    end = datetime.fromtimestamp(center_unix + window_seconds / 2.0, tz=timezone.utc)
    reloaded = load_static_context(source, start, end)
    return reloaded or static


def sample_altitude_km(sample: dict) -> float:
    if sample.get("altitude_km") is not None:
        return float(sample["altitude_km"])
    if sample.get("position_km"):
        position = np.asarray(sample["position_km"], dtype=float)
        return float(np.linalg.norm(position) - MARS_RADIUS_KM)
    if sample.get("position_rm"):
        position = np.asarray(sample["position_rm"], dtype=float)
        return float((np.linalg.norm(position) - 1.0) * MARS_RADIUS_KM)
    return float("nan")


def resolve_pad_panel_data(
    swe: dict,
    band_index: int = 0,
    fallback_bands_eV: tuple[tuple[float, float], ...] = DEFAULT_PAD_ENERGY_BANDS_EV,
) -> tuple[np.ndarray, tuple[float, float]]:
    bands = swe.get("pad_bands") or []
    if 0 <= band_index < len(bands):
        band = bands[band_index]
        return (
            np.asarray(band.get("eflux", []), dtype=float),
            validate_energy_band(tuple(band.get("energy_band_eV"))),
        )
    if band_index == 0 and "pad_eflux" in swe:
        band = swe.get("pad_energy_band_eV") or DEFAULT_PAD_ENERGY_BAND_EV
        return np.asarray(swe.get("pad_eflux", []), dtype=float), validate_energy_band(tuple(band))
    fallback_bands = validate_energy_bands(fallback_bands_eV)
    fallback_index = min(max(band_index, 0), len(fallback_bands) - 1)
    return np.asarray([], dtype=float), fallback_bands[fallback_index]


def plot_heatmap(
    ax,
    matrix,
    times_unix,
    y_values,
    title: str,
    ylabel: str,
    log_y: bool = False,
    norm=None,
    cmap: str = FLUX_CMAP,
):
    times = finite_array(times_unix)
    if len(times) == 0 or len(y_values) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=18)
        ax.set_title(title, fontsize=18)
        return None
    z, y_sorted = prepare_heatmap(matrix, y_values, log_y=log_y)
    if z.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=18)
        ax.set_title(title, fontsize=18)
        return None

    time_edges = axis_edges(unix_to_matplotlib_dates(times), log_scale=False)
    y_edges = axis_edges(y_sorted, log_scale=log_y)
    mesh = ax.pcolormesh(time_edges, y_edges, z, shading="auto", cmap=cmap, norm=norm)
    ax.set_title(title, loc="left", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if log_y:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    return mesh


def plot_variable_heatmap(
    ax,
    matrix_by_time,
    times_unix,
    y_values_by_time,
    title: str,
    ylabel: str,
    log_y: bool = False,
    norm=None,
    cmap: str = FLUX_CMAP,
):
    times = finite_array(times_unix)
    values = [] if matrix_by_time is None else list(matrix_by_time)
    y_rows = [] if y_values_by_time is None else list(y_values_by_time)
    if len(times) == 0 or not values or not y_rows:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=18)
        ax.set_title(title, fontsize=18)
        return None

    count = min(len(times), len(values), len(y_rows))
    times = times[:count]
    time_edges = axis_edges(unix_to_matplotlib_dates(times), log_scale=False)
    mesh = None
    for index in range(count):
        row = np.asarray(values[index], dtype=float).reshape(-1)
        y = np.asarray(y_rows[index], dtype=float).reshape(-1)
        if row.size == 0 or y.size == 0:
            continue
        usable = np.isfinite(row) & np.isfinite(y)
        if log_y:
            usable &= y > 0.0
        if not np.any(usable):
            continue
        row = row[usable]
        y = y[usable]
        order = np.argsort(y)
        row = row[order]
        y = y[order]
        if norm is not None:
            row = np.where(row > 0.0, row, np.nan)
        y_edges = axis_edges(y, log_scale=log_y)
        mesh = ax.pcolormesh(
            np.asarray([time_edges[index], time_edges[index + 1]], dtype=float),
            y_edges,
            row[:, None],
            shading="auto",
            cmap=cmap,
            norm=norm,
        )

    if mesh is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=18)
        ax.set_title(title, fontsize=18)
        return None

    ax.set_title(title, loc="left", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if log_y:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    return mesh


def plot_line_panel(ax, times_unix, traces: list[tuple[str, str, np.ndarray]], title: str, ylabel: str, y_range=None):
    times = finite_array(times_unix)
    if len(times) == 0 or not traces:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=18)
        ax.set_title(title, loc="left", fontsize=18)
        return
    x = [datetime.fromtimestamp(float(value), tz=timezone.utc) for value in times]
    for label, color, values in traces:
        ax.plot(x, values, color=color, linewidth=1.2, label=label)
    ax.set_title(title, loc="left", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if y_range:
        ax.set_ylim(*y_range)
    ax.legend(loc="upper right", fontsize=18, frameon=False, ncol=min(3, len(traces)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))


def plot_region_id_panel(ax, times_unix, region_ids) -> None:
    times = finite_array(times_unix)
    ids = np.asarray(region_ids, dtype=float)
    count = min(times.size, ids.size)
    if count == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=18)
        ax.set_ylabel("region_id", fontsize=18)
        return

    times = times[:count]
    ids = ids[:count]
    valid = np.isfinite(times) & np.isfinite(ids)
    if not np.any(valid):
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=18)
        ax.set_ylabel("region_id", fontsize=18)
        return

    times = times[valid]
    ids = ids[valid].astype(int)
    x = [datetime.fromtimestamp(float(value), tz=timezone.utc) for value in times]
    ax.step(x, ids, where="mid", color="#252A31", linewidth=1.0)
    for region_id, color in REGION_ID_COLORS.items():
        selected = ids == region_id
        if np.any(selected):
            ax.scatter(
                np.asarray(x, dtype=object)[selected],
                ids[selected],
                s=12,
                color=color,
                edgecolors="none",
                zorder=3,
            )
    ax.set_ylabel("region_id", fontsize=18)
    ax.set_yticks(list(REGION_ID_COLORS))
    ax.set_ylim(-0.45, 4.45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))


def mark_target_time(ax, target_unix: float) -> None:
    ax.axvline(
        mdates.date2num(datetime.fromtimestamp(float(target_unix), tz=timezone.utc)),
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
        zorder=10,
    )


def mark_panel_id(ax, panel_id: int) -> None:
    is_coordinate_panel = panel_id == COORDINATE_PANEL_ID
    ax.text(
        0.994 if is_coordinate_panel else 0.006,
        0.96,
        f"[{panel_id}]",
        transform=ax.transAxes,
        ha="right" if is_coordinate_panel else "left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#20242a",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        zorder=20,
    )


def nearest_value_at_time(times: np.ndarray, values: np.ndarray, target_unix: float) -> float:
    if times.size == 0 or values.size == 0:
        return float("nan")
    index = int(np.argmin(np.abs(times - target_unix)))
    if index >= values.size:
        return float("nan")
    return float(values[index])


def format_annotation_value(value: float, precision: int = 2, signed: bool = True) -> str:
    if not np.isfinite(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign}.{precision}f}"


def draw_bottom_coordinate_axis(
    ax,
    mag: dict,
    start_unix: float,
    end_unix: float,
    sample_count: int = 6,
) -> None:
    ax.set_xlim(
        mdates.date2num(datetime.fromtimestamp(start_unix, tz=timezone.utc)),
        mdates.date2num(datetime.fromtimestamp(end_unix, tz=timezone.utc)),
    )
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    row_positions = [
        ("UTC", 0.93),
        ("X_MSO (R_M)", 0.79),
        ("Y_MSO (R_M)", 0.65),
        ("Z_MSO (R_M)", 0.51),
        ("Latitude", 0.37),
        ("Longitude", 0.23),
        ("Altitude (km)", 0.09),
    ]
    for label, y in row_positions:
        ax.text(-0.014, y, label, transform=ax.transAxes, ha="right", va="center", fontsize=18, weight="bold")

    annotation_times = np.linspace(start_unix, end_unix, sample_count + 2)[1:-1]
    mag_times = finite_array(mag.get("times_unix"))
    x_values = finite_array(mag.get("x_mso_rm"))
    y_values = finite_array(mag.get("y_mso_rm"))
    z_values = finite_array(mag.get("z_mso_rm"))
    lat_values = finite_array(mag.get("latitude_deg"))
    lon_values = finite_array(mag.get("longitude_deg"))
    altitude_values = finite_array(mag.get("altitude_km"))
    if altitude_values.size == 0 and x_values.size and y_values.size and z_values.size:
        count = min(x_values.size, y_values.size, z_values.size)
        altitude_values = (np.linalg.norm(np.column_stack([x_values[:count], y_values[:count], z_values[:count]]), axis=1) - 1.0) * MARS_RADIUS_KM

    for value in annotation_times:
        x = mdates.date2num(datetime.fromtimestamp(float(value), tz=timezone.utc))
        rows = [
            datetime.fromtimestamp(float(value), tz=timezone.utc).strftime("%H:%M:%S"),
            format_annotation_value(nearest_value_at_time(mag_times, x_values, value), precision=2, signed=True),
            format_annotation_value(nearest_value_at_time(mag_times, y_values, value), precision=2, signed=True),
            format_annotation_value(nearest_value_at_time(mag_times, z_values, value), precision=2, signed=True),
            format_annotation_value(nearest_value_at_time(mag_times, lat_values, value), precision=1, signed=True),
            format_annotation_value(nearest_value_at_time(mag_times, lon_values, value), precision=1, signed=False),
            format_annotation_value(nearest_value_at_time(mag_times, altitude_values, value), precision=0, signed=False),
        ]
        for (_, y), text in zip(row_positions, rows):
            ax.text(x, y, text, ha="center", va="center", fontsize=18)


def add_panel_colorbar(fig, ax, mesh, label="eflux", x=0.875, width=0.012, fontsize=18):
    """
    Add a colorbar in a fixed right-side position without shrinking the main axes.
    This keeps all data panels horizontally aligned.
    """
    if mesh is None:
        return None

    pos = ax.get_position()
    cax = fig.add_axes([x, pos.y0, width, pos.height])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    return cbar


def plot_data_panels(
    summary: dict,
    target_time: datetime,
    output_path: Path,
    window_minutes: float = 20.0,
    pad_energy_bands_eV: tuple[tuple[float, float], ...] = DEFAULT_PAD_ENERGY_BANDS_EV,
    figure_size: tuple[float, float] | None = None,
    center_on_target_time: bool = False,
    panel_ids: tuple[int, ...] = DEFAULT_PANEL_IDS,
) -> dict:
    panel_ids = validate_panel_ids(panel_ids)
    pad_energy_bands_eV = validate_energy_bands(pad_energy_bands_eV)
    required_pad_bands = max(
        (PAD_PANEL_BAND_INDEX[panel_id] + 1 for panel_id in panel_ids if panel_id in PAD_PANEL_BAND_INDEX),
        default=0,
    )
    if len(pad_energy_bands_eV) < required_pad_bands:
        raise ValueError(
            f"Selected panels require at least {required_pad_bands} PAD energy band(s), "
            f"but {len(pad_energy_bands_eV)} were provided."
        )
    samples = summary.get("samples", [])
    selected_index = nearest_sample_index(samples, target_time)
    selected = samples[selected_index]

    center_unix = target_time.timestamp() if center_on_target_time else iso_to_unix(selected["target_time"])
    target_unix = target_time.timestamp()
    window_seconds = window_minutes * 60.0

    window_start = mdates.date2num(
        datetime.fromtimestamp(center_unix - window_seconds / 2.0, tz=timezone.utc)
    )
    window_end = mdates.date2num(
        datetime.fromtimestamp(center_unix + window_seconds / 2.0, tz=timezone.utc)
    )

    overview = summary.get("context_overview", {})

    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_height = sum(float(PANEL_CATALOG[panel_id]["height_ratio"]) for panel_id in panel_ids)
    default_height = sum(float(PANEL_CATALOG[panel_id]["height_ratio"]) for panel_id in DEFAULT_PANEL_IDS)
    if figure_size is None:
        figure_size = (16.0, max(3.5, 20.0 * selected_height / default_height))
    fig = plt.figure(figsize=figure_size)
    grid = fig.add_gridspec(
        len(panel_ids),
        1,
        height_ratios=[
            float(PANEL_CATALOG[panel_id]["height_ratio"])
            for panel_id in panel_ids
        ],
        hspace=0,
    )

    axes_by_id = {}
    first_axis = None
    for row_index, panel_id in enumerate(panel_ids):
        axis = fig.add_subplot(
            grid[row_index, 0],
            sharex=first_axis if first_axis is not None else None,
        )
        axes_by_id[panel_id] = axis
        if first_axis is None:
            first_axis = axis

    # Fixed layout. Do not use constrained_layout=True or tight_layout().
    # right is kept smaller to reserve a fixed colorbar column.
    fig.subplots_adjust(
        left=0.16,
        right=0.86,
        top=0.985,
        bottom=0.04,
        hspace=0,
    )

    label_fs = 18
    tick_fs = 18
    legend_fs = 18
    cbar_fs = 18

    static = reload_static_context_for_window(
        overview.get("static") or {},
        center_unix,
        window_seconds,
    )
    static_indices = window_indices(static.get("times_unix"), center_unix, window_seconds)

    static_times = (
        np.asarray(static.get("times_unix", []), dtype=float)[static_indices]
        if len(static_indices)
        else []
    )
    static_energy_all = np.asarray(static.get("energy_eflux", []), dtype=float)
    static_energy_matrix = static_energy_all[static_indices] if len(static_indices) and static_energy_all.size else []
    energy_values_by_time_all = static.get("energy_eflux_by_time") or []
    energy_axis_by_time_all = static.get("energy_eV_by_time") or []
    static_energy_by_time = [energy_values_by_time_all[i] for i in static_indices] if energy_values_by_time_all else []
    static_energy_axis_by_time = [energy_axis_by_time_all[i] for i in static_indices] if energy_axis_by_time_all else []

    if 1 in axes_by_id:
        if static_energy_by_time and static_energy_axis_by_time:
            static_energy_norm_values = np.concatenate(
                [np.asarray(row, dtype=float).reshape(-1) for row in static_energy_by_time]
            )
            mesh = plot_variable_heatmap(
                axes_by_id[1],
                static_energy_by_time,
                static_times,
                static_energy_axis_by_time,
                "",
                "STATIC Energy\n(eV)",
                log_y=True,
                norm=positive_log_norm(static_energy_norm_values),
            )
        else:
            mesh = plot_heatmap(
                axes_by_id[1],
                static_energy_matrix,
                static_times,
                static.get("energy_eV", []),
                "",
                "STATIC Energy\n(eV)",
                log_y=True,
                norm=positive_log_norm(static_energy_matrix),
            )
        add_panel_colorbar(fig, axes_by_id[1], mesh, label="eflux", fontsize=cbar_fs)

    mass_light_all = np.asarray(static.get("mass_eflux_0_1p5", []), dtype=float)
    mass_heavy_all = np.asarray(static.get("mass_eflux_gt_1p5", []), dtype=float)
    if (not mass_light_all.size or not mass_heavy_all.size) and static.get("mass_eflux") is not None:
        mass_all = np.asarray(static.get("mass_eflux", []), dtype=float)
        mass_axis = np.asarray(static.get("mass_amu", []), dtype=float)
        light_mask = mass_axis < STATIC_MASS_SPLIT_AMU
        heavy_mask = mass_axis > STATIC_MASS_SPLIT_AMU
        if mass_all.ndim == 2 and mass_axis.size == mass_all.shape[1]:
            mass_light_all = mass_all[:, light_mask]
            mass_heavy_all = mass_all[:, heavy_mask]
            static["mass_amu_0_1p5"] = mass_axis[light_mask].tolist()
            static["mass_amu_gt_1p5"] = mass_axis[heavy_mask].tolist()

    mass_light_matrix = mass_light_all[static_indices] if len(static_indices) and mass_light_all.size else []
    mass_heavy_matrix = mass_heavy_all[static_indices] if len(static_indices) and mass_heavy_all.size else []
    light_values_by_time_all = static.get("mass_eflux_0_1p5_by_time") or []
    light_axis_by_time_all = static.get("mass_amu_0_1p5_by_time") or []
    heavy_values_by_time_all = static.get("mass_eflux_gt_1p5_by_time") or []
    heavy_axis_by_time_all = static.get("mass_amu_gt_1p5_by_time") or []
    mass_light_by_time = [light_values_by_time_all[i] for i in static_indices] if light_values_by_time_all else []
    mass_light_axis_by_time = [light_axis_by_time_all[i] for i in static_indices] if light_axis_by_time_all else []
    mass_heavy_by_time = [heavy_values_by_time_all[i] for i in static_indices] if heavy_values_by_time_all else []
    mass_heavy_axis_by_time = [heavy_axis_by_time_all[i] for i in static_indices] if heavy_axis_by_time_all else []
    mass_norm = positive_log_norm(
        np.concatenate(
            [
                np.asarray(mass_light_matrix, dtype=float).reshape(-1),
                np.asarray(mass_heavy_matrix, dtype=float).reshape(-1),
            ]
        )
    )

    if 2 in axes_by_id:
        if mass_light_by_time and mass_light_axis_by_time:
            mesh = plot_variable_heatmap(
                axes_by_id[2],
                mass_light_by_time,
                static_times,
                mass_light_axis_by_time,
                "",
                f"STATIC Mass \n0-{STATIC_MASS_SPLIT_AMU:g} amu",
                norm=mass_norm,
            )
        else:
            mesh = plot_heatmap(
                axes_by_id[2],
                mass_light_matrix,
                static_times,
                static.get("mass_amu_0_1p5", []),
                "",
                f"STATIC Mass \n0-{STATIC_MASS_SPLIT_AMU:g} amu",
                norm=mass_norm,
            )
        axes_by_id[2].set_ylim(0.5, STATIC_MASS_SPLIT_AMU)
        axes_by_id[2].set_yscale("log")
        add_panel_colorbar(fig, axes_by_id[2], mesh, label="eflux", fontsize=cbar_fs)

    if 3 in axes_by_id:
        if mass_heavy_by_time and mass_heavy_axis_by_time:
            mesh = plot_variable_heatmap(
                axes_by_id[3],
                mass_heavy_by_time,
                static_times,
                mass_heavy_axis_by_time,
                "",
                f"STATIC Mass \n> {STATIC_MASS_SPLIT_AMU:g} amu",
                norm=mass_norm,
            )
        else:
            mesh = plot_heatmap(
                axes_by_id[3],
                mass_heavy_matrix,
                static_times,
                static.get("mass_amu_gt_1p5", []),
                "",
                f"STATIC Mass \n> {STATIC_MASS_SPLIT_AMU:g} amu",
                norm=mass_norm,
            )
        axes_by_id[3].set_ylim(bottom=STATIC_MASS_SPLIT_AMU)
        axes_by_id[3].set_yscale("log")
        add_panel_colorbar(fig, axes_by_id[3], mesh, label="eflux", fontsize=cbar_fs)

    swe = overview.get("swe") or {}
    swe_indices = window_indices(swe.get("times_unix"), center_unix, window_seconds)

    swe_times = (
        np.asarray(swe.get("times_unix", []), dtype=float)[swe_indices]
        if len(swe_indices)
        else []
    )

    electron_energy_matrix = (
        np.asarray(swe.get("omni_eflux", []), dtype=float)[swe_indices]
        if len(swe_indices)
        else []
    )

    if 4 in axes_by_id:
        mesh = plot_heatmap(
            axes_by_id[4],
            electron_energy_matrix,
            swe_times,
            swe.get("energy_eV", []),
            "",
            "SWE Electron\nEnergy (eV)",
            log_y=True,
            norm=LogNorm(vmin=1e3, vmax=1e9),
            cmap=FLUX_CMAP,
        )
        add_panel_colorbar(fig, axes_by_id[4], mesh, label="eflux", fontsize=cbar_fs)

    mag = overview.get("mag") or {}
    mag_indices = window_indices(mag.get("times_unix"), center_unix, window_seconds)

    mag_times = (
        np.asarray(mag.get("times_unix", []), dtype=float)[mag_indices]
        if len(mag_indices)
        else []
    )

    if 5 in axes_by_id:
        plot_line_panel(
            axes_by_id[5],
            mag_times,
            [
                (
                    "|B|",
                    LINE_COLORS["bmag"],
                    np.asarray(mag.get("bmag_nT", []), dtype=float)[mag_indices],
                )
            ],
            "",
            "|B|\n(nT)",
            y_range=(0.0, 50.0),
        )

    if 6 in axes_by_id:
        plot_line_panel(
            axes_by_id[6],
            mag_times,
            [
                (
                    "Bx",
                    LINE_COLORS["bx"],
                    np.asarray(mag.get("bx_nT", []), dtype=float)[mag_indices],
                ),
                (
                    "By",
                    LINE_COLORS["by"],
                    np.asarray(mag.get("by_nT", []), dtype=float)[mag_indices],
                ),
                (
                    "Bz",
                    LINE_COLORS["bz"],
                    np.asarray(mag.get("bz_nT", []), dtype=float)[mag_indices],
                ),
            ]
            if len(mag_indices)
            else [],
            "",
            "B_MSO\n(nT)",
            y_range=(-50.0, 50.0),
        )

    region = overview.get("region_id") or {}
    region_indices = window_indices(
        region.get("times_unix"),
        center_unix,
        window_seconds,
    )
    if REGION_ID_PANEL_ID in axes_by_id:
        region_times = (
            np.asarray(region.get("times_unix", []), dtype=float)[region_indices]
            if len(region_indices)
            else []
        )
        region_values = (
            np.asarray(region.get("region_id", []), dtype=float)[region_indices]
            if len(region_indices)
            else []
        )
        plot_region_id_panel(
            axes_by_id[REGION_ID_PANEL_ID],
            region_times,
            region_values,
        )

    resolved_pad_bands = []
    resolved_pad_panels = {}
    for panel_id in (7, 8):
        if panel_id not in axes_by_id:
            continue
        band_index = PAD_PANEL_BAND_INDEX[panel_id]
        pad_all, resolved_pad_band_eV = resolve_pad_panel_data(swe, band_index, pad_energy_bands_eV)
        pad_matrix = pad_all[swe_indices] if len(swe_indices) and pad_all.size else []
        pad_bands = swe.get("pad_bands") or []
        pad_pitch = (
            pad_bands[band_index].get("pitch_deg", [])
            if band_index < len(pad_bands)
            else swe.get("pitch_deg", [])
        )
        mesh = plot_heatmap(
            axes_by_id[panel_id],
            pad_matrix,
            swe_times,
            pad_pitch,
            "",
            f"SWE PAD\n({energy_band_label(resolved_pad_band_eV)})",
            norm=positive_log_norm(pad_matrix),
            cmap=PAD_CMAP,
        )
        add_panel_colorbar(fig, axes_by_id[panel_id], mesh, label="eflux", fontsize=cbar_fs)
        resolved_pad_bands.append(resolved_pad_band_eV)
        resolved_pad_panels[str(panel_id)] = list(resolved_pad_band_eV)

    if COORDINATE_PANEL_ID in axes_by_id:
        draw_bottom_coordinate_axis(
            axes_by_id[COORDINATE_PANEL_ID],
            mag,
            center_unix - window_seconds / 2.0,
            center_unix + window_seconds / 2.0,
        )

    data_panel_ids = [
        panel_id for panel_id in panel_ids if panel_id != COORDINATE_PANEL_ID
    ]
    for panel_id in data_panel_ids:
        ax = axes_by_id[panel_id]
        ax.set_xlim(window_start, window_end)
        mark_target_time(ax, target_unix)
        ax.grid(True, linestyle=":", alpha=0.25)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.yaxis.label.set_size(label_fs)

    if COORDINATE_PANEL_ID in panel_ids:
        for panel_id in data_panel_ids:
            axes_by_id[panel_id].tick_params(labelbottom=False)
    elif data_panel_ids:
        for panel_id in data_panel_ids[:-1]:
            axes_by_id[panel_id].tick_params(labelbottom=False)
        axes_by_id[data_panel_ids[-1]].set_xlabel("Time (UTC)", fontsize=label_fs)

    for panel_id in data_panel_ids:
        ax = axes_by_id[panel_id]
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(legend_fs)

    for panel_id in panel_ids:
        mark_panel_id(axes_by_id[panel_id], panel_id)

    # No figure title.
    # Do not call fig.suptitle(...)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "selected_index": selected_index,
        "selected_time": selected.get("target_time"),
        "panel_ids": list(panel_ids),
        "panel_names": [PANEL_CATALOG[panel_id]["name"] for panel_id in panel_ids],
        "pad_energy_bands_eV": [list(band) for band in resolved_pad_bands],
        "pad_energy_bands_by_panel": resolved_pad_panels,
        "output_path": str(output_path),
    }

def default_event_output_path(output_root: Path, target_time: datetime) -> Path:
    return output_root / target_time.strftime("%Y%m%dT%H%M%S") / "maven_data_panels.png"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render MAVEN data panels for one event time.")
    parser.add_argument(
        "--summary-json",
        help="Optional path to topology_summary.json or data_panel_context_summary.json. If omitted, local data files are used directly.",
    )
    parser.add_argument("--time", required=True, help="UTC target time.")
    parser.add_argument("--window-minutes", type=float, default=20.0)
    parser.add_argument("--step-seconds", type=int, default=60, help="Cadence for altitude samples when reading local data.")
    parser.add_argument(
        "--region-id-cadence-seconds",
        type=float,
        default=10.0,
        help="Classification cadence used only when panel 10 is selected.",
    )
    parser.add_argument(
        "--panels",
        "--panel-ids",
        nargs="+",
        default=[str(panel_id) for panel_id in DEFAULT_PANEL_IDS],
        metavar="ID",
        help=(
            "Panel IDs to include. Space-separated and comma-separated forms are "
            f"accepted; output follows catalog order. Catalog: {panel_catalog_help()}"
        ),
    )
    parser.add_argument(
        "--pad-energy-bands",
        "--pad-energy-band",
        nargs="+",
        type=float,
        default=flatten_energy_bands(DEFAULT_PAD_ENERGY_BANDS_EV),
        metavar="EV",
        help="Electron PAD energy bands as LOW HIGH pairs, for example: 20 80 111 140.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for local MAVEN data.")
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Download missing SWE/STATIC/MAG daily files. By default only local files are used.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory used for per-event output folders when --output is not supplied.",
    )
    parser.add_argument("--output", help="Explicit PNG output path. Overrides --output-root.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    target_time = parse_iso_timestamp(args.time)
    panel_ids = validate_panel_ids(args.panels)
    pad_energy_bands_eV = validate_energy_bands(args.pad_energy_bands)
    data_root = Path(args.data_root).expanduser().resolve()
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        log_step(f"Loading summary context: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        log_step(f"Building panel context directly from local data: {data_root}")
        summary = build_data_panel_summary_from_data(
            target_time=target_time,
            window_minutes=args.window_minutes,
            step_seconds=args.step_seconds,
            data_root=data_root,
            auto_download_missing_data=args.auto_download,
            pad_energy_bands_eV=pad_energy_bands_eV,
        )

    if REGION_ID_PANEL_ID in panel_ids:
        selected_index = nearest_sample_index(summary.get("samples", []), target_time)
        center_time = parse_iso_timestamp(
            summary["samples"][selected_index]["target_time"]
        )
        half_window = timedelta(minutes=args.window_minutes / 2.0)
        log_step(
            "Classifying region_id for panel 10: "
            f"{(center_time - half_window).isoformat()} to "
            f"{(center_time + half_window).isoformat()}"
        )
        summary.setdefault("context_overview", {})["region_id"] = build_region_id_context(
            start=center_time - half_window,
            end=center_time + half_window,
            data_root=data_root,
            cadence_seconds=args.region_id_cadence_seconds,
        )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_event_output_path(Path(args.output_root).expanduser().resolve(), target_time)
    )
    log_step(f"Selected panels: {', '.join(str(panel_id) for panel_id in panel_ids)}")
    log_step(f"Writing data panels to: {output_path}")
    result = plot_data_panels(
        summary=summary,
        target_time=target_time,
        output_path=output_path,
        window_minutes=args.window_minutes,
        pad_energy_bands_eV=pad_energy_bands_eV,
        panel_ids=panel_ids,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
