from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import cdflib
import numpy as np

from process_maven_spectra import (
    infer_daily_file,
    load_pad_data,
    unix_seconds_from_cdf_epoch,
    unix_seconds_from_numeric_time,
)


def iter_utc_days(start: datetime, end: datetime) -> list[datetime]:
    current = datetime.combine(start.date(), time.min, tzinfo=timezone.utc)
    last = datetime.combine(end.date(), time.min, tzinfo=timezone.utc)
    days: list[datetime] = []
    while current <= last:
        days.append(current)
        current += timedelta(days=1)
    return days


def nearest_feature_index(
    feature_data: dict | None,
    target_unix: float,
    max_delta_seconds: float,
) -> tuple[int, float] | None:
    if feature_data is None:
        return None
    times = np.asarray(feature_data.get("times_unix", []), dtype=float)
    if times.size == 0:
        return None
    insertion = int(np.searchsorted(times, float(target_unix), side="left"))
    candidates = [
        index
        for index in (insertion - 1, insertion)
        if 0 <= index < times.size
    ]
    index = min(candidates, key=lambda item: abs(float(times[item]) - target_unix))
    delta = abs(float(times[index]) - float(target_unix))
    if delta > max_delta_seconds:
        return None
    return index, delta


def _finite_band_median(
    energy_eV: np.ndarray,
    flux: np.ndarray,
    lower_eV: float,
    upper_eV: float,
) -> float:
    energy = np.asarray(energy_eV, dtype=float)
    values = np.asarray(flux, dtype=float)
    mask = (
        np.isfinite(energy)
        & np.isfinite(values)
        & (energy >= lower_eV)
        & (energy <= upper_eV)
        & (values > 0.0)
        & (values < 1e30)
    )
    if not np.any(mask):
        return float("nan")
    return float(np.nanmedian(values[mask]))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def _omnidirectional_flux(flux: np.ndarray) -> np.ndarray:
    values = np.asarray(flux, dtype=float)
    valid = np.isfinite(values) & (values > 0.0) & (values < 1e30)
    summed = np.sum(np.where(valid, values, 0.0), axis=1)
    counts = np.sum(valid, axis=1)
    return np.divide(
        summed,
        counts,
        out=np.full_like(summed, np.nan, dtype=float),
        where=counts > 0,
    )


def load_swe_features_interval(
    data_root: Path,
    start: datetime,
    end: datetime,
    photoelectron_ratio_threshold: float,
    electron_void_target_energy_eV: float,
    electron_void_flux_threshold: float,
) -> dict | None:
    times_parts: list[np.ndarray] = []
    photoelectron_parts: list[np.ndarray] = []
    ratio_parts: list[np.ndarray] = []
    flux_40_parts: list[np.ndarray] = []
    void_parts: list[np.ndarray] = []
    high_suppression_parts: list[np.ndarray] = []
    source_files: list[str] = []

    for day in iter_utc_days(start, end):
        try:
            path = infer_daily_file(data_root, "swe", "svypad", day, "cdf")
            pad = load_pad_data(path)
        except (FileNotFoundError, OSError, KeyError, ValueError):
            continue

        times = np.asarray(pad["times"], dtype=float)
        selected = np.where((times >= start.timestamp()) & (times <= end.timestamp()))[0]
        if selected.size == 0:
            continue

        energy = np.asarray(pad["energy"], dtype=float)
        omni = _omnidirectional_flux(np.asarray(pad["flux"], dtype=float)[selected])
        target_index = int(np.nanargmin(np.abs(energy - electron_void_target_energy_eV)))
        photoelectron_flags = np.zeros(selected.size, dtype=bool)
        photoelectron_ratios = np.full(selected.size, np.nan, dtype=float)
        target_flux = np.full(selected.size, np.nan, dtype=float)
        void_flags = np.zeros(selected.size, dtype=bool)
        high_suppression = np.full(selected.size, np.nan, dtype=float)

        for row_index, spectrum in enumerate(omni):
            center = _finite_band_median(energy, spectrum, 20.0, 30.0)
            left = _finite_band_median(energy, spectrum, 12.0, 18.0)
            right = _finite_band_median(energy, spectrum, 32.0, 50.0)
            shoulders = [value for value in (left, right) if np.isfinite(value) and value > 0.0]
            shoulder_level = max(shoulders) if shoulders else float("nan")
            ratio = _safe_ratio(center, shoulder_level)
            photoelectron_ratios[row_index] = ratio
            photoelectron_flags[row_index] = bool(
                np.isfinite(ratio) and ratio >= photoelectron_ratio_threshold
            )

            value_40 = float(spectrum[target_index])
            target_flux[row_index] = value_40
            void_flags[row_index] = bool(
                np.isfinite(value_40)
                and value_40 >= 0.0
                and value_40 < electron_void_flux_threshold
            )
            high_flux = _finite_band_median(energy, spectrum, 80.0, 120.0)
            high_suppression[row_index] = _safe_ratio(center, high_flux)

        times_parts.append(times[selected])
        photoelectron_parts.append(photoelectron_flags)
        ratio_parts.append(photoelectron_ratios)
        flux_40_parts.append(target_flux)
        void_parts.append(void_flags)
        high_suppression_parts.append(high_suppression)
        source_files.append(str(path))

    if not times_parts:
        return None

    times = np.concatenate(times_parts)
    order = np.argsort(times)
    return {
        "times_unix": times[order],
        "photoelectron_present": np.concatenate(photoelectron_parts)[order],
        "photoelectron_ratio": np.concatenate(ratio_parts)[order],
        "electron_flux_target": np.concatenate(flux_40_parts)[order],
        "electron_void": np.concatenate(void_parts)[order],
        "high_energy_suppression_ratio": np.concatenate(high_suppression_parts)[order],
        "electron_void_target_energy_eV": float(electron_void_target_energy_eV),
        "electron_void_flux_threshold": float(electron_void_flux_threshold),
        "source_files": source_files,
    }


def _cdf_variable_names(cdf: cdflib.CDF) -> set[str]:
    info = cdf.cdf_info()
    return set(info.zVariables) | set(info.rVariables)


def _cdf_times(cdf: cdflib.CDF) -> np.ndarray:
    available = _cdf_variable_names(cdf)
    for name in ("epoch", "time_unix", "time_met"):
        if name not in available:
            continue
        values = np.asarray(cdf.varget(name))
        if name == "epoch" or (values.dtype.kind in {"i", "u"} and np.nanmedian(values) > 1e12):
            return unix_seconds_from_cdf_epoch(values)
        return unix_seconds_from_numeric_time(values)
    raise KeyError("No usable STATIC time variable was found.")


def _varget_records(
    cdf: cdflib.CDF,
    name: str,
    first_record: int,
    last_record: int,
) -> np.ndarray:
    try:
        return np.asarray(
            cdf.varget(name, startrec=int(first_record), endrec=int(last_record))
        )
    except (TypeError, ValueError):
        return np.asarray(cdf.varget(name))[first_record : last_record + 1]


def _infer_sweep_indices(values: np.ndarray, n_sweeps: int) -> np.ndarray:
    raw = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(raw)
    result = np.full(raw.shape, -1, dtype=int)
    result[finite] = raw[finite].astype(int)
    valid_values = result[finite]
    if valid_values.size and valid_values.min() >= 1 and valid_values.max() == n_sweeps:
        result[finite] -= 1
    return result


def _weighted_peak_energy(
    energy: np.ndarray,
    flux: np.ndarray,
    population_mask: np.ndarray,
) -> float:
    channel_flux = np.sum(np.where(population_mask, flux, 0.0), axis=0)
    channel_energy = np.full(channel_flux.shape, np.nan, dtype=float)
    for channel in range(channel_flux.size):
        values = energy[:, channel][population_mask[:, channel]]
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            channel_energy[channel] = float(np.nanmedian(values))
    usable = np.isfinite(channel_energy) & np.isfinite(channel_flux) & (channel_flux > 0.0)
    if not np.any(usable):
        return float("nan")
    usable_indices = np.where(usable)[0]
    peak_channel = int(usable_indices[np.argmax(channel_flux[usable])])
    return float(channel_energy[peak_channel])


def _weighted_log_width(
    energy: np.ndarray,
    flux: np.ndarray,
    population_mask: np.ndarray,
) -> float:
    valid = (
        population_mask
        & np.isfinite(energy)
        & (energy > 0.0)
        & np.isfinite(flux)
        & (flux > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        return float("nan")
    x = np.log10(energy[valid])
    weights = flux[valid]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return float("nan")
    mean = float(np.sum(weights * x) / weight_sum)
    variance = float(np.sum(weights * (x - mean) ** 2) / weight_sum)
    return float(np.sqrt(max(variance, 0.0)))


def load_static_features_interval(
    data_root: Path,
    start: datetime,
    end: datetime,
) -> dict | None:
    records: list[dict] = []
    source_files: list[str] = []

    for day in iter_utc_days(start, end):
        try:
            path = infer_daily_file(data_root, "sta", "c6-32e64m", day, "cdf")
            cdf = cdflib.CDF(str(path))
            times = _cdf_times(cdf)
        except (FileNotFoundError, OSError, KeyError, ValueError):
            continue

        selected = np.where((times >= start.timestamp()) & (times <= end.timestamp()))[0]
        if selected.size == 0:
            continue
        available = _cdf_variable_names(cdf)
        required = {"eflux", "energy", "mass_arr", "swp_ind"}
        if not required.issubset(available):
            continue

        first = int(selected[0])
        last = int(selected[-1])
        eflux_rows = _varget_records(cdf, "eflux", first, last)
        sweep_values = _varget_records(cdf, "swp_ind", first, last)
        energy_lut = np.asarray(cdf.varget("energy"), dtype=float)
        mass_lut = np.asarray(cdf.varget("mass_arr"), dtype=float)
        if eflux_rows.ndim == 2:
            eflux_rows = eflux_rows[None, ...]
        if eflux_rows.ndim != 3 or energy_lut.ndim != 3 or mass_lut.ndim != 3:
            continue

        sweep_indices = _infer_sweep_indices(sweep_values, energy_lut.shape[2])
        selected_set = set(int(index) for index in selected)
        for local_index, global_index in enumerate(range(first, last + 1)):
            if global_index not in selected_set:
                continue
            sweep = int(sweep_indices[local_index])
            if sweep < 0 or sweep >= energy_lut.shape[2]:
                continue

            flux = np.asarray(eflux_rows[local_index], dtype=float)
            energy = np.asarray(energy_lut[:, :, sweep], dtype=float)
            mass = np.asarray(mass_lut[:, :, sweep], dtype=float)
            valid = (
                np.isfinite(flux)
                & (flux > 0.0)
                & (flux < 1e30)
                & np.isfinite(energy)
                & (energy > 0.0)
                & np.isfinite(mass)
                & (mass > 0.0)
            )
            clean_flux = np.where(valid, flux, 0.0)
            total_flux = float(np.sum(clean_flux))
            hplus_mask = valid & (mass >= 0.5) & (mass <= 2.0)
            planetary_heavy_mask = valid & (mass >= 12.0)
            hplus_flux = float(np.sum(np.where(hplus_mask, clean_flux, 0.0)))
            heavy_flux = float(np.sum(np.where(planetary_heavy_mask, clean_flux, 0.0)))

            records.append(
                {
                    "time_unix": float(times[global_index]),
                    "static_valid_bin_count": int(np.count_nonzero(valid)),
                    "planetary_heavy_ion_valid_bin_count": int(
                        np.count_nonzero(planetary_heavy_mask)
                    ),
                    "total_valid_ion_flux": total_flux,
                    "planetary_heavy_ion_integrated_flux": heavy_flux,
                    "hplus_flux_fraction": (
                        hplus_flux / total_flux if total_flux > 0.0 else float("nan")
                    ),
                    "planetary_heavy_ion_flux_fraction": (
                        heavy_flux / total_flux if total_flux > 0.0 else float("nan")
                    ),
                    "hplus_peak_energy_eV": _weighted_peak_energy(
                        energy,
                        clean_flux,
                        hplus_mask,
                    ),
                    "heavy_ion_peak_energy_eV": _weighted_peak_energy(
                        energy,
                        clean_flux,
                        planetary_heavy_mask,
                    ),
                    "hplus_log_energy_width": _weighted_log_width(
                        energy,
                        clean_flux,
                        hplus_mask,
                    ),
                }
            )
        source_files.append(str(path))

    if not records:
        return None

    records.sort(key=lambda row: float(row["time_unix"]))
    keys = [key for key in records[0] if key != "time_unix"]
    result = {
        "times_unix": np.asarray([row["time_unix"] for row in records], dtype=float),
        "source_files": source_files,
    }
    for key in keys:
        result[key] = np.asarray([row[key] for row in records], dtype=float)
    return result
