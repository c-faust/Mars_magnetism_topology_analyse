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
    data_root: Path | tuple[Path, ...] | list[Path],
    start: datetime,
    end: datetime,
    photoelectron_ratio_threshold: float,
    electron_void_target_energy_eV: float,
    electron_void_flux_threshold: float,
    electron_depletion_lower_eV: float = 30.0,
    electron_depletion_upper_eV: float = 80.0,
    electron_depletion_flux_threshold: float = 1.0e5,
    electron_depletion_min_valid_bins: int = 3,
    electron_depletion_min_low_fraction: float = 0.75,
) -> dict | None:
    times_parts: list[np.ndarray] = []
    photoelectron_parts: list[np.ndarray] = []
    ratio_parts: list[np.ndarray] = []
    flux_40_parts: list[np.ndarray] = []
    void_parts: list[np.ndarray] = []
    high_suppression_parts: list[np.ndarray] = []
    depletion_median_parts: list[np.ndarray] = []
    depletion_fraction_parts: list[np.ndarray] = []
    depletion_bin_count_parts: list[np.ndarray] = []
    depletion_flag_parts: list[np.ndarray] = []
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
        depletion_median = np.full(selected.size, np.nan, dtype=float)
        depletion_fraction = np.full(selected.size, np.nan, dtype=float)
        depletion_bin_count = np.zeros(selected.size, dtype=int)
        depletion_flag = np.zeros(selected.size, dtype=bool)

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

            depletion_band = (
                np.isfinite(energy)
                & (energy >= electron_depletion_lower_eV)
                & (energy <= electron_depletion_upper_eV)
                & np.isfinite(spectrum)
                & (spectrum > 0.0)
                & (spectrum < 1e30)
            )
            band_values = spectrum[depletion_band]
            depletion_bin_count[row_index] = int(band_values.size)
            if band_values.size:
                depletion_median[row_index] = float(np.nanmedian(band_values))
                depletion_fraction[row_index] = float(
                    np.mean(band_values < electron_depletion_flux_threshold)
                )
            depletion_flag[row_index] = bool(
                band_values.size >= electron_depletion_min_valid_bins
                and np.isfinite(depletion_fraction[row_index])
                and depletion_fraction[row_index] >= electron_depletion_min_low_fraction
                and np.isfinite(depletion_median[row_index])
                and depletion_median[row_index] < electron_depletion_flux_threshold
            )

        times_parts.append(times[selected])
        photoelectron_parts.append(photoelectron_flags)
        ratio_parts.append(photoelectron_ratios)
        flux_40_parts.append(target_flux)
        void_parts.append(void_flags)
        high_suppression_parts.append(high_suppression)
        depletion_median_parts.append(depletion_median)
        depletion_fraction_parts.append(depletion_fraction)
        depletion_bin_count_parts.append(depletion_bin_count)
        depletion_flag_parts.append(depletion_flag)
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
        "electron_depletion_band_median_flux": np.concatenate(
            depletion_median_parts
        )[order],
        "electron_depletion_low_fraction": np.concatenate(
            depletion_fraction_parts
        )[order],
        "electron_depletion_valid_bin_count": np.concatenate(
            depletion_bin_count_parts
        )[order],
        "multichannel_electron_depletion": np.concatenate(
            depletion_flag_parts
        )[order],
        "electron_void_target_energy_eV": float(electron_void_target_energy_eV),
        "electron_void_flux_threshold": float(electron_void_flux_threshold),
        "electron_depletion_band_eV": [
            float(electron_depletion_lower_eV),
            float(electron_depletion_upper_eV),
        ],
        "electron_depletion_flux_threshold": float(
            electron_depletion_flux_threshold
        ),
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
    raise KeyError("No usable CDF time variable was found.")


def _cdf_attribute_scalar(cdf: cdflib.CDF, name: str, attribute: str) -> float:
    """Return a numeric CDF variable attribute, or NaN when it is absent."""
    try:
        value = cdf.varattsget(name).get(attribute)
        array = np.asarray(value, dtype=float).reshape(-1)
    except (AttributeError, KeyError, TypeError, ValueError):
        return float("nan")
    return float(array[0]) if array.size else float("nan")


def _clean_cdf_values(cdf: cdflib.CDF, name: str, values: np.ndarray) -> np.ndarray:
    """Convert documented fill/out-of-range CDF samples to NaN."""
    result = np.asarray(values, dtype=float).copy()
    fill = _cdf_attribute_scalar(cdf, name, "FILLVAL")
    valid_min = _cdf_attribute_scalar(cdf, name, "VALIDMIN")
    valid_max = _cdf_attribute_scalar(cdf, name, "VALIDMAX")
    invalid = ~np.isfinite(result)
    if np.isfinite(fill):
        invalid |= np.isclose(result, fill, rtol=1.0e-7, atol=0.0)
    if np.isfinite(valid_min):
        invalid |= result < valid_min
    if np.isfinite(valid_max):
        invalid |= result > valid_max
    result[invalid] = np.nan
    return result


def _load_cdf_records_for_interval(
    cdf: cdflib.CDF,
    name: str,
    selected: np.ndarray,
) -> np.ndarray:
    first = int(selected[0])
    last = int(selected[-1])
    values = _varget_records(cdf, name, first, last)
    relative = selected - first
    return np.asarray(values)[relative]


def load_swia_moments_interval(
    data_root: Path | tuple[Path, ...] | list[Path],
    start: datetime,
    end: datetime,
) -> dict | None:
    """Load quality-screened SWIA onboard survey proton moments.

    The onboard moment product assumes a proton distribution. Density and bulk
    velocity are used as primary region evidence; temperature is retained as an
    auxiliary feature because alpha particles and field-of-view limitations can
    bias it.
    """
    keys = (
        "density_cm3",
        "velocity_mso_km_s",
        "speed_km_s",
        "temperature_mso_eV",
        "temperature_eV",
        "quality_flag",
        "decom_flag",
        "telem_mode",
        "atten_state",
        "swia_moment_quality_valid",
        "swia_temperature_valid",
    )
    parts: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    time_parts: list[np.ndarray] = []
    source_files: list[str] = []

    for day in iter_utc_days(start, end):
        try:
            path = infer_daily_file(data_root, "swi", "onboardsvymom", day, "cdf")
            cdf = cdflib.CDF(str(path))
            times = _cdf_times(cdf)
        except (FileNotFoundError, OSError, KeyError, ValueError):
            continue
        selected = np.where((times >= start.timestamp()) & (times <= end.timestamp()))[0]
        available = _cdf_variable_names(cdf)
        required = {"density", "velocity_mso", "quality_flag", "decom_flag"}
        if selected.size == 0 or not required.issubset(available):
            continue

        density = _clean_cdf_values(
            cdf,
            "density",
            _load_cdf_records_for_interval(cdf, "density", selected),
        ).reshape(-1)
        velocity = _clean_cdf_values(
            cdf,
            "velocity_mso",
            _load_cdf_records_for_interval(cdf, "velocity_mso", selected),
        )
        if velocity.ndim != 2 or velocity.shape[1] < 3:
            continue
        velocity = velocity[:, :3]
        speed = np.linalg.norm(velocity, axis=1)
        quality = np.asarray(
            _load_cdf_records_for_interval(cdf, "quality_flag", selected),
            dtype=float,
        ).reshape(-1)
        decom = np.asarray(
            _load_cdf_records_for_interval(cdf, "decom_flag", selected),
            dtype=float,
        ).reshape(-1)

        optional_scalars: dict[str, np.ndarray] = {}
        for name in ("telem_mode", "atten_state"):
            if name in available:
                optional_scalars[name] = np.asarray(
                    _load_cdf_records_for_interval(cdf, name, selected),
                    dtype=float,
                ).reshape(-1)
            else:
                optional_scalars[name] = np.full(selected.size, np.nan, dtype=float)

        if "temperature_mso" in available:
            temperature_vector = _clean_cdf_values(
                cdf,
                "temperature_mso",
                _load_cdf_records_for_interval(cdf, "temperature_mso", selected),
            )
            if temperature_vector.ndim == 2 and temperature_vector.shape[1] >= 3:
                temperature_vector = temperature_vector[:, :3]
                temperature = np.nanmean(temperature_vector, axis=1)
            else:
                temperature_vector = np.full((selected.size, 3), np.nan, dtype=float)
                temperature = np.full(selected.size, np.nan, dtype=float)
        else:
            temperature_vector = np.full((selected.size, 3), np.nan, dtype=float)
            temperature = np.full(selected.size, np.nan, dtype=float)

        moment_valid = (
            (quality == 1.0)
            & (decom == 1.0)
            & (optional_scalars["atten_state"] != 3.0)
            & np.isfinite(density)
            & (density > 0.0)
            & np.all(np.isfinite(velocity), axis=1)
            & np.isfinite(speed)
            & (speed > 0.0)
        )
        temperature_valid = moment_valid & np.isfinite(temperature) & (temperature > 0.0)

        time_parts.append(times[selected])
        parts["density_cm3"].append(density)
        parts["velocity_mso_km_s"].append(velocity)
        parts["speed_km_s"].append(speed)
        parts["temperature_mso_eV"].append(temperature_vector)
        parts["temperature_eV"].append(temperature)
        parts["quality_flag"].append(quality)
        parts["decom_flag"].append(decom)
        parts["telem_mode"].append(optional_scalars["telem_mode"])
        parts["atten_state"].append(optional_scalars["atten_state"])
        parts["swia_moment_quality_valid"].append(moment_valid)
        parts["swia_temperature_valid"].append(temperature_valid)
        source_files.append(str(path))

    if not time_parts:
        return None
    times = np.concatenate(time_parts)
    order = np.argsort(times)
    result: dict[str, object] = {
        "times_unix": times[order],
        "source_files": source_files,
    }
    for key, arrays in parts.items():
        result[key] = np.concatenate(arrays, axis=0)[order]
    return result


def load_swia_spectra_interval(
    data_root: Path | tuple[Path, ...] | list[Path],
    start: datetime,
    end: datetime,
) -> dict | None:
    """Load compact shape features from SWIA onboard survey energy spectra."""
    time_parts: list[np.ndarray] = []
    peak_parts: list[np.ndarray] = []
    width_parts: list[np.ndarray] = []
    entropy_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    source_files: list[str] = []

    for day in iter_utc_days(start, end):
        try:
            path = infer_daily_file(data_root, "swi", "onboardsvyspec", day, "cdf")
            cdf = cdflib.CDF(str(path))
            times = _cdf_times(cdf)
        except (FileNotFoundError, OSError, KeyError, ValueError):
            continue
        selected = np.where((times >= start.timestamp()) & (times <= end.timestamp()))[0]
        available = _cdf_variable_names(cdf)
        required = {"energy_spectra", "spectra_diff_en_fluxes"}
        if selected.size == 0 or not required.issubset(available):
            continue

        energy_raw = np.asarray(cdf.varget("energy_spectra"), dtype=float)
        if energy_raw.ndim == 1:
            energy = np.broadcast_to(energy_raw, (selected.size, energy_raw.size))
        else:
            energy = _load_cdf_records_for_interval(cdf, "energy_spectra", selected)
        flux = _clean_cdf_values(
            cdf,
            "spectra_diff_en_fluxes",
            _load_cdf_records_for_interval(cdf, "spectra_diff_en_fluxes", selected),
        )
        if flux.ndim != 2 or energy.shape != flux.shape:
            continue
        if "spectra_counts" in available:
            counts = _clean_cdf_values(
                cdf,
                "spectra_counts",
                _load_cdf_records_for_interval(cdf, "spectra_counts", selected),
            )
            measured = np.isfinite(counts) & (counts > 0.0)
        else:
            measured = np.ones(flux.shape, dtype=bool)
        valid = (
            measured
            & np.isfinite(energy)
            & (energy > 0.0)
            & np.isfinite(flux)
            & (flux > 0.0)
        )
        peak = np.full(selected.size, np.nan, dtype=float)
        width = np.full(selected.size, np.nan, dtype=float)
        entropy = np.full(selected.size, np.nan, dtype=float)
        valid_count = np.sum(valid, axis=1)
        for row_index in range(selected.size):
            row_valid = valid[row_index]
            if np.count_nonzero(row_valid) < 3:
                continue
            row_energy = energy[row_index, row_valid]
            row_flux = flux[row_index, row_valid]
            peak[row_index] = float(row_energy[int(np.argmax(row_flux))])
            log_energy = np.log10(row_energy)
            weight_sum = float(np.sum(row_flux))
            weights = row_flux / weight_sum
            center = float(np.sum(weights * log_energy))
            width[row_index] = float(
                np.sqrt(max(float(np.sum(weights * (log_energy - center) ** 2)), 0.0))
            )
            entropy[row_index] = float(
                -np.sum(weights * np.log(np.clip(weights, 1.0e-300, None)))
                / np.log(weights.size)
            )

        decom = (
            np.asarray(
                _load_cdf_records_for_interval(cdf, "decom_flag", selected),
                dtype=float,
            ).reshape(-1)
            if "decom_flag" in available
            else np.ones(selected.size, dtype=float)
        )
        atten = (
            np.asarray(
                _load_cdf_records_for_interval(cdf, "atten_state", selected),
                dtype=float,
            ).reshape(-1)
            if "atten_state" in available
            else np.full(selected.size, np.nan, dtype=float)
        )
        spectrum_valid = (decom == 1.0) & (atten != 3.0) & (valid_count >= 3)
        time_parts.append(times[selected])
        peak_parts.append(peak)
        width_parts.append(width)
        entropy_parts.append(entropy)
        count_parts.append(valid_count)
        valid_parts.append(spectrum_valid)
        source_files.append(str(path))

    if not time_parts:
        return None
    times = np.concatenate(time_parts)
    order = np.argsort(times)
    return {
        "times_unix": times[order],
        "swia_spectrum_peak_energy_eV": np.concatenate(peak_parts)[order],
        "swia_spectrum_log_energy_width": np.concatenate(width_parts)[order],
        "swia_spectrum_entropy": np.concatenate(entropy_parts)[order],
        "swia_spectrum_valid_bin_count": np.concatenate(count_parts)[order],
        "swia_spectrum_quality_valid": np.concatenate(valid_parts)[order],
        "source_files": source_files,
    }


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
    data_root: Path | tuple[Path, ...] | list[Path],
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
