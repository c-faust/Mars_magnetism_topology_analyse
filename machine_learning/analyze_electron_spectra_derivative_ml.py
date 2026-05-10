from __future__ import annotations
"""
Machine-learning exploration of MAVEN SWE electron-spectra derivatives.

This script mirrors `analyze_electron_spectra_ml.py`, but the feature vector is
the energy derivative of the paired spectra:

    [d(parallel flux)/dE, d(anti_parallel flux)/dE]

Each timestamp remains one sample when `--direction both` is used.
"""

import argparse
import csv
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import cdflib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from download_maven_data import PIPELINE_PRODUCTS, build_session, download_product_for_day, parse_filename
from process_maven_spectra import (
    format_unix_time,
    load_pad_data,
    pick_first_variable,
    unix_seconds_from_cdf_epoch,
    unix_seconds_from_numeric_time,
)
from machine_learning.analyze_electron_spectra_ml import (
    DEFAULT_DATA_ROOT,
    ML_ROOT,
    SpectrumSample,
    build_run_name,
    choose_cluster_count,
    clean_flux_matrix,
    ensure_required_swe_data,
    extract_directional_fluxes,
    infer_swe_files,
    log_step,
    parse_iso_datetime,
    parse_iso_time,
    pca_scores,
    representative_indices,
    sanitize_for_json,
    unique_output_dir,
    valid_flux_fraction,
)


DEFAULT_OUTPUT_ROOT = ML_ROOT / "outputs" / "derivative_analysis"
LPW_PRODUCTS = tuple(spec for spec in PIPELINE_PRODUCTS if spec.instrument == "lpw" and spec.datatype == "mrgscpot")
DEFAULT_SCPOT_MIN_FLAG = 50.0
DEFAULT_MIN_PLASMA_ENERGY_EV = 5.0


def maven_file_date(path: Path) -> date | None:
    parsed = parse_filename(path.name)
    if not parsed:
        return None
    return date(int(parsed["year"]), int(parsed["month"]), int(parsed["day"]))


def find_local_lpw_files_for_day(data_root: Path, day: date) -> list[Path]:
    day_code = day.strftime("%Y%m%d")
    matches: list[tuple[int, int, Path]] = []
    for path in data_root.rglob("mvn_lpw_l2_mrgscpot_*.cdf"):
        parsed = parse_filename(path.name)
        if not parsed:
            continue
        if f"{parsed['year']}{parsed['month']}{parsed['day']}" == day_code:
            matches.append((int(parsed["version"]), int(parsed["revision"]), path))
    return [path for _, _, path in sorted(matches, key=lambda item: (item[0], item[1], str(item[2])), reverse=True)]


def load_lpw_spacecraft_potential(path: Path, min_flag: float = DEFAULT_SCPOT_MIN_FLAG) -> dict[str, np.ndarray]:
    cdf = cdflib.CDF(str(path))
    time_values = pick_first_variable(cdf, ["time_unix", "epoch", "time_met"])
    potential = pick_first_variable(cdf, ["data", "spacecraft_potential", "scpot"])
    flag = pick_first_variable(cdf, ["flag"])
    if time_values is None or potential is None or flag is None:
        raise KeyError(f"No usable time/data/flag variables were found in {path.name}.")

    if time_values.dtype.kind in {"i", "u"} and np.nanmedian(time_values) > 1e12:
        times = unix_seconds_from_cdf_epoch(time_values)
    else:
        times = unix_seconds_from_numeric_time(time_values)

    times = np.asarray(times, dtype=float).reshape(-1)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    flag = np.asarray(flag, dtype=float).reshape(-1)
    usable = np.isfinite(times) & np.isfinite(potential) & np.isfinite(flag) & (flag > min_flag)
    if not np.any(usable):
        raise ValueError(f"No high-quality LPW spacecraft-potential samples with flag>{min_flag:g} in {path.name}.")

    good_times = times[usable]
    good_potential = potential[usable]
    good_flag = flag[usable]
    order = np.argsort(good_times)
    good_times = good_times[order]
    good_potential = good_potential[order]
    good_flag = good_flag[order]
    unique_times, unique_indices = np.unique(good_times, return_index=True)
    return {
        "times": unique_times,
        "spacecraft_potential": good_potential[unique_indices],
        "flag": good_flag[unique_indices],
    }


def inspect_lpw_file(path: Path, min_flag: float = DEFAULT_SCPOT_MIN_FLAG) -> dict:
    try:
        data = load_lpw_spacecraft_potential(path, min_flag=min_flag)
        times = data["times"]
        return {
            "status": "ok",
            "path": str(path),
            "time_start_utc": format_unix_time(float(times[0])),
            "time_end_utc": format_unix_time(float(times[-1])),
            "high_quality_samples": int(times.size),
            "min_flag": min_flag,
        }
    except Exception as exc:
        return {
            "status": "read_error",
            "path": str(path),
            "time_start_utc": None,
            "time_end_utc": None,
            "high_quality_samples": 0,
            "min_flag": min_flag,
            "error": f"{type(exc).__name__}: {exc}",
        }


def ensure_required_lpw_data(
    data_root: Path,
    required_dates: list[date],
    auto_download: bool,
    min_flag: float = DEFAULT_SCPOT_MIN_FLAG,
) -> tuple[dict[date, Path], dict]:
    unique_dates = sorted(set(required_dates))
    log_step(f"Checking local LPW mrgscpot coverage for {len(unique_dates)} SWE day(s).")
    report = {
        "required_dates": [item.isoformat() for item in unique_dates],
        "available": [],
        "missing_dates": [],
        "corrupt": [],
        "downloaded": [],
        "min_flag": min_flag,
    }
    usable_files_by_date: dict[date, Path] = {}

    for day in unique_dates:
        candidates = find_local_lpw_files_for_day(data_root, day)
        if not candidates:
            log_step(f"Missing local LPW mrgscpot file for {day.isoformat()}.")
            report["missing_dates"].append(day.isoformat())
            continue

        for path in candidates:
            info = inspect_lpw_file(path, min_flag=min_flag)
            if info["status"] == "ok":
                usable_files_by_date[day] = path
                report["available"].append({"date": day.isoformat(), **info})
                log_step(f"Available LPW {day.isoformat()}: {path.name}, {info['high_quality_samples']} high-quality sample(s).")
                break
            report["corrupt"].append({"date": day.isoformat(), **info})
            log_step(f"Unreadable LPW file for {day.isoformat()}: {path.name} ({info.get('error', info['status'])}).")

        if day not in usable_files_by_date:
            report["missing_dates"].append(day.isoformat())

    missing_dates = [date.fromisoformat(item) for item in report["missing_dates"]]
    if missing_dates and auto_download:
        if not LPW_PRODUCTS:
            raise RuntimeError("No LPW mrgscpot product specification is available for auto-download.")
        log_step(f"Auto-download enabled; downloading {len(missing_dates)} missing LPW day(s).")
        session = build_session()
        spec = LPW_PRODUCTS[0]
        for index, day in enumerate(missing_dates, start=1):
            try:
                log_step(f"Downloading missing LPW day {index}/{len(missing_dates)}: {day.isoformat()}")
                local_path = download_product_for_day(session=session, spec=spec, day=day, data_root=data_root)
                info = inspect_lpw_file(local_path, min_flag=min_flag)
                report["downloaded"].append({"date": day.isoformat(), **info})
                if info["status"] == "ok":
                    usable_files_by_date[day] = local_path
            except FileNotFoundError as exc:
                report["downloaded"].append({"date": day.isoformat(), "status": "missing_remote", "error": str(exc)})
                log_step(f"No remote LPW mrgscpot file for {day.isoformat()}; SWE data for this day will be skipped.")

    skipped_dates = [day.isoformat() for day in unique_dates if day not in usable_files_by_date]
    report["skipped_swe_dates_without_usable_lpw"] = skipped_dates
    log_step(f"LPW coverage check complete: {len(usable_files_by_date)}/{len(unique_dates)} SWE day(s) usable.")
    return usable_files_by_date, report


def interpolate_spacecraft_potential(lpw_data: dict[str, np.ndarray], target_times: np.ndarray) -> np.ndarray:
    times = np.asarray(lpw_data["times"], dtype=float)
    potential = np.asarray(lpw_data["spacecraft_potential"], dtype=float)
    targets = np.asarray(target_times, dtype=float)
    if times.size < 2:
        return np.full(targets.shape, np.nan, dtype=float)
    return np.interp(targets, times, potential, left=np.nan, right=np.nan)


def correct_flux_to_plasma_energy(
    measured_energy_eV: np.ndarray,
    flux: np.ndarray,
    spacecraft_potential_V: float,
    target_energy_eV: np.ndarray,
    min_plasma_energy_eV: float = DEFAULT_MIN_PLASMA_ENERGY_EV,
) -> np.ndarray:
    corrected_energy = np.asarray(measured_energy_eV, dtype=float) - float(spacecraft_potential_V)
    values = np.asarray(flux, dtype=float)
    usable = (
        np.isfinite(corrected_energy)
        & np.isfinite(values)
        & (corrected_energy >= min_plasma_energy_eV)
    )
    if np.count_nonzero(usable) < 2:
        return np.full(target_energy_eV.shape, np.nan, dtype=float)

    source_energy = corrected_energy[usable]
    source_flux = values[usable]
    order = np.argsort(source_energy)
    source_energy = source_energy[order]
    source_flux = source_flux[order]
    unique_energy, unique_indices = np.unique(source_energy, return_index=True)
    unique_flux = source_flux[unique_indices]
    if unique_energy.size < 2:
        return np.full(target_energy_eV.shape, np.nan, dtype=float)
    return np.interp(target_energy_eV, unique_energy, unique_flux, left=np.nan, right=np.nan)


def normalize_derivative_features(matrix: np.ndarray, method: str) -> np.ndarray:
    data = np.asarray(matrix, dtype=float).copy()
    data[~np.isfinite(data)] = 0.0

    if method == "none":
        return data
    if method == "zscore":
        center = np.nanmean(data, axis=1, keepdims=True)
        scale = np.nanstd(data, axis=1, keepdims=True)
        scale[scale == 0.0] = 1.0
        return (data - center) / scale
    if method == "global_zscore":
        center = np.nanmean(data, axis=0, keepdims=True)
        scale = np.nanstd(data, axis=0, keepdims=True)
        scale[scale == 0.0] = 1.0
        return (data - center) / scale
    if method == "minmax":
        lower = np.nanmin(data, axis=1, keepdims=True)
        upper = np.nanmax(data, axis=1, keepdims=True)
        span = upper - lower
        span[span == 0.0] = 1.0
        return (data - lower) / span
    if method == "l2":
        norm = np.linalg.norm(data, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        return data / norm
    raise ValueError(f"Unsupported derivative normalization method: {method}")


def flux_derivative(energy_eV: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    energy = np.asarray(energy_eV, dtype=float)
    cleaned_flux = clean_flux_matrix(np.asarray(flux, dtype=float))
    order = np.argsort(energy)
    sorted_energy = energy[order]
    sorted_flux = cleaned_flux[order]
    if sorted_energy.size < 2:
        return sorted_energy, np.zeros_like(sorted_flux)
    derivative = np.gradient(sorted_flux, sorted_energy, edge_order=1)
    derivative[~np.isfinite(derivative)] = 0.0
    return sorted_energy, derivative


def load_derivative_samples(
    files: list[Path],
    start_unix: float | None,
    end_unix: float | None,
    stride: int,
    normalization: str,
    direction: str,
    parallel_pitch_max_deg: float,
    anti_parallel_pitch_min_deg: float,
    min_direction_valid_fraction: float,
    lpw_files_by_date: dict[date, Path] | None = None,
    spacecraft_potential_min_flag: float = DEFAULT_SCPOT_MIN_FLAG,
    min_plasma_energy_eV: float = DEFAULT_MIN_PLASMA_ENERGY_EV,
) -> tuple[np.ndarray, list[SpectrumSample]]:
    all_times: list[float] = []
    all_parallel_derivatives: list[np.ndarray] = []
    all_anti_derivatives: list[np.ndarray] = []
    all_feature_vectors: list[np.ndarray] = []
    all_files: list[str] = []
    derivative_energy: np.ndarray | None = None
    skipped_sparse = 0
    skipped_no_lpw = 0
    skipped_no_spacecraft_potential = 0
    lpw_cache: dict[date, dict[str, np.ndarray]] = {}
    apply_spacecraft_potential = lpw_files_by_date is not None

    if direction == "both":
        log_step("Derivative feature mode: paired dF/dE parallel + anti_parallel; one timestamp is one sample.")
    else:
        log_step(f"Derivative feature mode: single-direction dF/dE; using {direction}.")
    log_step(f"Minimum valid positive energy-bin fraction per used direction: {min_direction_valid_fraction:g}.")
    if apply_spacecraft_potential:
        log_step(
            "Applying LPW spacecraft-potential correction "
            f"(flag>{spacecraft_potential_min_flag:g}, Eplasma>= {min_plasma_energy_eV:g} eV)."
        )

    for file_index, path in enumerate(files, start=1):
        log_step(f"Loading SWE file {file_index}/{len(files)}: {path.name}")
        file_day = maven_file_date(path)
        lpw_data = None
        if apply_spacecraft_potential:
            if file_day is None or file_day not in lpw_files_by_date:
                skipped_no_lpw += 1
                log_step(f"No usable LPW mrgscpot file for {path.name}; skipping this SWE day.")
                continue
            if file_day not in lpw_cache:
                lpw_cache[file_day] = load_lpw_spacecraft_potential(
                    lpw_files_by_date[file_day],
                    min_flag=spacecraft_potential_min_flag,
                )
            lpw_data = lpw_cache[file_day]

        pad_data = load_pad_data(path)
        times = np.asarray(pad_data["times"], dtype=float)
        flux = np.asarray(pad_data["flux"], dtype=float)
        energy = np.asarray(pad_data["energy"], dtype=float)
        target_plasma_energy = None
        if apply_spacecraft_potential:
            sorted_measured_energy = np.sort(energy[np.isfinite(energy)])
            target_plasma_energy = sorted_measured_energy[sorted_measured_energy >= min_plasma_energy_eV]
            if target_plasma_energy.size < 2:
                raise ValueError(f"Energy grid in {path} has fewer than two bins at or above {min_plasma_energy_eV:g} eV.")
            if derivative_energy is None:
                derivative_energy = target_plasma_energy
            elif derivative_energy.shape != target_plasma_energy.shape or not np.allclose(derivative_energy, target_plasma_energy):
                log_step(f"Energy grid in {path.name} differs; spectra will be interpolated onto the first plasma-energy grid.")
                target_plasma_energy = derivative_energy

        mask = np.ones(times.shape, dtype=bool)
        if start_unix is not None:
            mask &= times >= start_unix
        if end_unix is not None:
            mask &= times <= end_unix
        selected = np.flatnonzero(mask)[:: max(stride, 1)]
        if selected.size == 0:
            log_step(f"No timestamps selected from {path.name}; skipping.")
            continue

        before_count = len(all_feature_vectors)
        pitch = np.asarray(pad_data["pitch"], dtype=float)
        spacecraft_potential = None
        if apply_spacecraft_potential and lpw_data is not None:
            spacecraft_potential = interpolate_spacecraft_potential(lpw_data, times[selected])
        for selected_position, time_index in enumerate(selected):
            phi_sc = None
            if apply_spacecraft_potential:
                if spacecraft_potential is None:
                    skipped_no_spacecraft_potential += 1
                    continue
                phi_sc = float(spacecraft_potential[selected_position])
                if not np.isfinite(phi_sc):
                    skipped_no_spacecraft_potential += 1
                    continue

            directional_fluxes = extract_directional_fluxes(
                flux_at_time=np.asarray(flux[time_index], dtype=float),
                pitch=pitch,
                time_index=int(time_index),
                parallel_pitch_max_deg=parallel_pitch_max_deg,
                anti_parallel_pitch_min_deg=anti_parallel_pitch_min_deg,
            )
            parallel_flux = directional_fluxes["parallel"]
            anti_flux = directional_fluxes["anti_parallel"]
            if apply_spacecraft_potential:
                parallel_flux = correct_flux_to_plasma_energy(
                    measured_energy_eV=energy,
                    flux=parallel_flux,
                    spacecraft_potential_V=phi_sc,
                    target_energy_eV=target_plasma_energy,
                    min_plasma_energy_eV=min_plasma_energy_eV,
                )
                anti_flux = correct_flux_to_plasma_energy(
                    measured_energy_eV=energy,
                    flux=anti_flux,
                    spacecraft_potential_V=phi_sc,
                    target_energy_eV=target_plasma_energy,
                    min_plasma_energy_eV=min_plasma_energy_eV,
                )
            if direction == "both":
                if (
                    valid_flux_fraction(parallel_flux) < min_direction_valid_fraction
                    or valid_flux_fraction(anti_flux) < min_direction_valid_fraction
                ):
                    skipped_sparse += 1
                    continue
            elif valid_flux_fraction(directional_fluxes[direction]) < min_direction_valid_fraction:
                skipped_sparse += 1
                continue

            derivative_input_energy = target_plasma_energy if apply_spacecraft_potential else energy
            sorted_energy, parallel_derivative = flux_derivative(derivative_input_energy, parallel_flux)
            _, anti_derivative = flux_derivative(derivative_input_energy, anti_flux)
            if not apply_spacecraft_potential:
                if derivative_energy is None:
                    derivative_energy = sorted_energy
                elif derivative_energy.shape != sorted_energy.shape or not np.allclose(derivative_energy, sorted_energy):
                    raise ValueError(f"Energy grid in {path} does not match the first SWE file.")

            if direction == "both":
                feature_vector = np.concatenate([parallel_derivative, anti_derivative])
            elif direction == "parallel":
                feature_vector = parallel_derivative
            else:
                feature_vector = anti_derivative

            all_times.append(float(times[time_index]))
            all_parallel_derivatives.append(parallel_derivative)
            all_anti_derivatives.append(anti_derivative)
            all_feature_vectors.append(feature_vector)
            all_files.append(str(path))

        added = len(all_feature_vectors) - before_count
        log_step(f"Selected {selected.size} timestamp(s), added {added} derivative feature vector(s).")

    if derivative_energy is None or not all_feature_vectors:
        raise ValueError("No derivative spectra were found for the requested interval.")
    if skipped_no_lpw:
        log_step(f"Skipped {skipped_no_lpw} SWE file(s) because no usable LPW mrgscpot file was available for that date.")
    if skipped_no_spacecraft_potential:
        log_step(f"Skipped {skipped_no_spacecraft_potential} timestamp(s) because spacecraft potential could not be interpolated.")
    if skipped_sparse:
        log_step(f"Skipped {skipped_sparse} timestamp(s) because directional spectra were too sparse.")

    raw_features = np.asarray(all_feature_vectors, dtype=float)
    normalized = normalize_derivative_features(raw_features, normalization)
    samples = [
        SpectrumSample(
            time_unix=time_unix,
            source_file=source_file,
            parallel_flux=np.asarray(all_parallel_derivatives[index], dtype=float),
            anti_parallel_flux=np.asarray(all_anti_derivatives[index], dtype=float),
            normalized_flux=normalized[index],
        )
        for index, (time_unix, source_file) in enumerate(zip(all_times, all_files))
    ]
    return derivative_energy, samples


def plot_derivative_cluster_spectra(
    energy: np.ndarray,
    samples: list[SpectrumSample],
    labels: np.ndarray,
    reps: list[int],
    output: Path,
    direction: str,
) -> None:
    log_step(f"Writing characteristic derivative spectra plot: {output}")
    cluster_count = len(reps)
    cols = min(3, cluster_count)
    rows = int(np.ceil(cluster_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False)

    for cluster_index, sample_index in enumerate(reps):
        ax = axes[cluster_index // cols][cluster_index % cols]
        members = np.flatnonzero(labels == cluster_index)
        if direction == "parallel":
            member_values = np.asarray([samples[index].parallel_flux for index in members], dtype=float)
            ax.semilogx(energy, np.nanmedian(member_values, axis=0), color="#1f77b4", label="parallel median")
            ax.semilogx(energy, samples[sample_index].parallel_flux, color="#d62728", alpha=0.8, label="nearest time")
        elif direction == "anti_parallel":
            member_values = np.asarray([samples[index].anti_parallel_flux for index in members], dtype=float)
            ax.semilogx(energy, np.nanmedian(member_values, axis=0), color="#ff7f0e", label="anti-parallel median")
            ax.semilogx(energy, samples[sample_index].anti_parallel_flux, color="#d62728", alpha=0.8, label="nearest time")
        else:
            parallel_values = np.asarray([samples[index].parallel_flux for index in members], dtype=float)
            anti_values = np.asarray([samples[index].anti_parallel_flux for index in members], dtype=float)
            ax.semilogx(energy, np.nanmedian(parallel_values, axis=0), color="#1f77b4", label="parallel median")
            ax.semilogx(energy, np.nanmedian(anti_values, axis=0), color="#ff7f0e", label="anti-parallel median")
            ax.semilogx(energy, samples[sample_index].parallel_flux, color="#1f77b4", linestyle="--", alpha=0.5)
            ax.semilogx(energy, samples[sample_index].anti_parallel_flux, color="#ff7f0e", linestyle="--", alpha=0.5)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_title(f"Cluster {cluster_index + 1}: n={members.size}")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("dFlux/dE")
        ax.grid(True, which="both", linestyle="--", alpha=0.25)
        ax.legend(fontsize=8)

    for extra in range(cluster_count, rows * cols):
        axes[extra // cols][extra % cols].axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_pca(scores: np.ndarray, labels: np.ndarray, reps: list[int], output: Path) -> None:
    log_step(f"Writing PCA cluster plot: {output}")
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=labels, s=12, cmap="tab10", alpha=0.75)
    ax.scatter(scores[reps, 0], scores[reps, 1], marker="x", s=90, color="black", linewidths=1.8)
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def representative_rows(matrix: np.ndarray, samples: list[SpectrumSample], labels: np.ndarray, centroids: np.ndarray, reps: list[int]) -> list[dict]:
    rows = []
    for cluster_index, sample_index in enumerate(reps):
        members = np.flatnonzero(labels == cluster_index)
        rows.append(
            {
                "cluster": cluster_index + 1,
                "sample_count": int(members.size),
                "representative_time_utc": format_unix_time(samples[sample_index].time_unix),
                "parallel_valid_fraction": valid_flux_fraction(samples[sample_index].parallel_flux),
                "anti_parallel_valid_fraction": valid_flux_fraction(samples[sample_index].anti_parallel_flux),
                "representative_source_file": samples[sample_index].source_file,
                "distance_to_cluster_center": float(np.linalg.norm(matrix[sample_index] - centroids[cluster_index])),
            }
        )
    return rows


def write_representatives_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "cluster",
        "sample_count",
        "representative_time_utc",
        "parallel_valid_fraction",
        "anti_parallel_valid_fraction",
        "representative_source_file",
        "distance_to_cluster_center",
    ]
    log_step(f"Writing representative time table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_result_bundle(
    output_dir: Path,
    energy: np.ndarray,
    matrix: np.ndarray,
    samples: list[SpectrumSample],
    labels: np.ndarray,
    centroids: np.ndarray,
    direction: str,
    summary_extra: dict | None = None,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores, components, explained = pca_scores(matrix, n_components=2)
    reps = representative_indices(matrix, labels, centroids)
    rows = representative_rows(matrix, samples, labels, centroids, reps)
    plot_derivative_cluster_spectra(energy, samples, labels, reps, output_dir / "characteristic_derivative_spectra.png", direction)
    plot_pca(scores, labels, reps, output_dir / "pca_clusters.png")
    write_representatives_csv(output_dir / "representative_times.csv", rows)
    summary = {
        "sample_count": len(samples),
        "cluster_count": int(centroids.shape[0]),
        "cluster_sizes": [int(np.count_nonzero(labels == index)) for index in range(centroids.shape[0])],
        "pca_explained_variance_ratio": explained,
        "pca_components": components,
        "representatives": rows,
        "outputs": {
            "representative_times_csv": str(output_dir / "representative_times.csv"),
            "characteristic_derivative_spectra_png": str(output_dir / "characteristic_derivative_spectra.png"),
            "pca_clusters_png": str(output_dir / "pca_clusters.png"),
        },
        **(summary_extra or {}),
    }
    (output_dir / "cluster_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster MAVEN SWE energy-derivative spectra.")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--swe-file", action="append")
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--direction", choices=("both", "parallel", "anti_parallel"), default="both")
    parser.add_argument("--parallel-pitch-max", type=float, default=30.0)
    parser.add_argument("--anti-parallel-pitch-min", type=float, default=150.0)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--auto-clusters", action="store_true")
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.01)
    parser.add_argument("--no-save-candidates", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-direction-valid-fraction", type=float, default=0.1)
    parser.add_argument("--normalization", choices=("zscore", "global_zscore", "minmax", "l2", "none"), default="zscore")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    start_dt = parse_iso_datetime(args.start)
    end_dt = parse_iso_datetime(args.end)
    start_unix = parse_iso_time(args.start)
    end_unix = parse_iso_time(args.end)
    if start_unix is not None and end_unix is not None and end_unix < start_unix:
        raise ValueError("--end must be later than --start.")

    data_root = Path(args.data_root).expanduser().resolve()
    data_coverage_report = None
    if args.swe_file:
        log_step("Explicit --swe-file argument was supplied; skipping automatic data coverage check.")
        files = [Path(item).expanduser().resolve() for item in args.swe_file]
    else:
        files, data_coverage_report = ensure_required_swe_data(
            data_root=data_root,
            start_dt=start_dt,
            end_dt=end_dt,
            auto_download=not args.no_auto_download,
            fail_on_missing=not args.check_data_only,
        )
        if not files:
            files = infer_swe_files(data_root, start_unix, end_unix)

    if args.check_data_only:
        print(json.dumps(sanitize_for_json(data_coverage_report), indent=2, ensure_ascii=False))
        log_step("Data coverage check finished; exiting because --check-data-only was set.")
        return
    if not files:
        raise FileNotFoundError(f"No SWE svypad files were found under {data_root}.")

    run_name = f"derivative_{build_run_name(args, start_dt, end_dt)}"
    output_dir = unique_output_dir(Path(args.output_root).expanduser().resolve(), run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Output directory ready: {output_dir}")

    energy, samples = load_derivative_samples(
        files=files,
        start_unix=start_unix,
        end_unix=end_unix,
        stride=args.stride,
        normalization=args.normalization,
        direction=args.direction,
        parallel_pitch_max_deg=args.parallel_pitch_max,
        anti_parallel_pitch_min_deg=args.anti_parallel_pitch_min,
        min_direction_valid_fraction=args.min_direction_valid_fraction,
    )
    matrix = np.asarray([sample.normalized_flux for sample in samples], dtype=float)
    log_step(f"Derivative feature matrix ready: {matrix.shape[0]} sample(s) x {matrix.shape[1]} feature(s).")

    cluster_selection = None
    if args.auto_clusters:
        selected_clusters, labels, centroids, trials = choose_cluster_count(
            matrix,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            min_cluster_fraction=args.min_cluster_fraction,
        )
        cluster_selection = {
            "method": "davies_bouldin",
            "selected_clusters": selected_clusters,
            "min_cluster_fraction": args.min_cluster_fraction,
            "trials": trials,
        }
        print(f"Auto-selected {selected_clusters} clusters by lowest Davies-Bouldin score.")
    else:
        from machine_learning.analyze_electron_spectra_ml import kmeans

        selected_clusters = args.clusters
        log_step(f"Running k-means with manually selected k={selected_clusters}.")
        labels, centroids = kmeans(matrix, selected_clusters)

    rows = write_result_bundle(output_dir, energy, matrix, samples, labels, centroids, args.direction)
    if args.auto_clusters and cluster_selection is not None and not args.no_save_candidates:
        from machine_learning.analyze_electron_spectra_ml import kmeans

        base_dir = output_dir / "candidate_clusters"
        log_step(f"Writing candidate derivative cluster-count results under: {base_dir}")
        for trial in cluster_selection["trials"]:
            k = int(trial["clusters"])
            trial_labels, trial_centroids = kmeans(matrix, k)
            write_result_bundle(
                base_dir / f"k{k}",
                energy,
                matrix,
                samples,
                trial_labels,
                trial_centroids,
                args.direction,
                summary_extra={"selection_trial": trial},
            )

    summary = {
        "settings": {
            "method": "kmeans_on_energy_derivative_spectra",
            "start": args.start,
            "end": args.end,
            "clusters": selected_clusters,
            "requested_clusters": args.clusters,
            "auto_clusters": args.auto_clusters,
            "min_clusters": args.min_clusters,
            "max_clusters": args.max_clusters,
            "min_cluster_fraction": args.min_cluster_fraction,
            "stride": args.stride,
            "normalization": args.normalization,
            "direction": args.direction,
            "feature_meaning": (
                "paired d(parallel flux)/dE + d(anti_parallel flux)/dE per timestamp"
                if args.direction == "both"
                else f"d({args.direction} flux)/dE per timestamp"
            ),
            "parallel_pitch_max_deg": args.parallel_pitch_max,
            "anti_parallel_pitch_min_deg": args.anti_parallel_pitch_min,
            "min_direction_valid_fraction": args.min_direction_valid_fraction,
            "data_root": str(data_root),
            "input_files": [str(path) for path in files],
            "run_name": run_name,
            "output_dir": str(output_dir),
        },
        "data_coverage": data_coverage_report,
        "cluster_selection": cluster_selection,
        "sample_count": len(samples),
        "energy_eV": energy,
        "representatives": rows,
        "outputs": {
            "representative_times_csv": str(output_dir / "representative_times.csv"),
            "characteristic_derivative_spectra_png": str(output_dir / "characteristic_derivative_spectra.png"),
            "pca_clusters_png": str(output_dir / "pca_clusters.png"),
        },
    }
    (output_dir / "derivative_ml_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_step(f"Summary JSON written: {output_dir / 'derivative_ml_summary.json'}")
    print(f"Loaded {len(samples)} derivative spectra from {len(files)} SWE file(s).")
    print(f"Derivative ML output written to: {output_dir}")
    for row in rows:
        print(f"Cluster {row['cluster']}: n={row['sample_count']}, representative={row['representative_time_utc']}")


if __name__ == "__main__":
    main()
