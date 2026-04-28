from __future__ import annotations
"""
Machine-learning exploration of MAVEN SWE electron spectra.

Workflow:
1. Load one or more local SWE svypad CDF files.
2. Check whether local SWE files cover the requested interval, downloading missing days if needed.
3. Split each timestamp into parallel and anti-parallel spectra.
4. Preprocess with log scaling and per-spectrum normalization.
5. Cluster the normalized spectra to find characteristic spectral shapes.
6. Report the nearest real timestamp for each characteristic spectrum.

The implementation uses only NumPy so it can run in the current repository
without adding a scikit-learn dependency.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from download_maven_data import PIPELINE_PRODUCTS, build_session, download_product_for_day, parse_filename
from process_maven_spectra import format_unix_time, load_pad_data


ML_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ML_ROOT / "data" / "maven"
DEFAULT_OUTPUT_ROOT = ML_ROOT / "outputs" / "analysis"
SWE_PRODUCTS = tuple(spec for spec in PIPELINE_PRODUCTS if spec.instrument == "swe" and spec.datatype == "svypad")


def log_step(message: str) -> None:
    print(f"[analysis] {datetime.now().isoformat(timespec='seconds')} | {message}", flush=True)


@dataclass(frozen=True)
class SpectrumSample:
    time_unix: float
    source_file: str
    parallel_flux: np.ndarray
    anti_parallel_flux: np.ndarray
    normalized_flux: np.ndarray


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def parse_iso_time(value: str | None) -> float | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_time_for_name(value: datetime | None, fallback: str) -> str:
    if value is None:
        return fallback
    return value.strftime("%Y%m%dT%H%M%S")


def build_run_name(args: argparse.Namespace, start_dt: datetime | None, end_dt: datetime | None) -> str:
    start_label = format_time_for_name(start_dt, "start-all")
    end_label = format_time_for_name(end_dt, "end-all")
    cluster_label = f"auto-k{args.min_clusters}-{args.max_clusters}" if args.auto_clusters else f"k{args.clusters}"
    return f"{start_label}_{end_label}_{args.direction}_{cluster_label}_{args.normalization}"


def unique_output_dir(output_root: Path, run_name: str) -> Path:
    candidate = output_root / run_name
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = output_root / f"{run_name}_run{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def iter_required_dates(start_dt: datetime, end_dt: datetime) -> list[date]:
    """Return daily SWE file dates needed for a time interval.

    An end time exactly at 00:00 is treated as the exclusive boundary of the
    previous day, matching common commands such as end=2024-11-08T00:00:00 for
    a Nov 1-7 analysis interval.
    """
    effective_end = end_dt
    if end_dt.time() == datetime.min.time() and end_dt.date() > start_dt.date():
        effective_end = end_dt - timedelta(microseconds=1)

    current = start_dt.date()
    last = effective_end.date()
    dates: list[date] = []
    while current <= last:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def find_local_swe_files_for_day(data_root: Path, day: date) -> list[Path]:
    day_code = day.strftime("%Y%m%d")
    matches: list[Path] = []
    for path in data_root.rglob("mvn_swe_l2_svypad_*.cdf"):
        parsed = parse_filename(path.name)
        if not parsed:
            continue
        if f"{parsed['year']}{parsed['month']}{parsed['day']}" == day_code:
            matches.append(path)
    return sorted(matches)


def inspect_swe_file(path: Path) -> dict:
    try:
        pad_data = load_pad_data(path)
        times = np.asarray(pad_data["times"], dtype=float)
        if times.size == 0:
            return {"status": "empty", "path": str(path), "time_start_utc": None, "time_end_utc": None, "samples": 0}
        return {
            "status": "ok",
            "path": str(path),
            "time_start_utc": format_unix_time(float(times[0])),
            "time_end_utc": format_unix_time(float(times[-1])),
            "samples": int(times.size),
        }
    except Exception as exc:
        return {
            "status": "read_error",
            "path": str(path),
            "time_start_utc": None,
            "time_end_utc": None,
            "samples": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def ensure_required_swe_data(
    data_root: Path,
    start_dt: datetime | None,
    end_dt: datetime | None,
    auto_download: bool,
    fail_on_missing: bool = True,
) -> tuple[list[Path], dict]:
    if start_dt is None or end_dt is None:
        log_step("No complete --start/--end interval was given; skipping date-by-date data coverage check.")
        return [], {"required_dates": [], "available": [], "missing_dates": [], "corrupt": [], "downloaded": []}

    required_dates = iter_required_dates(start_dt, end_dt)
    log_step(f"Checking local SWE data coverage for {len(required_dates)} required day(s).")
    report = {
        "required_dates": [item.isoformat() for item in required_dates],
        "available": [],
        "missing_dates": [],
        "corrupt": [],
        "quarantined": [],
        "downloaded": [],
    }

    usable_files_by_date: dict[date, Path] = {}
    for day in required_dates:
        candidates = find_local_swe_files_for_day(data_root, day)
        if not candidates:
            log_step(f"Missing local SWE svypad file for {day.isoformat()}.")
            report["missing_dates"].append(day.isoformat())
            continue

        day_has_usable_file = False
        for path in candidates:
            info = inspect_swe_file(path)
            if info["status"] == "ok":
                day_has_usable_file = True
                usable_files_by_date[day] = path
                report["available"].append({"date": day.isoformat(), **info})
                log_step(f"Available {day.isoformat()}: {path.name}, {info['samples']} time sample(s).")
                break
            report["corrupt"].append({"date": day.isoformat(), **info})
            log_step(f"Unreadable SWE file for {day.isoformat()}: {path.name} ({info.get('error', info['status'])}).")

        if not day_has_usable_file:
            report["missing_dates"].append(day.isoformat())

    missing_dates = [date.fromisoformat(item) for item in report["missing_dates"]]
    if missing_dates and not auto_download and fail_on_missing:
        missing = ", ".join(item.isoformat() for item in missing_dates)
        raise FileNotFoundError(f"Missing or unreadable SWE svypad data for: {missing}")

    if missing_dates and auto_download:
        if not SWE_PRODUCTS:
            raise RuntimeError("No SWE svypad product specification is available for auto-download.")
        log_step(f"Auto-download enabled; downloading {len(missing_dates)} missing day(s).")
        session = build_session()
        spec = SWE_PRODUCTS[0]
        for index, day in enumerate(missing_dates, start=1):
            for corrupt_info in [item for item in report["corrupt"] if item["date"] == day.isoformat()]:
                corrupt_path = Path(corrupt_info["path"])
                if corrupt_path.exists():
                    quarantine_path = corrupt_path.with_name(f"{corrupt_path.name}.read_error")
                    suffix = 1
                    while quarantine_path.exists():
                        quarantine_path = corrupt_path.with_name(f"{corrupt_path.name}.read_error.{suffix}")
                        suffix += 1
                    corrupt_path.rename(quarantine_path)
                    report["quarantined"].append(
                        {"date": day.isoformat(), "original_path": str(corrupt_path), "quarantine_path": str(quarantine_path)}
                    )
                    log_step(f"Moved unreadable file aside before re-download: {quarantine_path.name}")
            log_step(f"Downloading missing day {index}/{len(missing_dates)}: {day.isoformat()}")
            local_path = download_product_for_day(session=session, spec=spec, day=day, data_root=data_root)
            info = inspect_swe_file(local_path)
            report["downloaded"].append({"date": day.isoformat(), **info})
            if info["status"] != "ok":
                raise RuntimeError(f"Downloaded file for {day.isoformat()} is not readable: {info}")
            usable_files_by_date[day] = local_path

    selected_files = [usable_files_by_date[day] for day in required_dates if day in usable_files_by_date]
    log_step(f"Data coverage check complete: {len(selected_files)}/{len(required_dates)} required day(s) usable.")
    return selected_files, report


def infer_swe_files(data_root: Path, start_unix: float | None, end_unix: float | None) -> list[Path]:
    log_step(f"Searching SWE svypad CDF files under {data_root}.")
    candidates: list[Path] = []
    for path in data_root.rglob("mvn_swe_l2_svypad_*.cdf"):
        parsed = parse_filename(path.name)
        if not parsed:
            continue
        day = datetime(
            int(parsed["year"]),
            int(parsed["month"]),
            int(parsed["day"]),
            tzinfo=timezone.utc,
        ).timestamp()
        if start_unix is not None and day + 86400.0 < start_unix:
            continue
        if end_unix is not None and day > end_unix:
            continue
        candidates.append(path)
    selected = sorted(candidates)
    log_step(f"Found {len(selected)} candidate SWE file(s).")
    return selected


def fill_nonfinite_by_energy(spectra: np.ndarray) -> np.ndarray:
    filled = np.asarray(spectra, dtype=float).copy()
    positive = filled[np.isfinite(filled) & (filled > 0.0)]
    fallback = float(np.nanmedian(positive)) if positive.size else 1.0

    for energy_index in range(filled.shape[1]):
        column = filled[:, energy_index]
        valid = np.isfinite(column) & (column > 0.0)
        replacement = float(np.nanmedian(column[valid])) if np.any(valid) else fallback
        column[~valid] = replacement
        filled[:, energy_index] = column
    return filled


def normalize_spectra(raw_spectra: np.ndarray, method: str) -> np.ndarray:
    log_step(f"Normalizing {raw_spectra.shape[0]} feature vector(s) with method={method}.")
    filled = fill_nonfinite_by_energy(raw_spectra)
    logged = np.log10(np.clip(filled, 1e-30, None))

    if method == "log":
        return logged

    if method == "global_zscore":
        center = np.nanmean(logged, axis=0, keepdims=True)
        scale = np.nanstd(logged, axis=0, keepdims=True)
        scale[scale == 0.0] = 1.0
        return (logged - center) / scale

    if method == "zscore":
        center = np.nanmean(logged, axis=1, keepdims=True)
        scale = np.nanstd(logged, axis=1, keepdims=True)
        scale[scale == 0.0] = 1.0
        return (logged - center) / scale

    if method == "minmax":
        lower = np.nanmin(logged, axis=1, keepdims=True)
        upper = np.nanmax(logged, axis=1, keepdims=True)
        span = upper - lower
        span[span == 0.0] = 1.0
        return (logged - lower) / span

    if method == "l2":
        norm = np.linalg.norm(logged, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        return logged / norm

    raise ValueError(f"Unsupported normalization method: {method}")


def extract_directional_fluxes(
    flux_at_time: np.ndarray,
    pitch: np.ndarray,
    time_index: int,
    parallel_pitch_max_deg: float,
    anti_parallel_pitch_min_deg: float,
) -> dict[str, np.ndarray]:
    if pitch.ndim == 1:
        parallel_mask = pitch < parallel_pitch_max_deg
        anti_parallel_mask = pitch > anti_parallel_pitch_min_deg
        return {
            "parallel": np.nanmean(flux_at_time[parallel_mask, :], axis=0),
            "anti_parallel": np.nanmean(flux_at_time[anti_parallel_mask, :], axis=0),
        }

    pitch_at_time = pitch[time_index]
    parallel_flux = np.full(flux_at_time.shape[1], np.nan, dtype=float)
    anti_parallel_flux = np.full(flux_at_time.shape[1], np.nan, dtype=float)
    for energy_index in range(flux_at_time.shape[1]):
        pitch_column = pitch_at_time[:, energy_index]
        parallel_mask = pitch_column < parallel_pitch_max_deg
        anti_parallel_mask = pitch_column > anti_parallel_pitch_min_deg
        if np.any(parallel_mask):
            parallel_flux[energy_index] = np.nanmean(flux_at_time[parallel_mask, energy_index])
        if np.any(anti_parallel_mask):
            anti_parallel_flux[energy_index] = np.nanmean(flux_at_time[anti_parallel_mask, energy_index])
    return {"parallel": parallel_flux, "anti_parallel": anti_parallel_flux}


def valid_flux_fraction(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.count_nonzero(np.isfinite(array) & (array > 0.0)) / array.size)


def load_samples(
    files: list[Path],
    start_unix: float | None,
    end_unix: float | None,
    stride: int,
    normalization: str,
    direction: str,
    parallel_pitch_max_deg: float,
    anti_parallel_pitch_min_deg: float,
    min_direction_valid_fraction: float,
) -> tuple[np.ndarray, list[SpectrumSample]]:
    all_times: list[float] = []
    all_parallel_fluxes: list[np.ndarray] = []
    all_anti_parallel_fluxes: list[np.ndarray] = []
    all_feature_fluxes: list[np.ndarray] = []
    all_files: list[str] = []
    energy_reference: np.ndarray | None = None
    if direction == "both":
        log_step("Feature mode: paired parallel + anti_parallel spectra; one timestamp is one sample.")
    else:
        log_step(f"Feature mode: single-direction spectra; using {direction}.")
    log_step(f"Minimum valid positive energy-bin fraction per used direction: {min_direction_valid_fraction:g}.")
    skipped_sparse = 0

    for file_index, path in enumerate(files, start=1):
        log_step(f"Loading SWE file {file_index}/{len(files)}: {path.name}")
        pad_data = load_pad_data(path)
        times = np.asarray(pad_data["times"], dtype=float)
        flux = np.asarray(pad_data["flux"], dtype=float)
        energy = np.asarray(pad_data["energy"], dtype=float)

        if energy_reference is None:
            energy_reference = energy
        elif energy_reference.shape != energy.shape or not np.allclose(energy_reference, energy, equal_nan=True):
            raise ValueError(f"Energy grid in {path} does not match the first SWE file.")

        mask = np.ones(times.shape, dtype=bool)
        if start_unix is not None:
            mask &= times >= start_unix
        if end_unix is not None:
            mask &= times <= end_unix
        selected = np.flatnonzero(mask)[:: max(stride, 1)]
        if selected.size == 0:
            log_step(f"No timestamps selected from {path.name}; skipping.")
            continue

        before_count = len(all_feature_fluxes)
        pitch = np.asarray(pad_data["pitch"], dtype=float)
        for time_index in selected:
            directional_fluxes = extract_directional_fluxes(
                flux_at_time=np.asarray(flux[time_index], dtype=float),
                pitch=pitch,
                time_index=int(time_index),
                parallel_pitch_max_deg=parallel_pitch_max_deg,
                anti_parallel_pitch_min_deg=anti_parallel_pitch_min_deg,
            )
            parallel_flux = directional_fluxes["parallel"]
            anti_parallel_flux = directional_fluxes["anti_parallel"]
            if direction == "both":
                if (
                    valid_flux_fraction(parallel_flux) < min_direction_valid_fraction
                    or valid_flux_fraction(anti_parallel_flux) < min_direction_valid_fraction
                ):
                    skipped_sparse += 1
                    continue
                feature_flux = np.concatenate([parallel_flux, anti_parallel_flux])
            else:
                if valid_flux_fraction(directional_fluxes[direction]) < min_direction_valid_fraction:
                    skipped_sparse += 1
                    continue
                feature_flux = directional_fluxes[direction]
            if not np.any(np.isfinite(feature_flux)):
                continue
            all_times.append(float(times[time_index]))
            all_parallel_fluxes.append(parallel_flux)
            all_anti_parallel_fluxes.append(anti_parallel_flux)
            all_feature_fluxes.append(feature_flux)
            all_files.append(str(path))
        added = len(all_feature_fluxes) - before_count
        log_step(f"Selected {selected.size} timestamp(s), added {added} time-sample feature vector(s).")

    if energy_reference is None or not all_feature_fluxes:
        raise ValueError("No SWE spectra were found for the requested interval.")
    if skipped_sparse:
        log_step(f"Skipped {skipped_sparse} timestamp(s) because directional spectra were too sparse.")

    raw = np.asarray(all_feature_fluxes, dtype=float)
    normalized = normalize_spectra(raw, normalization)
    samples = [
        SpectrumSample(
            time_unix=float(time_unix),
            source_file=source_file,
            parallel_flux=np.asarray(all_parallel_fluxes[index], dtype=float),
            anti_parallel_flux=np.asarray(all_anti_parallel_fluxes[index], dtype=float),
            normalized_flux=normalized[index],
        )
        for index, (time_unix, source_file) in enumerate(zip(all_times, all_files))
    ]
    return energy_reference, samples


def initialize_centroids(matrix: np.ndarray, k: int) -> np.ndarray:
    if matrix.shape[0] < k:
        raise ValueError(f"Need at least {k} spectra, but only {matrix.shape[0]} were loaded.")

    centroids = [matrix[0]]
    distances = np.full(matrix.shape[0], np.inf, dtype=float)
    for _ in range(1, k):
        latest = centroids[-1]
        distances = np.minimum(distances, np.sum((matrix - latest) ** 2, axis=1))
        centroids.append(matrix[int(np.argmax(distances))])
    return np.asarray(centroids, dtype=float)


def kmeans(matrix: np.ndarray, k: int, max_iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    centroids = initialize_centroids(matrix, k)
    labels = np.full(matrix.shape[0], -1, dtype=int)

    for _ in range(max_iterations):
        distances = np.sum((matrix[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for cluster_index in range(k):
            members = matrix[labels == cluster_index]
            if members.size:
                centroids[cluster_index] = np.nanmean(members, axis=0)
    return labels, centroids


def davies_bouldin_score(matrix: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Return a Davies-Bouldin score; lower values indicate cleaner clusters."""
    cluster_count = centroids.shape[0]
    scatters = np.zeros(cluster_count, dtype=float)

    for cluster_index in range(cluster_count):
        members = matrix[labels == cluster_index]
        if members.size == 0:
            return float("inf")
        distances = np.linalg.norm(members - centroids[cluster_index], axis=1)
        scatters[cluster_index] = float(np.nanmean(distances))

    centroid_distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    np.fill_diagonal(centroid_distances, np.inf)
    ratios = (scatters[:, None] + scatters[None, :]) / centroid_distances
    return float(np.nanmean(np.nanmax(ratios, axis=1)))


def choose_cluster_count(
    matrix: np.ndarray,
    min_clusters: int,
    max_clusters: int,
    min_cluster_fraction: float,
) -> tuple[int, np.ndarray, np.ndarray, list[dict]]:
    upper = min(max_clusters, matrix.shape[0])
    lower = max(2, min_clusters)
    if upper < lower:
        raise ValueError(f"Cannot auto-select clusters from {lower} to {upper}; not enough spectra were loaded.")

    trials: list[dict] = []
    best_score = float("inf")
    best_k = lower
    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None

    for k in range(lower, upper + 1):
        log_step(f"Auto-cluster trial: k={k}.")
        labels, centroids = kmeans(matrix, k)
        score = davies_bouldin_score(matrix, labels, centroids)
        cluster_sizes = [int(np.count_nonzero(labels == cluster_index)) for cluster_index in range(k)]
        min_allowed_size = max(1, int(np.ceil(matrix.shape[0] * min_cluster_fraction)))
        has_small_cluster = min(cluster_sizes) < min_allowed_size
        effective_score = float("inf") if has_small_cluster else score
        trials.append(
            {
                "clusters": k,
                "davies_bouldin_score": score,
                "effective_score": effective_score,
                "cluster_sizes": cluster_sizes,
                "min_allowed_cluster_size": min_allowed_size,
                "rejected_for_small_cluster": has_small_cluster,
            }
        )
        status = "rejected small cluster" if has_small_cluster else "accepted"
        log_step(f"k={k}: DB={score:.4g}, min cluster size={min(cluster_sizes)}, {status}.")
        if effective_score < best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_centroids = centroids

    if best_labels is None or best_centroids is None:
        raise ValueError("Auto-cluster selection did not produce a valid clustering. Try lowering --min-cluster-fraction.")
    return best_k, best_labels, best_centroids, trials


def pca_scores(matrix: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = matrix - np.nanmean(matrix, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    scores = centered @ components.T
    variance = singular_values**2 / max(matrix.shape[0] - 1, 1)
    explained = variance[:n_components] / np.sum(variance) if np.sum(variance) > 0 else np.zeros(n_components)
    return scores, components, explained


def representative_indices(matrix: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> list[int]:
    representatives: list[int] = []
    for cluster_index in range(centroids.shape[0]):
        member_indices = np.flatnonzero(labels == cluster_index)
        distances = np.sum((matrix[member_indices] - centroids[cluster_index]) ** 2, axis=1)
        representatives.append(int(member_indices[int(np.argmin(distances))]))
    return representatives


def plot_cluster_spectra(
    energy: np.ndarray,
    samples: list[SpectrumSample],
    labels: np.ndarray,
    reps: list[int],
    output: Path,
    direction: str,
) -> None:
    log_step(f"Writing characteristic spectra plot: {output}")
    cluster_count = len(reps)
    cols = min(3, cluster_count)
    rows = int(np.ceil(cluster_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False)

    for cluster_index, sample_index in enumerate(reps):
        ax = axes[cluster_index // cols][cluster_index % cols]
        members = np.flatnonzero(labels == cluster_index)
        if direction == "parallel":
            member_flux = np.asarray([samples[index].parallel_flux for index in members], dtype=float)
            center_flux = np.nanmedian(fill_nonfinite_by_energy(member_flux), axis=0)
            representative_flux = samples[sample_index].parallel_flux
            ax.loglog(energy, center_flux, color="#1f77b4", linewidth=1.6, label="parallel median")
            ax.loglog(energy, representative_flux, color="#d62728", linewidth=1.1, alpha=0.8, label="nearest time")
        elif direction == "anti_parallel":
            member_flux = np.asarray([samples[index].anti_parallel_flux for index in members], dtype=float)
            center_flux = np.nanmedian(fill_nonfinite_by_energy(member_flux), axis=0)
            representative_flux = samples[sample_index].anti_parallel_flux
            ax.loglog(energy, center_flux, color="#1f77b4", linewidth=1.6, label="anti-parallel median")
            ax.loglog(energy, representative_flux, color="#d62728", linewidth=1.1, alpha=0.8, label="nearest time")
        else:
            parallel_member_flux = np.asarray([samples[index].parallel_flux for index in members], dtype=float)
            anti_member_flux = np.asarray([samples[index].anti_parallel_flux for index in members], dtype=float)
            parallel_center = np.nanmedian(fill_nonfinite_by_energy(parallel_member_flux), axis=0)
            anti_center = np.nanmedian(fill_nonfinite_by_energy(anti_member_flux), axis=0)
            ax.loglog(energy, parallel_center, color="#1f77b4", linewidth=1.6, label="parallel median")
            ax.loglog(energy, anti_center, color="#ff7f0e", linewidth=1.6, label="anti-parallel median")
            ax.loglog(
                energy,
                samples[sample_index].parallel_flux,
                color="#1f77b4",
                linewidth=0.9,
                alpha=0.5,
                linestyle="--",
                label="nearest parallel",
            )
            ax.loglog(
                energy,
                samples[sample_index].anti_parallel_flux,
                color="#ff7f0e",
                linewidth=0.9,
                alpha=0.5,
                linestyle="--",
                label="nearest anti-parallel",
            )
        ax.set_title(f"Cluster {cluster_index + 1}: n={members.size}")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Flux")
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


def write_representatives_csv(path: Path, rows: list[dict]) -> None:
    log_step(f"Writing representative time table: {path}")
    fieldnames = [
        "cluster",
        "sample_count",
        "representative_time_utc",
        "parallel_valid_fraction",
        "anti_parallel_valid_fraction",
        "representative_source_file",
        "distance_to_cluster_center",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_representative_rows(
    matrix: np.ndarray,
    samples: list[SpectrumSample],
    labels: np.ndarray,
    centroids: np.ndarray,
    reps: list[int],
) -> list[dict]:
    rows = []
    for cluster_index, sample_index in enumerate(reps):
        members = np.flatnonzero(labels == cluster_index)
        distance = float(np.linalg.norm(matrix[sample_index] - centroids[cluster_index]))
        rows.append(
            {
                "cluster": cluster_index + 1,
                "sample_count": int(members.size),
                "representative_time_utc": format_unix_time(samples[sample_index].time_unix),
                "parallel_valid_fraction": valid_flux_fraction(samples[sample_index].parallel_flux),
                "anti_parallel_valid_fraction": valid_flux_fraction(samples[sample_index].anti_parallel_flux),
                "representative_source_file": samples[sample_index].source_file,
                "distance_to_cluster_center": distance,
            }
        )
    return rows


def write_cluster_result_bundle(
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
    representative_rows = build_representative_rows(matrix, samples, labels, centroids, reps)
    plot_cluster_spectra(energy, samples, labels, reps, output_dir / "characteristic_spectra.png", direction)
    plot_pca(scores, labels, reps, output_dir / "pca_clusters.png")
    write_representatives_csv(output_dir / "representative_times.csv", representative_rows)
    summary = {
        "sample_count": len(samples),
        "cluster_count": int(centroids.shape[0]),
        "cluster_sizes": [int(np.count_nonzero(labels == index)) for index in range(centroids.shape[0])],
        "pca_explained_variance_ratio": explained,
        "pca_components": components,
        "representatives": representative_rows,
        "outputs": {
            "representative_times_csv": str(output_dir / "representative_times.csv"),
            "characteristic_spectra_png": str(output_dir / "characteristic_spectra.png"),
            "pca_clusters_png": str(output_dir / "pca_clusters.png"),
        },
        **(summary_extra or {}),
    }
    (output_dir / "cluster_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return representative_rows


def write_candidate_cluster_results(
    base_dir: Path,
    trials: list[dict],
    energy: np.ndarray,
    matrix: np.ndarray,
    samples: list[SpectrumSample],
    direction: str,
) -> None:
    log_step(f"Writing candidate cluster-count results under: {base_dir}")
    for trial in trials:
        cluster_count = int(trial["clusters"])
        labels, centroids = kmeans(matrix, cluster_count)
        write_cluster_result_bundle(
            output_dir=base_dir / f"k{cluster_count}",
            energy=energy,
            matrix=matrix,
            samples=samples,
            labels=labels,
            centroids=centroids,
            direction=direction,
            summary_extra={"selection_trial": trial},
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and cluster MAVEN SWE electron spectra.")
    parser.add_argument("--start", help="Optional UTC start time, for example 2024-11-07T00:00:00.")
    parser.add_argument("--end", help="Optional UTC end time, for example 2024-11-08T00:00:00.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for downloaded MAVEN data.")
    parser.add_argument("--swe-file", action="append", help="Explicit SWE svypad CDF file. Can be repeated.")
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Only check local data coverage; fail instead of downloading missing SWE files.",
    )
    parser.add_argument(
        "--check-data-only",
        action="store_true",
        help="Check/download required SWE files, print the coverage report, then exit before ML analysis.",
    )
    parser.add_argument(
        "--direction",
        choices=("both", "parallel", "anti_parallel"),
        default="both",
        help="Use paired parallel+anti_parallel spectra per timestamp, or only one direction.",
    )
    parser.add_argument(
        "--parallel-pitch-max",
        type=float,
        default=30.0,
        help="Parallel spectrum uses pitch angles below this value in degrees.",
    )
    parser.add_argument(
        "--anti-parallel-pitch-min",
        type=float,
        default=150.0,
        help="Anti-parallel spectrum uses pitch angles above this value in degrees.",
    )
    parser.add_argument("--clusters", type=int, default=4, help="Number of characteristic spectral groups.")
    parser.add_argument(
        "--auto-clusters",
        action="store_true",
        help="Automatically choose the number of clusters using the Davies-Bouldin score.",
    )
    parser.add_argument("--min-clusters", type=int, default=2, help="Smallest cluster count tried by --auto-clusters.")
    parser.add_argument("--max-clusters", type=int, default=10, help="Largest cluster count tried by --auto-clusters.")
    parser.add_argument(
        "--min-cluster-fraction",
        type=float,
        default=0.01,
        help="Reject auto-cluster choices where any cluster has less than this fraction of all spectra.",
    )
    parser.add_argument(
        "--no-save-candidates",
        action="store_true",
        help="When using --auto-clusters, do not write plots/tables for every tried cluster count.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth timestamp to reduce runtime.")
    parser.add_argument(
        "--min-direction-valid-fraction",
        type=float,
        default=0.1,
        help="Discard samples where a used direction has less than this fraction of positive finite energy bins.",
    )
    parser.add_argument(
        "--normalization",
        choices=("log", "global_zscore", "zscore", "minmax", "l2"),
        default="log",
        help="Preprocessing after missing-value filling; log preserves parallel/anti-parallel flux differences.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for ML outputs.")
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
    log_step(f"Using {len(files)} SWE file(s) for analysis.")

    output_root = Path(args.output_root).expanduser().resolve()
    run_name = build_run_name(args, start_dt, end_dt)
    output_dir = unique_output_dir(output_root, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Output directory ready: {output_dir}")

    energy, samples = load_samples(
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
    log_step(f"Feature matrix ready: {matrix.shape[0]} sample(s) x {matrix.shape[1]} energy bin(s).")
    cluster_selection = None
    if args.auto_clusters:
        selected_clusters, labels, centroids, cluster_trials = choose_cluster_count(
            matrix,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            min_cluster_fraction=args.min_cluster_fraction,
        )
        cluster_selection = {
            "method": "davies_bouldin",
            "selected_clusters": selected_clusters,
            "min_cluster_fraction": args.min_cluster_fraction,
            "trials": cluster_trials,
        }
        print(f"Auto-selected {selected_clusters} clusters by lowest Davies-Bouldin score.")
    else:
        selected_clusters = args.clusters
        log_step(f"Running k-means with manually selected k={selected_clusters}.")
        labels, centroids = kmeans(matrix, selected_clusters)

    log_step("Writing best cluster-count outputs.")
    representative_rows = write_cluster_result_bundle(
        output_dir=output_dir,
        energy=energy,
        matrix=matrix,
        samples=samples,
        labels=labels,
        centroids=centroids,
        direction=args.direction,
    )
    scores, components, explained = pca_scores(matrix, n_components=2)
    if args.auto_clusters and cluster_selection is not None and not args.no_save_candidates:
        write_candidate_cluster_results(
            base_dir=output_dir / "candidate_clusters",
            trials=cluster_selection["trials"],
            energy=energy,
            matrix=matrix,
            samples=samples,
            direction=args.direction,
        )

    summary = {
        "settings": {
            "start": args.start,
            "end": args.end,
            "clusters": selected_clusters,
            "requested_clusters": args.clusters,
            "auto_clusters": args.auto_clusters,
            "min_clusters": args.min_clusters,
            "max_clusters": args.max_clusters,
            "min_cluster_fraction": args.min_cluster_fraction,
            "save_candidate_cluster_results": args.auto_clusters and not args.no_save_candidates,
            "stride": args.stride,
            "normalization": args.normalization,
            "direction": args.direction,
            "feature_meaning": (
                "paired parallel + anti_parallel spectra per timestamp"
                if args.direction == "both"
                else f"{args.direction} spectrum per timestamp"
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
        "pca_explained_variance_ratio": explained,
        "pca_components": components,
        "representatives": representative_rows,
        "outputs": {
            "representative_times_csv": str(output_dir / "representative_times.csv"),
            "characteristic_spectra_png": str(output_dir / "characteristic_spectra.png"),
            "pca_clusters_png": str(output_dir / "pca_clusters.png"),
        },
    }
    (output_dir / "ml_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_step(f"Summary JSON written: {output_dir / 'ml_summary.json'}")
    print(f"Loaded {len(samples)} spectra from {len(files)} SWE file(s).")
    print(f"ML output written to: {output_dir}")
    for row in representative_rows:
        print(
            f"Cluster {row['cluster']}: n={row['sample_count']}, "
            f"representative={row['representative_time_utc']}"
        )


if __name__ == "__main__":
    main()
