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
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process_maven_spectra import format_unix_time, load_pad_data
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
) -> tuple[np.ndarray, list[SpectrumSample]]:
    all_times: list[float] = []
    all_parallel_derivatives: list[np.ndarray] = []
    all_anti_derivatives: list[np.ndarray] = []
    all_feature_vectors: list[np.ndarray] = []
    all_files: list[str] = []
    derivative_energy: np.ndarray | None = None
    skipped_sparse = 0

    if direction == "both":
        log_step("Derivative feature mode: paired dF/dE parallel + anti_parallel; one timestamp is one sample.")
    else:
        log_step(f"Derivative feature mode: single-direction dF/dE; using {direction}.")
    log_step(f"Minimum valid positive energy-bin fraction per used direction: {min_direction_valid_fraction:g}.")

    for file_index, path in enumerate(files, start=1):
        log_step(f"Loading SWE file {file_index}/{len(files)}: {path.name}")
        pad_data = load_pad_data(path)
        times = np.asarray(pad_data["times"], dtype=float)
        flux = np.asarray(pad_data["flux"], dtype=float)
        energy = np.asarray(pad_data["energy"], dtype=float)

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
        for time_index in selected:
            directional_fluxes = extract_directional_fluxes(
                flux_at_time=np.asarray(flux[time_index], dtype=float),
                pitch=pitch,
                time_index=int(time_index),
                parallel_pitch_max_deg=parallel_pitch_max_deg,
                anti_parallel_pitch_min_deg=anti_parallel_pitch_min_deg,
            )
            parallel_flux = directional_fluxes["parallel"]
            anti_flux = directional_fluxes["anti_parallel"]
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

            sorted_energy, parallel_derivative = flux_derivative(energy, parallel_flux)
            _, anti_derivative = flux_derivative(energy, anti_flux)
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
