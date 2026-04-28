from __future__ import annotations
"""
Gaussian Mixture Model analysis for MAVEN SWE electron spectra.

This script shares the same data handling and paired parallel/anti-parallel
feature construction as `analyze_electron_spectra_ml.py`, but replaces k-means
with a diagonal-covariance Gaussian Mixture Model trained by EM.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process_maven_spectra import format_unix_time
from machine_learning.analyze_electron_spectra_ml import (
    DEFAULT_DATA_ROOT,
    ML_ROOT,
    build_run_name,
    ensure_required_swe_data,
    infer_swe_files,
    initialize_centroids,
    load_samples,
    log_step,
    parse_iso_datetime,
    parse_iso_time,
    pca_scores,
    plot_cluster_spectra,
    plot_pca,
    representative_indices,
    sanitize_for_json,
    unique_output_dir,
    write_cluster_result_bundle,
    write_representatives_csv,
)


DEFAULT_OUTPUT_ROOT = ML_ROOT / "outputs" / "gmm_analysis"


def logsumexp(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    maximum = np.max(matrix, axis=axis, keepdims=True)
    stabilized = np.exp(matrix - maximum)
    return (maximum + np.log(np.sum(stabilized, axis=axis, keepdims=True))).squeeze(axis)


def estimate_log_gaussian_probability(matrix: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    dimension = matrix.shape[1]
    log_det = np.sum(np.log(variances), axis=1)
    squared = ((matrix[:, None, :] - means[None, :, :]) ** 2) / variances[None, :, :]
    return -0.5 * (dimension * np.log(2.0 * np.pi) + log_det[None, :] + np.sum(squared, axis=2))


def fit_gmm_diagonal(
    matrix: np.ndarray,
    component_count: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    regularization: float = 1e-6,
) -> dict:
    if matrix.shape[0] < component_count:
        raise ValueError(f"Need at least {component_count} spectra, but only {matrix.shape[0]} were loaded.")

    means = initialize_centroids(matrix, component_count)
    global_variance = np.var(matrix, axis=0) + regularization
    variances = np.tile(global_variance, (component_count, 1))
    weights = np.full(component_count, 1.0 / component_count, dtype=float)
    previous_log_likelihood = -np.inf

    for iteration in range(1, max_iterations + 1):
        weighted_log_prob = estimate_log_gaussian_probability(matrix, means, variances) + np.log(weights[None, :])
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_likelihood = float(np.sum(log_prob_norm))
        responsibilities = np.exp(weighted_log_prob - log_prob_norm[:, None])

        if abs(log_likelihood - previous_log_likelihood) < tolerance:
            break
        previous_log_likelihood = log_likelihood

        effective_counts = np.sum(responsibilities, axis=0) + 10.0 * np.finfo(float).eps
        weights = effective_counts / matrix.shape[0]
        means = (responsibilities.T @ matrix) / effective_counts[:, None]
        for component_index in range(component_count):
            diff = matrix - means[component_index]
            variances[component_index] = (responsibilities[:, component_index][:, None] * diff**2).sum(axis=0)
            variances[component_index] /= effective_counts[component_index]
        variances = np.maximum(variances, regularization)

    weighted_log_prob = estimate_log_gaussian_probability(matrix, means, variances) + np.log(weights[None, :])
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    responsibilities = np.exp(weighted_log_prob - log_prob_norm[:, None])
    labels = np.argmax(responsibilities, axis=1)
    log_likelihood = float(np.sum(log_prob_norm))
    parameter_count = component_count * (2 * matrix.shape[1]) + (component_count - 1)
    bic = -2.0 * log_likelihood + parameter_count * np.log(matrix.shape[0])
    return {
        "labels": labels,
        "means": means,
        "variances": variances,
        "weights": weights,
        "responsibilities": responsibilities,
        "log_likelihood": log_likelihood,
        "bic": float(bic),
        "iterations": iteration,
    }


def choose_gmm_component_count(
    matrix: np.ndarray,
    min_components: int,
    max_components: int,
    min_cluster_fraction: float,
    max_iterations: int,
    tolerance: float,
    regularization: float,
) -> tuple[int, dict, list[dict]]:
    upper = min(max_components, matrix.shape[0])
    lower = max(1, min_components)
    if upper < lower:
        raise ValueError(f"Cannot auto-select GMM components from {lower} to {upper}; not enough spectra were loaded.")

    best_model: dict | None = None
    best_components = lower
    best_effective_bic = float("inf")
    trials: list[dict] = []

    for component_count in range(lower, upper + 1):
        log_step(f"GMM auto-cluster trial: k={component_count}.")
        model = fit_gmm_diagonal(
            matrix,
            component_count=component_count,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
        )
        labels = model["labels"]
        cluster_sizes = [int(np.count_nonzero(labels == index)) for index in range(component_count)]
        min_allowed_size = max(1, int(np.ceil(matrix.shape[0] * min_cluster_fraction)))
        has_small_cluster = min(cluster_sizes) < min_allowed_size
        effective_bic = float("inf") if has_small_cluster else float(model["bic"])
        trials.append(
            {
                "components": component_count,
                "bic": float(model["bic"]),
                "effective_bic": effective_bic,
                "log_likelihood": float(model["log_likelihood"]),
                "iterations": int(model["iterations"]),
                "cluster_sizes": cluster_sizes,
                "min_allowed_cluster_size": min_allowed_size,
                "rejected_for_small_cluster": has_small_cluster,
            }
        )
        status = "rejected small cluster" if has_small_cluster else "accepted"
        log_step(f"k={component_count}: BIC={model['bic']:.4g}, min cluster size={min(cluster_sizes)}, {status}.")
        if effective_bic < best_effective_bic:
            best_effective_bic = effective_bic
            best_components = component_count
            best_model = model

    if best_model is None:
        raise ValueError("GMM auto-selection did not produce a valid model. Try lowering --min-cluster-fraction.")
    return best_components, best_model, trials


def write_candidate_gmm_results(
    base_dir: Path,
    trials: list[dict],
    energy: np.ndarray,
    matrix: np.ndarray,
    samples: list,
    direction: str,
    max_iterations: int,
    tolerance: float,
    regularization: float,
) -> None:
    log_step(f"Writing candidate GMM component-count results under: {base_dir}")
    for trial in trials:
        component_count = int(trial["components"])
        model = fit_gmm_diagonal(
            matrix,
            component_count=component_count,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
        )
        write_cluster_result_bundle(
            output_dir=base_dir / f"k{component_count}",
            energy=energy,
            matrix=matrix,
            samples=samples,
            labels=model["labels"],
            centroids=model["means"],
            direction=direction,
            summary_extra={
                "selection_trial": trial,
                "gmm": {
                    "weights": model["weights"],
                    "bic": model["bic"],
                    "log_likelihood": model["log_likelihood"],
                    "iterations": model["iterations"],
                },
            },
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a GMM to MAVEN SWE paired electron spectra.")
    parser.add_argument("--start", help="Optional UTC start time, for example 2024-11-07T00:00:00.")
    parser.add_argument("--end", help="Optional UTC end time, for example 2024-11-08T00:00:00.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for downloaded MAVEN data.")
    parser.add_argument("--swe-file", action="append", help="Explicit SWE svypad CDF file. Can be repeated.")
    parser.add_argument("--no-auto-download", action="store_true", help="Fail instead of downloading missing SWE files.")
    parser.add_argument("--check-data-only", action="store_true", help="Check/download data coverage and exit before GMM.")
    parser.add_argument(
        "--direction",
        choices=("both", "parallel", "anti_parallel"),
        default="both",
        help="Use paired parallel+anti_parallel spectra per timestamp, or only one direction.",
    )
    parser.add_argument("--parallel-pitch-max", type=float, default=30.0)
    parser.add_argument("--anti-parallel-pitch-min", type=float, default=150.0)
    parser.add_argument("--components", type=int, default=4, help="Number of GMM components when not using --auto-clusters.")
    parser.add_argument("--clusters", type=int, help="Alias for --components.")
    parser.add_argument("--auto-clusters", action="store_true", help="Choose GMM component count by lowest BIC.")
    parser.add_argument("--min-clusters", type=int, default=2, help="Smallest component count tried by --auto-clusters.")
    parser.add_argument("--max-clusters", type=int, default=10, help="Largest component count tried by --auto-clusters.")
    parser.add_argument("--min-cluster-fraction", type=float, default=0.01)
    parser.add_argument(
        "--no-save-candidates",
        action="store_true",
        help="When using --auto-clusters, do not write plots/tables for every tried component count.",
    )
    parser.add_argument("--stride", type=int, default=1)
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
        help="Preprocessing method; log preserves parallel/anti-parallel flux differences.",
    )
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum EM iterations.")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="EM log-likelihood convergence tolerance.")
    parser.add_argument("--regularization", type=float, default=1e-6, help="Minimum diagonal covariance.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for GMM outputs.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.clusters is not None:
        args.components = args.clusters
    # Reuse the k-means naming helper by giving it the attribute it expects.
    args.clusters = args.components

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
    log_step(f"Using {len(files)} SWE file(s) for GMM analysis.")

    output_root = Path(args.output_root).expanduser().resolve()
    run_name = f"gmm_{build_run_name(args, start_dt, end_dt)}"
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
    log_step(f"Feature matrix ready: {matrix.shape[0]} sample(s) x {matrix.shape[1]} feature(s).")

    if args.auto_clusters:
        selected_components, model, cluster_trials = choose_gmm_component_count(
            matrix,
            min_components=args.min_clusters,
            max_components=args.max_clusters,
            min_cluster_fraction=args.min_cluster_fraction,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            regularization=args.regularization,
        )
        cluster_selection = {
            "method": "bic",
            "selected_components": selected_components,
            "min_cluster_fraction": args.min_cluster_fraction,
            "trials": cluster_trials,
        }
        print(f"Auto-selected {selected_components} GMM components by lowest BIC.")
    else:
        selected_components = args.components
        log_step(f"Running GMM with manually selected k={selected_components}.")
        model = fit_gmm_diagonal(
            matrix,
            component_count=selected_components,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            regularization=args.regularization,
        )
        cluster_selection = None

    labels = model["labels"]
    centroids = model["means"]
    log_step("Computing PCA projection.")
    scores, components, explained = pca_scores(matrix, n_components=2)
    reps = representative_indices(matrix, labels, centroids)

    representative_rows = []
    for cluster_index, sample_index in enumerate(reps):
        members = np.flatnonzero(labels == cluster_index)
        distance = float(np.linalg.norm(matrix[sample_index] - centroids[cluster_index]))
        representative_rows.append(
            {
                "cluster": cluster_index + 1,
                "sample_count": int(members.size),
                "representative_time_utc": format_unix_time(samples[sample_index].time_unix),
                "representative_source_file": samples[sample_index].source_file,
                "distance_to_cluster_center": distance,
            }
        )

    plot_cluster_spectra(energy, samples, labels, reps, output_dir / "characteristic_spectra.png", args.direction)
    plot_pca(scores, labels, reps, output_dir / "pca_clusters.png")
    write_representatives_csv(output_dir / "representative_times.csv", representative_rows)
    if args.auto_clusters and cluster_selection is not None and not args.no_save_candidates:
        write_candidate_gmm_results(
            base_dir=output_dir / "candidate_clusters",
            trials=cluster_selection["trials"],
            energy=energy,
            matrix=matrix,
            samples=samples,
            direction=args.direction,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            regularization=args.regularization,
        )

    summary = {
        "settings": {
            "method": "gmm_diagonal_covariance",
            "start": args.start,
            "end": args.end,
            "components": selected_components,
            "requested_components": args.components,
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
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "regularization": args.regularization,
            "data_root": str(data_root),
            "input_files": [str(path) for path in files],
            "run_name": run_name,
            "output_dir": str(output_dir),
        },
        "data_coverage": data_coverage_report,
        "cluster_selection": cluster_selection,
        "gmm": {
            "weights": model["weights"],
            "bic": model["bic"],
            "log_likelihood": model["log_likelihood"],
            "iterations": model["iterations"],
        },
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
    (output_dir / "gmm_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_step(f"Summary JSON written: {output_dir / 'gmm_summary.json'}")
    print(f"Loaded {len(samples)} spectra from {len(files)} SWE file(s).")
    print(f"GMM output written to: {output_dir}")
    for row in representative_rows:
        print(
            f"Component {row['cluster']}: n={row['sample_count']}, "
            f"representative={row['representative_time_utc']}"
        )


if __name__ == "__main__":
    main()
