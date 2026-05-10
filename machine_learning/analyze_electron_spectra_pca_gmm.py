from __future__ import annotations
"""
Large-sample PCA + GMM analysis for MAVEN SWE electron spectra.

This script keeps the same paired parallel/anti-parallel feature construction
as `analyze_electron_spectra_gmm.py`, but optimizes the clustering workflow for
large sample counts:

1. Reduce the normalized spectral vectors with PCA.
2. Train diagonal-covariance GMMs on a reproducible random subset.
3. Scan k on the subset with BIC/AIC and cluster-size checks.
4. Use the selected GMM to predict labels/probabilities for all PCA scores.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process_maven_spectra import format_unix_time
from machine_learning.analyze_electron_spectra_gmm import (
    estimate_log_gaussian_probability,
    fit_gmm_diagonal,
    logsumexp,
)
from machine_learning.analyze_electron_spectra_ml import (
    DEFAULT_DATA_ROOT,
    ML_ROOT,
    build_run_name,
    ensure_required_swe_data,
    infer_swe_files,
    load_samples,
    log_step,
    parse_iso_datetime,
    parse_iso_time,
    plot_cluster_spectra,
    plot_pca,
    representative_indices,
    sanitize_for_json,
    unique_output_dir,
    valid_flux_fraction,
)


DEFAULT_OUTPUT_ROOT = ML_ROOT / "outputs" / "pca_gmm_analysis"


def fit_pca(matrix: np.ndarray, n_components: int) -> dict:
    if n_components < 1:
        raise ValueError("--pca-components must be at least 1.")
    if matrix.shape[1] < n_components:
        raise ValueError(f"Cannot keep {n_components} PCA components from {matrix.shape[1]} input features.")

    log_step(f"Fitting PCA: {matrix.shape[1]} feature(s) -> {n_components} component(s).")
    mean = np.nanmean(matrix, axis=0)
    centered = matrix - mean[None, :]
    covariance = (centered.T @ centered) / max(matrix.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    components = eigenvectors[:, :n_components].T
    scores = centered @ components.T
    total_variance = float(np.sum(eigenvalues))
    explained = eigenvalues[:n_components] / total_variance if total_variance > 0 else np.zeros(n_components)
    return {
        "scores": scores,
        "mean": mean,
        "components": components,
        "explained_variance": eigenvalues[:n_components],
        "explained_variance_ratio": explained,
        "cumulative_explained_variance_ratio": np.cumsum(explained),
        "total_variance": total_variance,
    }


def predict_gmm_diagonal(matrix: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    weighted_log_prob = estimate_log_gaussian_probability(matrix, model["means"], model["variances"])
    weighted_log_prob = weighted_log_prob + np.log(model["weights"][None, :])
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    probabilities = np.exp(weighted_log_prob - log_prob_norm[:, None])
    labels = np.argmax(probabilities, axis=1)
    max_probability = np.max(probabilities, axis=1)
    log_likelihood = float(np.sum(log_prob_norm))
    return labels, probabilities, max_probability, log_likelihood


def gmm_parameter_count(component_count: int, dimension: int) -> int:
    return component_count * (2 * dimension) + (component_count - 1)


def information_criteria(log_likelihood: float, sample_count: int, component_count: int, dimension: int) -> tuple[float, float]:
    parameter_count = gmm_parameter_count(component_count, dimension)
    bic = -2.0 * log_likelihood + parameter_count * np.log(sample_count)
    aic = -2.0 * log_likelihood + 2.0 * parameter_count
    return float(bic), float(aic)


def choose_training_indices(sample_count: int, sample_size: int, random_seed: int) -> np.ndarray:
    if sample_size <= 0 or sample_size >= sample_count:
        log_step(f"Training GMM on all {sample_count} PCA score(s).")
        return np.arange(sample_count, dtype=int)
    rng = np.random.default_rng(random_seed)
    indices = np.sort(rng.choice(sample_count, size=sample_size, replace=False))
    log_step(f"Training GMM on a random subset: {indices.size}/{sample_count} sample(s), seed={random_seed}.")
    return indices


def scan_gmm_components(
    training_scores: np.ndarray,
    min_components: int,
    max_components: int,
    min_cluster_fraction: float,
    max_iterations: int,
    tolerance: float,
    regularization: float,
) -> tuple[int, dict, list[dict]]:
    upper = min(max_components, training_scores.shape[0])
    lower = max(1, min_components)
    if upper < lower:
        raise ValueError(f"Cannot scan k={lower}-{upper}; not enough training samples.")

    best_model: dict | None = None
    best_components = lower
    best_effective_bic = float("inf")
    trials: list[dict] = []

    for component_count in range(lower, upper + 1):
        log_step(f"PCA-GMM k scan: fitting k={component_count}.")
        model = fit_gmm_diagonal(
            training_scores,
            component_count=component_count,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
        )
        labels, probabilities, _, log_likelihood = predict_gmm_diagonal(training_scores, model)
        bic, aic = information_criteria(
            log_likelihood=log_likelihood,
            sample_count=training_scores.shape[0],
            component_count=component_count,
            dimension=training_scores.shape[1],
        )
        cluster_sizes = [int(np.count_nonzero(labels == index)) for index in range(component_count)]
        cluster_fractions = [float(size / training_scores.shape[0]) for size in cluster_sizes]
        min_allowed_size = max(1, int(np.ceil(training_scores.shape[0] * min_cluster_fraction)))
        has_small_cluster = min(cluster_sizes) < min_allowed_size
        effective_bic = float("inf") if has_small_cluster else bic
        trial = {
            "components": component_count,
            "bic": bic,
            "aic": aic,
            "effective_bic": effective_bic,
            "log_likelihood": log_likelihood,
            "iterations": int(model["iterations"]),
            "cluster_sizes": cluster_sizes,
            "cluster_fractions": cluster_fractions,
            "min_allowed_cluster_size": min_allowed_size,
            "rejected_for_small_cluster": has_small_cluster,
            "mean_max_probability": float(np.mean(np.max(probabilities, axis=1))),
        }
        trials.append(trial)
        status = "rejected small cluster" if has_small_cluster else "accepted"
        log_step(
            f"k={component_count}: BIC={bic:.4g}, AIC={aic:.4g}, "
            f"min cluster size={min(cluster_sizes)}, {status}."
        )
        if effective_bic < best_effective_bic:
            best_effective_bic = effective_bic
            best_components = component_count
            best_model = model

    if best_model is None:
        raise ValueError("No valid k was selected. Try lowering --min-cluster-fraction.")
    return best_components, best_model, trials


def write_k_scan_csv(path: Path, trials: list[dict]) -> None:
    fieldnames = [
        "components",
        "bic",
        "aic",
        "effective_bic",
        "log_likelihood",
        "iterations",
        "min_cluster_size",
        "max_cluster_size",
        "min_cluster_fraction",
        "max_cluster_fraction",
        "rejected_for_small_cluster",
        "cluster_sizes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            sizes = trial["cluster_sizes"]
            fractions = trial["cluster_fractions"]
            writer.writerow(
                {
                    "components": trial["components"],
                    "bic": trial["bic"],
                    "aic": trial["aic"],
                    "effective_bic": trial["effective_bic"],
                    "log_likelihood": trial["log_likelihood"],
                    "iterations": trial["iterations"],
                    "min_cluster_size": min(sizes),
                    "max_cluster_size": max(sizes),
                    "min_cluster_fraction": min(fractions),
                    "max_cluster_fraction": max(fractions),
                    "rejected_for_small_cluster": trial["rejected_for_small_cluster"],
                    "cluster_sizes": " ".join(str(size) for size in sizes),
                }
            )


def plot_k_scan(trials: list[dict], output_dir: Path) -> None:
    if not trials:
        return
    ks = np.asarray([trial["components"] for trial in trials], dtype=int)
    bic = np.asarray([trial["bic"] for trial in trials], dtype=float)
    aic = np.asarray([trial["aic"] for trial in trials], dtype=float)
    min_fraction = np.asarray([min(trial["cluster_fractions"]) for trial in trials], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), constrained_layout=True)
    axes[0].plot(ks, bic, marker="o", label="BIC")
    axes[0].plot(ks, aic, marker="s", label="AIC")
    axes[0].set_xlabel("GMM components")
    axes[0].set_ylabel("criterion")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(ks, min_fraction, marker="o", color="#4f7f3f")
    axes[1].set_xlabel("GMM components")
    axes[1].set_ylabel("minimum cluster fraction")
    axes[1].grid(alpha=0.25)
    fig.savefig(output_dir / "k_scan_bic_aic_cluster_size.png", dpi=180)
    plt.close(fig)


def write_full_assignments(
    path: Path,
    samples: list,
    labels: np.ndarray,
    max_probabilities: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_utc", "cluster", "max_probability", "source_file"])
        writer.writeheader()
        for sample, label, max_probability in zip(samples, labels, max_probabilities):
            writer.writerow(
                {
                    "time_utc": format_unix_time(sample.time_unix),
                    "cluster": int(label) + 1,
                    "max_probability": float(max_probability),
                    "source_file": sample.source_file,
                }
            )


def write_representatives_csv_dynamic(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "cluster",
        "sample_count",
        "representative_time_utc",
        "parallel_valid_fraction",
        "anti_parallel_valid_fraction",
        "representative_source_file",
        "distance_to_cluster_center",
        "representative_max_probability",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_representative_rows_for_pca_gmm(
    scores: np.ndarray,
    samples: list,
    labels: np.ndarray,
    centroids: np.ndarray,
    reps: list[int],
    max_probabilities: np.ndarray,
) -> list[dict]:
    rows = []
    for cluster_index, sample_index in enumerate(reps):
        members = np.flatnonzero(labels == cluster_index)
        distance = float(np.linalg.norm(scores[sample_index] - centroids[cluster_index]))
        rows.append(
            {
                "cluster": cluster_index + 1,
                "sample_count": int(members.size),
                "representative_time_utc": format_unix_time(samples[sample_index].time_unix),
                "parallel_valid_fraction": valid_flux_fraction(samples[sample_index].parallel_flux),
                "anti_parallel_valid_fraction": valid_flux_fraction(samples[sample_index].anti_parallel_flux),
                "representative_source_file": samples[sample_index].source_file,
                "distance_to_cluster_center": distance,
                "representative_max_probability": float(max_probabilities[sample_index]),
            }
        )
    return rows


def write_pca_gmm_bundle(
    output_dir: Path,
    energy: np.ndarray,
    samples: list,
    pca_scores: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    max_probabilities: np.ndarray,
    model: dict,
    direction: str,
    trial: dict | None = None,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reps = representative_indices(pca_scores, labels, model["means"])
    representative_rows = build_representative_rows_for_pca_gmm(
        pca_scores, samples, labels, model["means"], reps, max_probabilities
    )
    plot_cluster_spectra(energy, samples, labels, reps, output_dir / "characteristic_spectra.png", direction)
    plot_pca(pca_scores[:, :2], labels, reps, output_dir / "pca_clusters.png")
    write_representatives_csv_dynamic(output_dir / "representative_times.csv", representative_rows)
    cluster_sizes = [int(np.count_nonzero(labels == index)) for index in range(model["means"].shape[0])]
    summary = {
        "sample_count": len(samples),
        "cluster_count": int(model["means"].shape[0]),
        "cluster_sizes": cluster_sizes,
        "cluster_fractions": [float(size / len(samples)) for size in cluster_sizes],
        "mean_max_probability": float(np.mean(max_probabilities)),
        "representatives": representative_rows,
        "selection_trial": trial,
        "outputs": {
            "representative_times_csv": str(output_dir / "representative_times.csv"),
            "characteristic_spectra_png": str(output_dir / "characteristic_spectra.png"),
            "pca_clusters_png": str(output_dir / "pca_clusters.png"),
        },
    }
    np.savez_compressed(
        output_dir / "predict_proba_full.npz",
        labels=labels.astype(np.int16),
        probabilities=probabilities.astype(np.float32),
        max_probabilities=max_probabilities.astype(np.float32),
    )
    summary["outputs"]["predict_proba_full_npz"] = str(output_dir / "predict_proba_full.npz")
    (output_dir / "cluster_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return representative_rows


def write_candidate_pca_gmm_results(
    base_dir: Path,
    trials: list[dict],
    energy: np.ndarray,
    samples: list,
    all_scores: np.ndarray,
    training_scores: np.ndarray,
    direction: str,
    max_iterations: int,
    tolerance: float,
    regularization: float,
) -> None:
    log_step(f"Writing candidate PCA-GMM results under: {base_dir}")
    trial_by_k = {int(trial["components"]): trial for trial in trials}
    for component_count in sorted(trial_by_k):
        log_step(f"Writing candidate output for k={component_count}.")
        model = fit_gmm_diagonal(
            training_scores,
            component_count=component_count,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
        )
        labels, probabilities, max_probabilities, _ = predict_gmm_diagonal(all_scores, model)
        write_pca_gmm_bundle(
            output_dir=base_dir / f"k{component_count}",
            energy=energy,
            samples=samples,
            pca_scores=all_scores,
            labels=labels,
            probabilities=probabilities,
            max_probabilities=max_probabilities,
            model=model,
            direction=direction,
            trial=trial_by_k[component_count],
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PCA + sampled GMM on MAVEN SWE paired electron spectra.")
    parser.add_argument("--start", help="Optional UTC start time, for example 2024-11-07T00:00:00.")
    parser.add_argument("--end", help="Optional UTC end time, for example 2024-11-08T00:00:00.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for downloaded MAVEN data.")
    parser.add_argument("--swe-file", action="append", help="Explicit SWE svypad CDF file. Can be repeated.")
    parser.add_argument("--no-auto-download", action="store_true", help="Fail instead of downloading missing SWE files.")
    parser.add_argument("--check-data-only", action="store_true", help="Check/download data coverage and exit before analysis.")
    parser.add_argument(
        "--direction",
        choices=("both", "parallel", "anti_parallel"),
        default="both",
        help="Use paired parallel+anti_parallel spectra per timestamp, or only one direction.",
    )
    parser.add_argument("--parallel-pitch-max", type=float, default=30.0)
    parser.add_argument("--anti-parallel-pitch-min", type=float, default=150.0)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--sample-size", type=int, default=100000, help="Random subset size used to train/scan GMM.")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--components", type=int, help="Manual GMM component count. If omitted, scan --min-clusters..--max-clusters.")
    parser.add_argument("--clusters", type=int, help="Alias for --components.")
    parser.add_argument("--min-clusters", type=int, default=6)
    parser.add_argument("--max-clusters", type=int, default=15)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.005)
    parser.add_argument("--no-save-candidates", action="store_true", help="Do not write per-k candidate result folders.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-direction-valid-fraction", type=float, default=0.7)
    parser.add_argument(
        "--normalization",
        choices=("log", "global_zscore", "zscore", "minmax", "l2"),
        default="log",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for PCA-GMM outputs.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.clusters is not None:
        args.components = args.clusters
    args.auto_clusters = args.components is None
    args.clusters = args.components if args.components is not None else args.max_clusters

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
    log_step(f"Using {len(files)} SWE file(s) for PCA-GMM analysis.")

    output_root = Path(args.output_root).expanduser().resolve()
    run_name = f"pca{args.pca_components}_sample{args.sample_size}_gmm_{build_run_name(args, start_dt, end_dt)}"
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

    pca = fit_pca(matrix, n_components=args.pca_components)
    all_scores = np.asarray(pca["scores"], dtype=float)
    training_indices = choose_training_indices(all_scores.shape[0], args.sample_size, args.random_seed)
    training_scores = all_scores[training_indices]

    if args.components is None:
        selected_components, model, cluster_trials = scan_gmm_components(
            training_scores=training_scores,
            min_components=args.min_clusters,
            max_components=args.max_clusters,
            min_cluster_fraction=args.min_cluster_fraction,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            regularization=args.regularization,
        )
        cluster_selection = {
            "method": "sample_bic_with_min_cluster_fraction",
            "selected_components": selected_components,
            "training_sample_count": int(training_scores.shape[0]),
            "min_cluster_fraction": args.min_cluster_fraction,
            "trials": cluster_trials,
        }
        print(f"Auto-selected {selected_components} GMM components by sampled BIC.")
    else:
        selected_components = args.components
        log_step(f"Running PCA-GMM with manually selected k={selected_components}.")
        model = fit_gmm_diagonal(
            training_scores,
            component_count=selected_components,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            regularization=args.regularization,
        )
        training_labels, training_probabilities, _, training_log_likelihood = predict_gmm_diagonal(training_scores, model)
        bic, aic = information_criteria(training_log_likelihood, training_scores.shape[0], selected_components, training_scores.shape[1])
        cluster_selection = {
            "method": "manual",
            "selected_components": selected_components,
            "training_sample_count": int(training_scores.shape[0]),
            "trials": [
                {
                    "components": selected_components,
                    "bic": bic,
                    "aic": aic,
                    "effective_bic": bic,
                    "log_likelihood": training_log_likelihood,
                    "iterations": int(model["iterations"]),
                    "cluster_sizes": [int(np.count_nonzero(training_labels == index)) for index in range(selected_components)],
                    "cluster_fractions": [
                        float(np.count_nonzero(training_labels == index) / training_scores.shape[0])
                        for index in range(selected_components)
                    ],
                    "min_allowed_cluster_size": max(1, int(np.ceil(training_scores.shape[0] * args.min_cluster_fraction))),
                    "rejected_for_small_cluster": False,
                    "mean_max_probability": float(np.mean(np.max(training_probabilities, axis=1))),
                }
            ],
        }

    labels, probabilities, max_probabilities, full_log_likelihood = predict_gmm_diagonal(all_scores, model)
    full_bic, full_aic = information_criteria(full_log_likelihood, all_scores.shape[0], selected_components, all_scores.shape[1])
    representative_rows = write_pca_gmm_bundle(
        output_dir=output_dir,
        energy=energy,
        samples=samples,
        pca_scores=all_scores,
        labels=labels,
        probabilities=probabilities,
        max_probabilities=max_probabilities,
        model=model,
        direction=args.direction,
        trial=None,
    )
    write_full_assignments(output_dir / "full_assignments.csv", samples, labels, max_probabilities)

    write_k_scan_csv(output_dir / "k_scan_metrics.csv", cluster_selection["trials"])
    plot_k_scan(cluster_selection["trials"], output_dir)
    if args.components is None and not args.no_save_candidates:
        write_candidate_pca_gmm_results(
            base_dir=output_dir / "candidate_clusters",
            trials=cluster_selection["trials"],
            energy=energy,
            samples=samples,
            all_scores=all_scores,
            training_scores=training_scores,
            direction=args.direction,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            regularization=args.regularization,
        )

    np.savez_compressed(
        output_dir / "pca_model_and_scores.npz",
        scores=all_scores.astype(np.float32),
        mean=np.asarray(pca["mean"], dtype=np.float64),
        components=np.asarray(pca["components"], dtype=np.float64),
        explained_variance=np.asarray(pca["explained_variance"], dtype=np.float64),
        explained_variance_ratio=np.asarray(pca["explained_variance_ratio"], dtype=np.float64),
        training_indices=training_indices.astype(np.int64),
    )
    summary = {
        "settings": {
            "method": "pca_sampled_gmm_diagonal_covariance",
            "start": args.start,
            "end": args.end,
            "components": selected_components,
            "requested_components": args.components,
            "auto_clusters": args.components is None,
            "min_clusters": args.min_clusters,
            "max_clusters": args.max_clusters,
            "min_cluster_fraction": args.min_cluster_fraction,
            "pca_components": args.pca_components,
            "sample_size": args.sample_size,
            "actual_training_sample_count": int(training_scores.shape[0]),
            "random_seed": args.random_seed,
            "save_candidate_cluster_results": args.components is None and not args.no_save_candidates,
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
            "bic_full": full_bic,
            "aic_full": full_aic,
            "log_likelihood_full": full_log_likelihood,
            "iterations": model["iterations"],
            "mean_max_probability_full": float(np.mean(max_probabilities)),
        },
        "sample_count": len(samples),
        "energy_eV": energy,
        "pca_explained_variance_ratio": pca["explained_variance_ratio"],
        "pca_cumulative_explained_variance_ratio": pca["cumulative_explained_variance_ratio"],
        "representatives": representative_rows,
        "outputs": {
            "representative_times_csv": str(output_dir / "representative_times.csv"),
            "characteristic_spectra_png": str(output_dir / "characteristic_spectra.png"),
            "pca_clusters_png": str(output_dir / "pca_clusters.png"),
            "k_scan_metrics_csv": str(output_dir / "k_scan_metrics.csv"),
            "k_scan_plot_png": str(output_dir / "k_scan_bic_aic_cluster_size.png"),
            "full_assignments_csv": str(output_dir / "full_assignments.csv"),
            "predict_proba_full_npz": str(output_dir / "predict_proba_full.npz"),
            "pca_model_and_scores_npz": str(output_dir / "pca_model_and_scores.npz"),
        },
    }
    (output_dir / "pca_gmm_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_step(f"Summary JSON written: {output_dir / 'pca_gmm_summary.json'}")
    print(f"Loaded {len(samples)} spectra from {len(files)} SWE file(s).")
    print(f"PCA-GMM output written to: {output_dir}")
    for row in representative_rows:
        print(
            f"Component {row['cluster']}: n={row['sample_count']}, "
            f"representative={row['representative_time_utc']}, "
            f"p={row['representative_max_probability']:.3f}"
        )


if __name__ == "__main__":
    main()
