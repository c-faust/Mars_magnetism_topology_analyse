from __future__ import annotations
"""
Large-sample PCA + GMM analysis for MAVEN SWE energy-derivative spectra.

The feature construction follows `analyze_electron_spectra_derivative_gmm.py`:
each timestamp is converted to dFlux/dE spectra, with parallel and
anti-parallel directions paired by default. The large-sample optimization then
matches `analyze_electron_spectra_pca_gmm.py`: PCA reduction, sampled GMM k
scan, and full-data GMM probability prediction.
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
from machine_learning.analyze_electron_spectra_derivative_ml import (
    DEFAULT_OUTPUT_ROOT as DERIVATIVE_OUTPUT_ROOT,
    DEFAULT_MIN_PLASMA_ENERGY_EV,
    DEFAULT_SCPOT_MIN_FLAG,
    ensure_required_lpw_data,
    load_derivative_samples,
    maven_file_date,
    plot_derivative_cluster_spectra,
    plot_pca,
    representative_rows,
)
from machine_learning.analyze_electron_spectra_ml import (
    DEFAULT_DATA_ROOT,
    build_run_name,
    ensure_required_swe_data,
    infer_swe_files,
    log_step,
    parse_iso_datetime,
    parse_iso_time,
    representative_indices,
    sanitize_for_json,
    unique_output_dir,
)
from machine_learning.analyze_electron_spectra_pca_gmm import (
    choose_training_indices,
    fit_pca,
    information_criteria,
    plot_k_scan,
    predict_gmm_diagonal,
    scan_gmm_components,
    write_full_assignments,
    write_k_scan_csv,
)
from machine_learning.analyze_electron_spectra_gmm import fit_gmm_diagonal


DEFAULT_OUTPUT_ROOT = DERIVATIVE_OUTPUT_ROOT.parent / "derivative_pca_gmm_analysis"


def required_dates_from_swe_files(files: list[Path]) -> list:
    dates = []
    for path in files:
        day = maven_file_date(path)
        if day is not None:
            dates.append(day)
    return sorted(set(dates))


def keep_swe_files_with_lpw(files: list[Path], lpw_files_by_date: dict) -> list[Path]:
    kept = []
    skipped = 0
    for path in files:
        day = maven_file_date(path)
        if day is not None and day in lpw_files_by_date:
            kept.append(path)
        else:
            skipped += 1
    if skipped:
        log_step(f"Skipped {skipped} SWE file(s) whose date has no usable LPW mrgscpot spacecraft-potential file.")
    return kept


def write_representatives_csv_with_probability(path: Path, rows: list[dict]) -> None:
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


def derivative_representative_rows(
    scores: np.ndarray,
    samples: list,
    labels: np.ndarray,
    means: np.ndarray,
    reps: list[int],
    max_probabilities: np.ndarray,
) -> list[dict]:
    rows = representative_rows(scores, samples, labels, means, reps)
    for row, sample_index in zip(rows, reps):
        row["representative_max_probability"] = float(max_probabilities[sample_index])
    return rows


def write_derivative_pca_gmm_bundle(
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
    rows = derivative_representative_rows(pca_scores, samples, labels, model["means"], reps, max_probabilities)
    plot_derivative_cluster_spectra(
        energy,
        samples,
        labels,
        reps,
        output_dir / "characteristic_derivative_spectra.png",
        direction,
    )
    plot_pca(pca_scores[:, :2], labels, reps, output_dir / "pca_clusters.png")
    write_representatives_csv_with_probability(output_dir / "representative_times.csv", rows)
    cluster_sizes = [int(np.count_nonzero(labels == index)) for index in range(model["means"].shape[0])]
    np.savez_compressed(
        output_dir / "predict_proba_full.npz",
        labels=labels.astype(np.int16),
        probabilities=probabilities.astype(np.float32),
        max_probabilities=max_probabilities.astype(np.float32),
    )
    summary = {
        "sample_count": len(samples),
        "component_count": int(model["means"].shape[0]),
        "cluster_sizes": cluster_sizes,
        "cluster_fractions": [float(size / len(samples)) for size in cluster_sizes],
        "mean_max_probability": float(np.mean(max_probabilities)),
        "representatives": rows,
        "selection_trial": trial,
        "outputs": {
            "representative_times_csv": str(output_dir / "representative_times.csv"),
            "characteristic_derivative_spectra_png": str(output_dir / "characteristic_derivative_spectra.png"),
            "pca_clusters_png": str(output_dir / "pca_clusters.png"),
            "predict_proba_full_npz": str(output_dir / "predict_proba_full.npz"),
        },
    }
    (output_dir / "cluster_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def write_candidate_derivative_pca_gmm_results(
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
    log_step(f"Writing candidate derivative PCA-GMM results under: {base_dir}")
    trial_by_k = {int(trial["components"]): trial for trial in trials}
    for component_count in sorted(trial_by_k):
        log_step(f"Writing derivative PCA-GMM candidate output for k={component_count}.")
        model = fit_gmm_diagonal(
            training_scores,
            component_count=component_count,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
        )
        labels, probabilities, max_probabilities, _ = predict_gmm_diagonal(all_scores, model)
        write_derivative_pca_gmm_bundle(
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
    parser = argparse.ArgumentParser(description="Run PCA + sampled GMM on MAVEN SWE energy-derivative spectra.")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--swe-file", action="append")
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--direction", choices=("both", "parallel", "anti_parallel"), default="both")
    parser.add_argument("--parallel-pitch-max", type=float, default=30.0)
    parser.add_argument("--anti-parallel-pitch-min", type=float, default=150.0)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--components", type=int, help="Manual GMM component count. If omitted, scan k.")
    parser.add_argument("--clusters", type=int, help="Alias for --components.")
    parser.add_argument("--min-clusters", type=int, default=6)
    parser.add_argument("--max-clusters", type=int, default=15)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.005)
    parser.add_argument("--no-save-candidates", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-direction-valid-fraction", type=float, default=0.1)
    parser.add_argument("--normalization", choices=("zscore", "global_zscore", "minmax", "l2", "none"), default="zscore")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
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

    lpw_files_by_date = {}
    required_lpw_dates = required_dates_from_swe_files(files)
    if required_lpw_dates:
        lpw_files_by_date, lpw_coverage_report = ensure_required_lpw_data(
            data_root=data_root,
            required_dates=required_lpw_dates,
            auto_download=not args.no_auto_download,
            min_flag=DEFAULT_SCPOT_MIN_FLAG,
        )
        if data_coverage_report is None:
            data_coverage_report = {}
        data_coverage_report["lpw_mrgscpot"] = lpw_coverage_report
        files = keep_swe_files_with_lpw(files, lpw_files_by_date)

    if args.check_data_only:
        print(json.dumps(sanitize_for_json(data_coverage_report), indent=2, ensure_ascii=False))
        log_step("Data coverage check finished; exiting because --check-data-only was set.")
        return
    if not files:
        raise FileNotFoundError(f"No SWE svypad files with usable LPW mrgscpot coverage were found under {data_root}.")

    output_root = Path(args.output_root).expanduser().resolve()
    run_name = f"derivative_pca{args.pca_components}_sample{args.sample_size}_gmm_{build_run_name(args, start_dt, end_dt)}"
    output_dir = unique_output_dir(output_root, run_name)
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
        lpw_files_by_date=lpw_files_by_date,
        spacecraft_potential_min_flag=DEFAULT_SCPOT_MIN_FLAG,
        min_plasma_energy_eV=DEFAULT_MIN_PLASMA_ENERGY_EV,
    )
    matrix = np.asarray([sample.normalized_flux for sample in samples], dtype=float)
    log_step(f"Derivative feature matrix ready: {matrix.shape[0]} sample(s) x {matrix.shape[1]} feature(s).")

    pca = fit_pca(matrix, n_components=args.pca_components)
    all_scores = np.asarray(pca["scores"], dtype=float)
    training_indices = choose_training_indices(all_scores.shape[0], args.sample_size, args.random_seed)
    training_scores = all_scores[training_indices]

    if args.components is None:
        selected_components, model, trials = scan_gmm_components(
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
            "trials": trials,
        }
        print(f"Auto-selected {selected_components} derivative PCA-GMM components by sampled BIC.")
    else:
        selected_components = args.components
        log_step(f"Running derivative PCA-GMM with manually selected k={selected_components}.")
        model = fit_gmm_diagonal(
            training_scores,
            component_count=selected_components,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            regularization=args.regularization,
        )
        training_labels, training_probabilities, _, training_log_likelihood = predict_gmm_diagonal(training_scores, model)
        bic, aic = information_criteria(
            training_log_likelihood,
            training_scores.shape[0],
            selected_components,
            training_scores.shape[1],
        )
        cluster_sizes = [int(np.count_nonzero(training_labels == index)) for index in range(selected_components)]
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
                    "cluster_sizes": cluster_sizes,
                    "cluster_fractions": [float(size / training_scores.shape[0]) for size in cluster_sizes],
                    "min_allowed_cluster_size": max(1, int(np.ceil(training_scores.shape[0] * args.min_cluster_fraction))),
                    "rejected_for_small_cluster": False,
                    "mean_max_probability": float(np.mean(np.max(training_probabilities, axis=1))),
                }
            ],
        }

    labels, probabilities, max_probabilities, full_log_likelihood = predict_gmm_diagonal(all_scores, model)
    full_bic, full_aic = information_criteria(
        full_log_likelihood,
        all_scores.shape[0],
        selected_components,
        all_scores.shape[1],
    )
    rows = write_derivative_pca_gmm_bundle(
        output_dir=output_dir,
        energy=energy,
        samples=samples,
        pca_scores=all_scores,
        labels=labels,
        probabilities=probabilities,
        max_probabilities=max_probabilities,
        model=model,
        direction=args.direction,
    )
    write_full_assignments(output_dir / "full_assignments.csv", samples, labels, max_probabilities)
    write_k_scan_csv(output_dir / "k_scan_metrics.csv", cluster_selection["trials"])
    plot_k_scan(cluster_selection["trials"], output_dir)

    if args.components is None and not args.no_save_candidates:
        write_candidate_derivative_pca_gmm_results(
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
            "method": "pca_sampled_gmm_diagonal_covariance_on_energy_derivative_spectra",
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
                "paired parallel + anti_parallel dFlux/dE spectra per timestamp"
                if args.direction == "both"
                else f"{args.direction} dFlux/dE spectrum per timestamp"
            ),
            "parallel_pitch_max_deg": args.parallel_pitch_max,
            "anti_parallel_pitch_min_deg": args.anti_parallel_pitch_min,
            "min_direction_valid_fraction": args.min_direction_valid_fraction,
            "spacecraft_potential_correction": {
                "enabled": True,
                "lpw_product": "lpw/l2/mrgscpot",
                "quality_rule": f"flag>{DEFAULT_SCPOT_MIN_FLAG:g}",
                "formula": "Eplasma_eV = Emeas_eV - phi_sc_V",
                "min_plasma_energy_eV": DEFAULT_MIN_PLASMA_ENERGY_EV,
                "input_lpw_files": [str(lpw_files_by_date[day]) for day in sorted(lpw_files_by_date)],
            },
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
        "representatives": rows,
        "outputs": {
            "representative_times_csv": str(output_dir / "representative_times.csv"),
            "characteristic_derivative_spectra_png": str(output_dir / "characteristic_derivative_spectra.png"),
            "pca_clusters_png": str(output_dir / "pca_clusters.png"),
            "k_scan_metrics_csv": str(output_dir / "k_scan_metrics.csv"),
            "k_scan_plot_png": str(output_dir / "k_scan_bic_aic_cluster_size.png"),
            "full_assignments_csv": str(output_dir / "full_assignments.csv"),
            "predict_proba_full_npz": str(output_dir / "predict_proba_full.npz"),
            "pca_model_and_scores_npz": str(output_dir / "pca_model_and_scores.npz"),
        },
    }
    (output_dir / "derivative_pca_gmm_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_step(f"Summary JSON written: {output_dir / 'derivative_pca_gmm_summary.json'}")
    print(f"Loaded {len(samples)} derivative spectra from {len(files)} SWE file(s).")
    print(f"Derivative PCA-GMM output written to: {output_dir}")
    for row in rows:
        print(
            f"Component {row['cluster']}: n={row['sample_count']}, "
            f"representative={row['representative_time_utc']}, "
            f"p={row['representative_max_probability']:.3f}"
        )


if __name__ == "__main__":
    main()
