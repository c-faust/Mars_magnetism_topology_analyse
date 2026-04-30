from __future__ import annotations
"""
GMM analysis of MAVEN SWE energy-derivative spectra.

This combines the derivative preprocessing from
`analyze_electron_spectra_derivative_ml.py` with the diagonal-covariance GMM
from `analyze_electron_spectra_gmm.py`.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.analyze_electron_spectra_derivative_ml import (
    DEFAULT_OUTPUT_ROOT as DERIVATIVE_OUTPUT_ROOT,
    load_derivative_samples,
    plot_derivative_cluster_spectra,
    plot_pca,
    representative_rows,
    write_representatives_csv,
)
from machine_learning.analyze_electron_spectra_gmm import (
    choose_gmm_component_count,
    fit_gmm_diagonal,
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


DEFAULT_OUTPUT_ROOT = DERIVATIVE_OUTPUT_ROOT.parent / "derivative_gmm_analysis"


def write_derivative_gmm_bundle(
    output_dir: Path,
    energy: np.ndarray,
    matrix: np.ndarray,
    samples: list,
    labels: np.ndarray,
    means: np.ndarray,
    direction: str,
    summary_extra: dict | None = None,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    from machine_learning.analyze_electron_spectra_ml import pca_scores

    scores, components, explained = pca_scores(matrix, n_components=2)
    reps = representative_indices(matrix, labels, means)
    rows = representative_rows(matrix, samples, labels, means, reps)
    plot_derivative_cluster_spectra(
        energy,
        samples,
        labels,
        reps,
        output_dir / "characteristic_derivative_spectra.png",
        direction,
    )
    plot_pca(scores, labels, reps, output_dir / "pca_clusters.png")
    write_representatives_csv(output_dir / "representative_times.csv", rows)
    summary = {
        "sample_count": len(samples),
        "component_count": int(means.shape[0]),
        "cluster_sizes": [int(np.count_nonzero(labels == index)) for index in range(means.shape[0])],
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


def write_candidate_derivative_gmm_results(
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
    log_step(f"Writing candidate derivative GMM component-count results under: {base_dir}")
    for trial in trials:
        component_count = int(trial["components"])
        model = fit_gmm_diagonal(
            matrix,
            component_count=component_count,
            max_iterations=max_iterations,
            tolerance=tolerance,
            regularization=regularization,
        )
        write_derivative_gmm_bundle(
            output_dir=base_dir / f"k{component_count}",
            energy=energy,
            matrix=matrix,
            samples=samples,
            labels=model["labels"],
            means=model["means"],
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
    parser = argparse.ArgumentParser(description="Fit a GMM to MAVEN SWE energy-derivative spectra.")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--swe-file", action="append")
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--direction", choices=("both", "parallel", "anti_parallel"), default="both")
    parser.add_argument("--parallel-pitch-max", type=float, default=30.0)
    parser.add_argument("--anti-parallel-pitch-min", type=float, default=150.0)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--clusters", type=int, help="Alias for --components.")
    parser.add_argument("--auto-clusters", action="store_true")
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.01)
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

    run_name = f"derivative_gmm_{build_run_name(args, start_dt, end_dt)}"
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
    log_step(f"Derivative GMM feature matrix ready: {matrix.shape[0]} sample(s) x {matrix.shape[1]} feature(s).")

    if args.auto_clusters:
        selected_components, model, trials = choose_gmm_component_count(
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
            "trials": trials,
        }
        print(f"Auto-selected {selected_components} derivative GMM components by lowest BIC.")
    else:
        selected_components = args.components
        log_step(f"Running derivative GMM with manually selected k={selected_components}.")
        model = fit_gmm_diagonal(
            matrix,
            component_count=selected_components,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            regularization=args.regularization,
        )
        cluster_selection = None

    rows = write_derivative_gmm_bundle(
        output_dir=output_dir,
        energy=energy,
        matrix=matrix,
        samples=samples,
        labels=model["labels"],
        means=model["means"],
        direction=args.direction,
    )
    if args.auto_clusters and cluster_selection is not None and not args.no_save_candidates:
        write_candidate_derivative_gmm_results(
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
            "method": "gmm_diagonal_covariance_on_energy_derivative_spectra",
            "start": args.start,
            "end": args.end,
            "components": selected_components,
            "requested_components": args.components,
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
        "representatives": rows,
        "outputs": {
            "representative_times_csv": str(output_dir / "representative_times.csv"),
            "characteristic_derivative_spectra_png": str(output_dir / "characteristic_derivative_spectra.png"),
            "pca_clusters_png": str(output_dir / "pca_clusters.png"),
        },
    }
    (output_dir / "derivative_gmm_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_step(f"Summary JSON written: {output_dir / 'derivative_gmm_summary.json'}")
    print(f"Loaded {len(samples)} derivative spectra from {len(files)} SWE file(s).")
    print(f"Derivative GMM output written to: {output_dir}")
    for row in rows:
        print(f"Component {row['cluster']}: n={row['sample_count']}, representative={row['representative_time_utc']}")


if __name__ == "__main__":
    main()
