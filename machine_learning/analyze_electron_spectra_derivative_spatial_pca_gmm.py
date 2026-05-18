from __future__ import annotations
"""
Run derivative PCA-GMM separately in altitude/SZA regions.

Samples are first assigned to one of 12 regions:

- A1: altitude < 800 km
- A2: 800 <= altitude < 2000 km
- A3: 2000 <= altitude < 4500 km
- A4: altitude >= 4500 km
- S1: SZA < 90 deg
- S2: 90 <= SZA < 110 deg
- S3: SZA >= 110 deg

Within each region, the feature construction follows
`analyze_electron_spectra_derivative_pca_gmm.py`, but derivative features are
not normalized before PCA.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_magnetic_topology import MARS_RADIUS_KM, load_mag_day
from machine_learning.analyze_electron_spectra_derivative_ml import (
    DEFAULT_MIN_PLASMA_ENERGY_EV,
    DEFAULT_SCPOT_MIN_FLAG,
    ensure_required_lpw_data,
    load_derivative_samples,
)
from machine_learning.analyze_electron_spectra_derivative_pca_gmm import (
    DEFAULT_OUTPUT_ROOT as DERIVATIVE_PCA_GMM_OUTPUT_ROOT,
    keep_swe_files_with_lpw,
    required_dates_from_swe_files,
    write_derivative_pca_gmm_bundle,
)
from machine_learning.analyze_electron_spectra_gmm import fit_gmm_diagonal
from machine_learning.analyze_electron_spectra_ml import (
    DEFAULT_DATA_ROOT,
    build_run_name,
    ensure_required_swe_data,
    infer_swe_files,
    log_step,
    parse_iso_datetime,
    parse_iso_time,
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
from machine_learning.plot_cluster_spatial_distributions import (
    download_mag_file,
    find_daily_mag_file,
    nearest_indices,
)
from process_maven_spectra import format_unix_time


DEFAULT_OUTPUT_ROOT = DERIVATIVE_PCA_GMM_OUTPUT_ROOT.parent / "derivative_spatial_pca_gmm_analysis"


@dataclass(frozen=True)
class SpatialBin:
    altitude_code: str
    altitude_label: str
    sza_code: str
    sza_label: str

    @property
    def code(self) -> str:
        return f"{self.altitude_code}_{self.sza_code}"

    @property
    def label(self) -> str:
        return f"{self.code}_{self.altitude_label}_{self.sza_label}"


@dataclass(frozen=True)
class SpatialContext:
    sample_index: int
    bin_code: str
    altitude_code: str
    altitude_label: str
    sza_code: str
    sza_label: str
    altitude_km: float
    sza_deg: float
    mag_ss_time_utc: str


SPATIAL_BINS = [
    SpatialBin("A1", "low_altitude", "S1", "dayside"),
    SpatialBin("A1", "low_altitude", "S2", "near_dayside_terminator"),
    SpatialBin("A1", "low_altitude", "S3", "deep_nightside"),
    SpatialBin("A2", "transition_altitude", "S1", "dayside"),
    SpatialBin("A2", "transition_altitude", "S2", "near_dayside_terminator"),
    SpatialBin("A2", "transition_altitude", "S3", "deep_nightside"),
    SpatialBin("A3", "middle_high_altitude", "S1", "dayside"),
    SpatialBin("A3", "middle_high_altitude", "S2", "near_dayside_terminator"),
    SpatialBin("A3", "middle_high_altitude", "S3", "deep_nightside"),
    SpatialBin("A4", "very_high_altitude", "S1", "dayside"),
    SpatialBin("A4", "very_high_altitude", "S2", "near_dayside_terminator"),
    SpatialBin("A4", "very_high_altitude", "S3", "deep_nightside"),
]
SPATIAL_BIN_BY_CODE = {item.code: item for item in SPATIAL_BINS}


def altitude_class(altitude_km: float) -> tuple[str, str]:
    if altitude_km < 800.0:
        return "A1", "low_altitude"
    if altitude_km < 2000.0:
        return "A2", "transition_altitude"
    if altitude_km < 4500.0:
        return "A3", "middle_high_altitude"
    return "A4", "very_high_altitude"


def sza_class(sza_deg: float) -> tuple[str, str]:
    if sza_deg < 90.0:
        return "S1", "dayside"
    if sza_deg < 110.0:
        return "S2", "near_dayside_terminator"
    return "S3", "deep_nightside"


def sample_days(samples: list) -> list[date]:
    days = {
        datetime.fromtimestamp(float(sample.time_unix), tz=timezone.utc).date()
        for sample in samples
    }
    return sorted(days)


def resolve_mag_ss_files(days: list[date], data_root: Path, auto_download: bool) -> dict[date, Path]:
    resolved: dict[date, Path] = {}
    for index, day in enumerate(days, start=1):
        log_step(f"Resolving MAG ss1s file {index}/{len(days)}: {day.isoformat()}")
        path = find_daily_mag_file([data_root], day, "ss1s")
        if path is None and auto_download:
            log_step(f"Downloading missing MAG ss1s for {day.isoformat()}.")
            path = download_mag_file(data_root, day, "ss1s")
        if path is None:
            raise FileNotFoundError(f"Missing MAG ss1s for {day.isoformat()}.")
        resolved[day] = path
    return resolved


def attach_spatial_context(samples: list, mag_ss_files: dict[date, Path]) -> list[SpatialContext]:
    by_day: dict[date, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        day = datetime.fromtimestamp(float(sample.time_unix), tz=timezone.utc).date()
        by_day[day].append(index)

    context: list[SpatialContext | None] = [None] * len(samples)
    for day, indices in sorted(by_day.items()):
        log_step(f"Attaching altitude/SZA metadata for {day.isoformat()} ({len(indices)} sample(s)).")
        mag_ss = load_mag_day(mag_ss_files[day])
        target_times = np.asarray([samples[index].time_unix for index in indices], dtype=float)
        mag_indices = nearest_indices(np.asarray(mag_ss["times"], dtype=float), target_times)
        positions = np.asarray(mag_ss["data"][mag_indices][:, mag_ss["pos_indices"]], dtype=float)
        radii = np.linalg.norm(positions, axis=1)
        altitude_km = radii - MARS_RADIUS_KM
        cos_sza = np.divide(positions[:, 0], radii, out=np.full_like(radii, np.nan), where=radii > 0.0)
        sza_deg = np.degrees(np.arccos(np.clip(cos_sza, -1.0, 1.0)))

        for local_index, sample_index in enumerate(indices):
            alt = float(altitude_km[local_index])
            sza = float(sza_deg[local_index])
            if not np.isfinite(alt) or not np.isfinite(sza):
                continue
            altitude_code, altitude_label = altitude_class(alt)
            sza_code, sza_label = sza_class(sza)
            context[sample_index] = SpatialContext(
                sample_index=sample_index,
                bin_code=f"{altitude_code}_{sza_code}",
                altitude_code=altitude_code,
                altitude_label=altitude_label,
                sza_code=sza_code,
                sza_label=sza_label,
                altitude_km=alt,
                sza_deg=sza,
                mag_ss_time_utc=format_unix_time(float(mag_ss["times"][mag_indices[local_index]])),
            )

    return [item for item in context if item is not None]


def write_spatial_sample_index(path: Path, samples: list, contexts: list[SpatialContext]) -> None:
    fieldnames = [
        "sample_index",
        "time_utc",
        "bin_code",
        "altitude_class",
        "altitude_label",
        "sza_class",
        "sza_label",
        "altitude_km",
        "sza_deg",
        "source_file",
        "mag_ss_time_utc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for context in contexts:
            sample = samples[context.sample_index]
            writer.writerow(
                {
                    "sample_index": context.sample_index,
                    "time_utc": format_unix_time(sample.time_unix),
                    "bin_code": context.bin_code,
                    "altitude_class": context.altitude_code,
                    "altitude_label": context.altitude_label,
                    "sza_class": context.sza_code,
                    "sza_label": context.sza_label,
                    "altitude_km": context.altitude_km,
                    "sza_deg": context.sza_deg,
                    "source_file": sample.source_file,
                    "mag_ss_time_utc": context.mag_ss_time_utc,
                }
            )


def write_spatial_assignments(path: Path, samples: list, contexts: list[SpatialContext], labels: np.ndarray, max_probabilities: np.ndarray) -> None:
    context_by_index = {context.sample_index: context for context in contexts}
    fieldnames = [
        "time_utc",
        "cluster",
        "max_probability",
        "bin_code",
        "altitude_km",
        "sza_deg",
        "source_file",
        "mag_ss_time_utc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for local_index, sample in enumerate(samples):
            context = context_by_index.get(local_index)
            writer.writerow(
                {
                    "time_utc": format_unix_time(sample.time_unix),
                    "cluster": int(labels[local_index]) + 1,
                    "max_probability": float(max_probabilities[local_index]),
                    "bin_code": context.bin_code if context else "",
                    "altitude_km": context.altitude_km if context else np.nan,
                    "sza_deg": context.sza_deg if context else np.nan,
                    "source_file": sample.source_file,
                    "mag_ss_time_utc": context.mag_ss_time_utc if context else "",
                }
            )


def run_one_spatial_bin(
    bin_dir: Path,
    spatial_bin: SpatialBin,
    energy: np.ndarray,
    samples: list,
    original_indices: np.ndarray,
    contexts: list[SpatialContext],
    args: argparse.Namespace,
) -> dict:
    sample_count = len(samples)
    required_samples = args.components if args.components is not None else args.min_clusters
    if sample_count < required_samples:
        return {
            "bin_code": spatial_bin.code,
            "status": "skipped_too_few_samples",
            "sample_count": sample_count,
            "required_samples": int(required_samples),
        }

    bin_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray([sample.normalized_flux for sample in samples], dtype=float)
    log_step(f"{spatial_bin.code}: derivative feature matrix {matrix.shape[0]} x {matrix.shape[1]} (normalization=none).")

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
    else:
        selected_components = args.components
        model = fit_gmm_diagonal(
            training_scores,
            component_count=selected_components,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            regularization=args.regularization,
        )
        training_labels, training_probabilities, _, training_log_likelihood = predict_gmm_diagonal(training_scores, model)
        bic, aic = information_criteria(training_log_likelihood, training_scores.shape[0], selected_components, training_scores.shape[1])
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
    full_bic, full_aic = information_criteria(full_log_likelihood, all_scores.shape[0], selected_components, all_scores.shape[1])
    rows = write_derivative_pca_gmm_bundle(
        output_dir=bin_dir,
        energy=energy,
        samples=samples,
        pca_scores=all_scores,
        labels=labels,
        probabilities=probabilities,
        max_probabilities=max_probabilities,
        model=model,
        direction=args.direction,
    )
    write_full_assignments(bin_dir / "full_assignments.csv", samples, labels, max_probabilities)
    local_contexts = [
        SpatialContext(
            sample_index=local_index,
            bin_code=context.bin_code,
            altitude_code=context.altitude_code,
            altitude_label=context.altitude_label,
            sza_code=context.sza_code,
            sza_label=context.sza_label,
            altitude_km=context.altitude_km,
            sza_deg=context.sza_deg,
            mag_ss_time_utc=context.mag_ss_time_utc,
        )
        for local_index, context in enumerate(contexts)
    ]
    write_spatial_assignments(bin_dir / "spatial_assignments.csv", samples, local_contexts, labels, max_probabilities)
    write_k_scan_csv(bin_dir / "k_scan_metrics.csv", cluster_selection["trials"])
    plot_k_scan(cluster_selection["trials"], bin_dir)

    np.savez_compressed(
        bin_dir / "pca_model_and_scores.npz",
        scores=all_scores.astype(np.float32),
        mean=np.asarray(pca["mean"], dtype=np.float64),
        components=np.asarray(pca["components"], dtype=np.float64),
        explained_variance=np.asarray(pca["explained_variance"], dtype=np.float64),
        explained_variance_ratio=np.asarray(pca["explained_variance_ratio"], dtype=np.float64),
        training_indices=training_indices.astype(np.int64),
        original_sample_indices=original_indices.astype(np.int64),
    )

    cluster_sizes = [int(np.count_nonzero(labels == index)) for index in range(selected_components)]
    summary = {
        "bin": {
            "code": spatial_bin.code,
            "altitude_class": spatial_bin.altitude_code,
            "altitude_label": spatial_bin.altitude_label,
            "sza_class": spatial_bin.sza_code,
            "sza_label": spatial_bin.sza_label,
        },
        "settings": {
            "normalization": "none",
            "pca_components": args.pca_components,
            "sample_size": args.sample_size,
            "actual_training_sample_count": int(training_scores.shape[0]),
            "random_seed": args.random_seed,
            "direction": args.direction,
            "components": selected_components,
            "auto_clusters": args.components is None,
            "min_clusters": args.min_clusters,
            "max_clusters": args.max_clusters,
            "min_cluster_fraction": args.min_cluster_fraction,
            "parallel_pitch_max_deg": args.parallel_pitch_max,
            "anti_parallel_pitch_min_deg": args.anti_parallel_pitch_min,
            "min_direction_valid_fraction": args.min_direction_valid_fraction,
        },
        "cluster_selection": cluster_selection,
        "gmm": {
            "weights": model["weights"],
            "bic_full": full_bic,
            "aic_full": full_aic,
            "log_likelihood_full": full_log_likelihood,
            "iterations": model["iterations"],
            "mean_max_probability_full": float(np.mean(max_probabilities)),
        },
        "sample_count": sample_count,
        "cluster_sizes": cluster_sizes,
        "energy_eV": energy,
        "pca_explained_variance_ratio": pca["explained_variance_ratio"],
        "pca_cumulative_explained_variance_ratio": pca["cumulative_explained_variance_ratio"],
        "representatives": rows,
        "outputs": {
            "representative_times_csv": str(bin_dir / "representative_times.csv"),
            "characteristic_derivative_spectra_png": str(bin_dir / "characteristic_derivative_spectra.png"),
            "pca_clusters_png": str(bin_dir / "pca_clusters.png"),
            "k_scan_metrics_csv": str(bin_dir / "k_scan_metrics.csv"),
            "full_assignments_csv": str(bin_dir / "full_assignments.csv"),
            "spatial_assignments_csv": str(bin_dir / "spatial_assignments.csv"),
            "predict_proba_full_npz": str(bin_dir / "predict_proba_full.npz"),
            "pca_model_and_scores_npz": str(bin_dir / "pca_model_and_scores.npz"),
        },
    }
    (bin_dir / "derivative_spatial_pca_gmm_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "bin_code": spatial_bin.code,
        "status": "ok",
        "sample_count": sample_count,
        "selected_components": int(selected_components),
        "cluster_sizes": cluster_sizes,
        "output_dir": str(bin_dir),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run derivative PCA-GMM separately for 12 altitude/SZA bins.")
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
    parser.add_argument("--components", type=int, help="Manual GMM component count per spatial bin. If omitted, scan k.")
    parser.add_argument("--clusters", type=int, help="Alias for --components.")
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=6)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.005)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-direction-valid-fraction", type=float, default=0.1)
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
    args.normalization = "none"

    start_dt = parse_iso_datetime(args.start)
    end_dt = parse_iso_datetime(args.end)
    start_unix = parse_iso_time(args.start)
    end_unix = parse_iso_time(args.end)
    if start_unix is not None and end_unix is not None and end_unix < start_unix:
        raise ValueError("--end must be later than --start.")

    data_root = Path(args.data_root).expanduser().resolve()
    data_coverage_report = None
    if args.swe_file:
        log_step("Explicit --swe-file argument was supplied; skipping automatic SWE data coverage check.")
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
    run_name = f"derivative_spatial_pca{args.pca_components}_sample{args.sample_size}_gmm_{build_run_name(args, start_dt, end_dt)}"
    output_dir = unique_output_dir(output_root, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Output directory ready: {output_dir}")

    energy, samples = load_derivative_samples(
        files=files,
        start_unix=start_unix,
        end_unix=end_unix,
        stride=args.stride,
        normalization="none",
        direction=args.direction,
        parallel_pitch_max_deg=args.parallel_pitch_max,
        anti_parallel_pitch_min_deg=args.anti_parallel_pitch_min,
        min_direction_valid_fraction=args.min_direction_valid_fraction,
        lpw_files_by_date=lpw_files_by_date,
        spacecraft_potential_min_flag=DEFAULT_SCPOT_MIN_FLAG,
        min_plasma_energy_eV=DEFAULT_MIN_PLASMA_ENERGY_EV,
    )

    mag_days = sample_days(samples)
    mag_ss_files = resolve_mag_ss_files(mag_days, data_root, auto_download=not args.no_auto_download)
    if data_coverage_report is None:
        data_coverage_report = {}
    data_coverage_report["mag_ss1s"] = {day.isoformat(): str(path) for day, path in sorted(mag_ss_files.items())}

    contexts = attach_spatial_context(samples, mag_ss_files)
    if not contexts:
        raise ValueError("No samples could be assigned to altitude/SZA bins.")
    write_spatial_sample_index(output_dir / "spatial_sample_index.csv", samples, contexts)

    contexts_by_bin: dict[str, list[SpatialContext]] = defaultdict(list)
    for context in contexts:
        contexts_by_bin[context.bin_code].append(context)

    bin_summaries = []
    for spatial_bin in SPATIAL_BINS:
        bin_contexts = contexts_by_bin.get(spatial_bin.code, [])
        indices = np.asarray([context.sample_index for context in bin_contexts], dtype=int)
        bin_samples = [samples[index] for index in indices]
        log_step(f"{spatial_bin.code}: {len(bin_samples)} sample(s) before PCA-GMM.")
        try:
            bin_summary = run_one_spatial_bin(
                bin_dir=output_dir / spatial_bin.label,
                spatial_bin=spatial_bin,
                energy=energy,
                samples=bin_samples,
                original_indices=indices,
                contexts=bin_contexts,
                args=args,
            )
        except Exception as exc:
            bin_summary = {
                "bin_code": spatial_bin.code,
                "status": "failed",
                "sample_count": len(bin_samples),
                "error": f"{type(exc).__name__}: {exc}",
            }
            log_step(f"{spatial_bin.code}: failed ({bin_summary['error']}).")
        bin_summaries.append(bin_summary)

    summary = {
        "settings": {
            "method": "altitude_sza_split_then_derivative_pca_sampled_gmm",
            "start": args.start,
            "end": args.end,
            "normalization": "none",
            "direction": args.direction,
            "feature_meaning": (
                "paired parallel + anti_parallel dFlux/dE spectra per timestamp, no derivative normalization"
                if args.direction == "both"
                else f"{args.direction} dFlux/dE spectrum per timestamp, no derivative normalization"
            ),
            "altitude_bins_km": {
                "A1": "alt < 800",
                "A2": "800 <= alt < 2000",
                "A3": "2000 <= alt < 4500",
                "A4": "alt >= 4500",
            },
            "sza_bins_deg": {
                "S1": "SZA < 90",
                "S2": "90 <= SZA < 110",
                "S3": "SZA >= 110",
            },
            "pca_components": args.pca_components,
            "sample_size": args.sample_size,
            "random_seed": args.random_seed,
            "components": args.components,
            "auto_clusters": args.components is None,
            "min_clusters": args.min_clusters,
            "max_clusters": args.max_clusters,
            "min_cluster_fraction": args.min_cluster_fraction,
            "stride": args.stride,
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
            "data_root": str(data_root),
            "input_files": [str(path) for path in files],
            "run_name": run_name,
            "output_dir": str(output_dir),
        },
        "data_coverage": data_coverage_report,
        "sample_count_total": len(samples),
        "sample_count_with_spatial_context": len(contexts),
        "spatial_bins": bin_summaries,
        "outputs": {
            "spatial_sample_index_csv": str(output_dir / "spatial_sample_index.csv"),
        },
    }
    (output_dir / "derivative_spatial_pca_gmm_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_step(f"Summary JSON written: {output_dir / 'derivative_spatial_pca_gmm_summary.json'}")
    print(f"Loaded {len(samples)} derivative spectra from {len(files)} SWE file(s).")
    print(f"Assigned {len(contexts)} sample(s) to altitude/SZA bins.")
    print(f"Spatial derivative PCA-GMM output written to: {output_dir}")
    for item in bin_summaries:
        print(f"{item['bin_code']}: {item['status']}, n={item['sample_count']}")


if __name__ == "__main__":
    main()
