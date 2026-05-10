from __future__ import annotations
"""
Plot spatial distributions for clustered MAVEN SWE spectra.

Input is a `full_assignments.csv` written by the PCA-GMM scripts. The file only
contains time and cluster labels, so this post-processing step attaches MAVEN
MAG positions:

- MAG sun-state/MSO (`ss1s`) for altitude and SZA.
- MAG planetocentric (`pc1s`) for longitude and latitude.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
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

from analyze_magnetic_topology import MARS_RADIUS_KM, load_mag_day
from download_maven_data import PIPELINE_PRODUCTS, build_session, download_product_for_day
from machine_learning.analyze_electron_spectra_ml import DEFAULT_DATA_ROOT, log_step, sanitize_for_json
from mars_crustal_model import DEFAULT_MODEL_ROOT
from plot_maven_orbit_map import load_or_build_crustal_field_grid, pc_position_to_lon_lat
from process_maven_spectra import format_unix_time, infer_daily_file


ROOT_DATA_FALLBACK = PROJECT_ROOT / "data" / "maven"
DAY_SIDE_SZA_DEG = 90.0


@dataclass(frozen=True)
class AssignmentRow:
    time_unix: float
    time_utc: str
    cluster: int
    max_probability: float
    source_file: str


def parse_time_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_assignments(path: Path, min_probability: float | None = None) -> list[AssignmentRow]:
    rows: list[AssignmentRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            time_dt = parse_time_utc(row["time_utc"])
            probability = float(row.get("max_probability") or np.nan)
            if min_probability is not None and np.isfinite(probability) and probability < min_probability:
                continue
            rows.append(
                AssignmentRow(
                    time_unix=float(time_dt.timestamp()),
                    time_utc=time_dt.isoformat(timespec="seconds"),
                    cluster=int(row["cluster"]),
                    max_probability=probability,
                    source_file=row.get("source_file", ""),
                )
            )
    if not rows:
        raise ValueError(f"No assignment rows were loaded from {path}.")
    return rows


def iter_days_from_times(times_unix: np.ndarray) -> list[date]:
    start = datetime.fromtimestamp(float(np.min(times_unix)), tz=timezone.utc).date()
    end = datetime.fromtimestamp(float(np.max(times_unix)), tz=timezone.utc).date()
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def find_daily_mag_file(search_roots: list[Path], day: date, alias: str) -> Path | None:
    day_dt = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    for root in search_roots:
        try:
            return infer_daily_file(root, "mag", alias, day_dt, "sts")
        except FileNotFoundError:
            continue
    return None


def download_mag_file(data_root: Path, day: date, alias: str) -> Path:
    specs = [
        spec
        for spec in PIPELINE_PRODUCTS
        if spec.instrument == "mag" and any(alias in item for item in spec.aliases)
    ]
    if not specs:
        raise RuntimeError(f"No MAG product specification found for alias={alias}.")
    session = build_session()
    return download_product_for_day(session=session, spec=specs[0], day=day, data_root=data_root)


def resolve_mag_files(
    days: list[date],
    data_root: Path,
    fallback_data_root: Path | None,
    auto_download: bool,
) -> dict[date, dict[str, Path]]:
    search_roots = [data_root]
    if fallback_data_root is not None and fallback_data_root != data_root:
        search_roots.append(fallback_data_root)

    resolved: dict[date, dict[str, Path]] = {}
    for index, day in enumerate(days, start=1):
        log_step(f"Resolving MAG position files {index}/{len(days)}: {day.isoformat()}")
        ss_file = find_daily_mag_file(search_roots, day, "ss1s")
        pc_file = find_daily_mag_file(search_roots, day, "pc1s")
        if (ss_file is None or pc_file is None) and auto_download:
            if ss_file is None:
                log_step(f"Downloading missing MAG ss1s for {day.isoformat()}.")
                ss_file = download_mag_file(data_root, day, "ss1s")
            if pc_file is None:
                log_step(f"Downloading missing MAG pc1s for {day.isoformat()}.")
                pc_file = download_mag_file(data_root, day, "pc1s")
        if ss_file is None or pc_file is None:
            missing = []
            if ss_file is None:
                missing.append("ss1s")
            if pc_file is None:
                missing.append("pc1s")
            raise FileNotFoundError(f"Missing MAG {', '.join(missing)} for {day.isoformat()}.")
        resolved[day] = {"ss": ss_file, "pc": pc_file}
    return resolved


def nearest_indices(sorted_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    insert = np.searchsorted(sorted_times, target_times)
    insert = np.clip(insert, 1, sorted_times.size - 1)
    left = insert - 1
    right = insert
    choose_right = np.abs(sorted_times[right] - target_times) < np.abs(target_times - sorted_times[left])
    return np.where(choose_right, right, left)


def attach_positions(assignments: list[AssignmentRow], mag_files: dict[date, dict[str, Path]]) -> list[dict]:
    by_day: dict[date, list[int]] = defaultdict(list)
    for index, row in enumerate(assignments):
        by_day[datetime.fromtimestamp(row.time_unix, tz=timezone.utc).date()].append(index)

    enriched: list[dict | None] = [None] * len(assignments)
    for day, indices in sorted(by_day.items()):
        log_step(f"Attaching MAG position metadata for {day.isoformat()} ({len(indices)} assignment rows).")
        mag_ss = load_mag_day(mag_files[day]["ss"])
        mag_pc = load_mag_day(mag_files[day]["pc"])
        target_times = np.asarray([assignments[index].time_unix for index in indices], dtype=float)

        ss_indices = nearest_indices(np.asarray(mag_ss["times"], dtype=float), target_times)
        pc_indices = nearest_indices(np.asarray(mag_pc["times"], dtype=float), target_times)
        positions_ss = np.asarray(mag_ss["data"][ss_indices][:, mag_ss["pos_indices"]], dtype=float)
        positions_pc = np.asarray(mag_pc["data"][pc_indices][:, mag_pc["pos_indices"]], dtype=float)

        radii_ss = np.linalg.norm(positions_ss, axis=1)
        altitude_km = radii_ss - MARS_RADIUS_KM
        cos_sza = np.divide(positions_ss[:, 0], radii_ss, out=np.full_like(radii_ss, np.nan), where=radii_ss > 0)
        sza_deg = np.degrees(np.arccos(np.clip(cos_sza, -1.0, 1.0)))

        lon_lat_radius = np.asarray([pc_position_to_lon_lat(position) for position in positions_pc], dtype=float)
        longitudes = lon_lat_radius[:, 0]
        latitudes = lon_lat_radius[:, 1]

        for local_index, assignment_index in enumerate(indices):
            row = assignments[assignment_index]
            enriched[assignment_index] = {
                "time_utc": row.time_utc,
                "time_unix": row.time_unix,
                "cluster": row.cluster,
                "max_probability": row.max_probability,
                "source_file": row.source_file,
                "sza_deg": float(sza_deg[local_index]),
                "altitude_km": float(altitude_km[local_index]),
                "longitude_deg": float(longitudes[local_index]),
                "latitude_deg": float(latitudes[local_index]),
                "day_night": "dayside" if sza_deg[local_index] < DAY_SIDE_SZA_DEG else "nightside",
                "mag_ss_time_utc": format_unix_time(float(mag_ss["times"][ss_indices[local_index]])),
                "mag_pc_time_utc": format_unix_time(float(mag_pc["times"][pc_indices[local_index]])),
            }
    return [item for item in enriched if item is not None]


def write_enriched_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "time_utc",
        "cluster",
        "max_probability",
        "sza_deg",
        "altitude_km",
        "longitude_deg",
        "latitude_deg",
        "day_night",
        "source_file",
        "mag_ss_time_utc",
        "mag_pc_time_utc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cluster_color(cluster: int) -> str:
    cmap = plt.get_cmap("tab20")
    return cmap((cluster - 1) % 20)


def plot_histogram(values: np.ndarray, cluster: int, title: str, xlabel: str, output: Path, bins: np.ndarray | int) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    finite = values[np.isfinite(values)]
    ax.hist(finite, bins=bins, color=cluster_color(cluster), alpha=0.82, edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sample count")
    ax.grid(alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def add_crustal_background(ax, lon_min: float, lon_max: float, args: argparse.Namespace) -> None:
    if args.no_crustal_background:
        ax.set_facecolor("#f2f2f2")
        return
    lon_values, lat_values, field_mag, _, _ = load_or_build_crustal_field_grid(
        lon_min=lon_min,
        lon_max=lon_max,
        altitude_km=args.crustal_altitude_km,
        grid_step_deg=args.grid_step_deg,
        model_root=Path(args.model_root).expanduser().resolve(),
        model_max_degree=args.model_max_degree,
    )
    image = ax.pcolormesh(lon_values, lat_values, field_mag, shading="auto", cmap="Greys", alpha=0.72)
    return image


def plot_cluster_lon_lat(rows: list[dict], cluster: int, output: Path, args: argparse.Namespace) -> None:
    cluster_rows = [row for row in rows if row["cluster"] == cluster]
    lon = np.asarray([row["longitude_deg"] for row in cluster_rows], dtype=float)
    lat = np.asarray([row["latitude_deg"] for row in cluster_rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True, constrained_layout=True)
    scatter_handles = []
    for ax, lon_min, lon_max in [(axes[0], 0.0, 180.0), (axes[1], 180.0, 360.0)]:
        image = add_crustal_background(ax, lon_min, lon_max, args)
        visible = (lon >= lon_min) & (lon <= lon_max)
        handle = ax.scatter(
            lon[visible],
            lat[visible],
            s=args.point_size,
            c=[cluster_color(cluster)],
            alpha=args.point_alpha,
            edgecolors="none",
            label=f"cluster {cluster}",
        )
        scatter_handles.append(handle)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xlabel("Planetocentric longitude (deg)")
        ax.grid(True, linestyle=":", alpha=0.35)
        if image is not None:
            fig.colorbar(image, ax=ax, pad=0.01, label=f"|B| at {args.crustal_altitude_km:g} km (nT)")
    axes[0].set_ylabel("Planetocentric latitude (deg)")
    fig.suptitle(f"Cluster {cluster}: longitude-latitude distribution (n={len(cluster_rows)})")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_all_clusters_lon_lat(rows: list[dict], output: Path, args: argparse.Namespace) -> None:
    clusters = sorted({int(row["cluster"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True, constrained_layout=True)
    for ax, lon_min, lon_max in [(axes[0], 0.0, 180.0), (axes[1], 180.0, 360.0)]:
        image = add_crustal_background(ax, lon_min, lon_max, args)
        for cluster in clusters:
            cluster_rows = [row for row in rows if row["cluster"] == cluster]
            lon = np.asarray([row["longitude_deg"] for row in cluster_rows], dtype=float)
            lat = np.asarray([row["latitude_deg"] for row in cluster_rows], dtype=float)
            visible = (lon >= lon_min) & (lon <= lon_max)
            ax.scatter(
                lon[visible],
                lat[visible],
                s=args.point_size,
                color=cluster_color(cluster),
                alpha=args.point_alpha,
                edgecolors="none",
                label=f"C{cluster}",
            )
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xlabel("Planetocentric longitude (deg)")
        ax.grid(True, linestyle=":", alpha=0.35)
        if image is not None:
            fig.colorbar(image, ax=ax, pad=0.01, label=f"|B| at {args.crustal_altitude_km:g} km (nT)")
    axes[0].set_ylabel("Planetocentric latitude (deg)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
    fig.suptitle("All clusters: longitude-latitude distribution")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize_clusters(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for cluster in sorted({int(row["cluster"]) for row in rows}):
        cluster_rows = [row for row in rows if row["cluster"] == cluster]
        sza = np.asarray([row["sza_deg"] for row in cluster_rows], dtype=float)
        altitude = np.asarray([row["altitude_km"] for row in cluster_rows], dtype=float)
        dayside = sum(1 for row in cluster_rows if row["day_night"] == "dayside")
        nightside = len(cluster_rows) - dayside
        summary.append(
            {
                "cluster": cluster,
                "sample_count": len(cluster_rows),
                "sza_mean_deg": float(np.nanmean(sza)),
                "sza_median_deg": float(np.nanmedian(sza)),
                "altitude_mean_km": float(np.nanmean(altitude)),
                "altitude_median_km": float(np.nanmedian(altitude)),
                "dayside_count": int(dayside),
                "nightside_count": int(nightside),
                "dayside_fraction": float(dayside / len(cluster_rows)) if cluster_rows else 0.0,
            }
        )
    return summary


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "cluster",
        "sample_count",
        "sza_mean_deg",
        "sza_median_deg",
        "altitude_mean_km",
        "altitude_median_km",
        "dayside_count",
        "nightside_count",
        "dayside_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot spatial distributions for GMM cluster assignments.")
    parser.add_argument("assignments_csv", help="Path to full_assignments.csv.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <assignment-run>/spatial_distributions.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Primary MAVEN data root.")
    parser.add_argument("--fallback-data-root", default=str(ROOT_DATA_FALLBACK), help="Fallback MAVEN data root.")
    parser.add_argument("--no-auto-download", action="store_true", help="Do not download missing MAG ss1s/pc1s files.")
    parser.add_argument("--min-probability", type=float, help="Optionally keep only samples above this max_probability.")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--crustal-altitude-km", type=float, default=185.0)
    parser.add_argument("--grid-step-deg", type=float, default=2.0)
    parser.add_argument("--model-max-degree", type=int, default=60)
    parser.add_argument("--no-crustal-background", action="store_true")
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--point-alpha", type=float, default=0.45)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    assignments_path = Path(args.assignments_csv).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else assignments_path.parent / "spatial_distributions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Reading assignments: {assignments_path}")
    assignments = read_assignments(assignments_path, min_probability=args.min_probability)
    times = np.asarray([row.time_unix for row in assignments], dtype=float)
    days = iter_days_from_times(times)
    log_step(f"Loaded {len(assignments)} assignment rows across {len(days)} UTC day(s).")

    mag_files = resolve_mag_files(
        days=days,
        data_root=Path(args.data_root).expanduser().resolve(),
        fallback_data_root=Path(args.fallback_data_root).expanduser().resolve() if args.fallback_data_root else None,
        auto_download=not args.no_auto_download,
    )
    enriched_rows = attach_positions(assignments, mag_files)
    write_enriched_csv(output_dir / "cluster_spatial_assignments.csv", enriched_rows)

    summary_rows = summarize_clusters(enriched_rows)
    write_summary_csv(output_dir / "cluster_spatial_summary.csv", summary_rows)

    sza_bins = np.linspace(0.0, 180.0, 37)
    altitude_values = np.asarray([row["altitude_km"] for row in enriched_rows], dtype=float)
    altitude_min = float(np.nanmin(altitude_values))
    altitude_max = float(np.nanmax(altitude_values))
    altitude_bins = np.linspace(altitude_min, altitude_max, 40) if altitude_max > altitude_min else 30

    clusters = sorted({int(row["cluster"]) for row in enriched_rows})
    for cluster in clusters:
        cluster_rows = [row for row in enriched_rows if row["cluster"] == cluster]
        sza = np.asarray([row["sza_deg"] for row in cluster_rows], dtype=float)
        altitude = np.asarray([row["altitude_km"] for row in cluster_rows], dtype=float)
        log_step(f"Plotting cluster {cluster}: n={len(cluster_rows)}")
        plot_histogram(
            sza,
            cluster,
            title=f"Cluster {cluster}: SZA distribution",
            xlabel="Solar zenith angle (deg)",
            output=output_dir / f"cluster_{cluster:02d}_sza_distribution.png",
            bins=sza_bins,
        )
        plot_histogram(
            altitude,
            cluster,
            title=f"Cluster {cluster}: altitude distribution",
            xlabel="Altitude (km)",
            output=output_dir / f"cluster_{cluster:02d}_altitude_distribution.png",
            bins=altitude_bins,
        )
        plot_cluster_lon_lat(
            enriched_rows,
            cluster,
            output_dir / f"cluster_{cluster:02d}_longitude_latitude_map.png",
            args,
        )

    plot_all_clusters_lon_lat(enriched_rows, output_dir / "all_clusters_longitude_latitude_map.png", args)
    summary = {
        "assignments_csv": str(assignments_path),
        "output_dir": str(output_dir),
        "sample_count": len(enriched_rows),
        "clusters": summary_rows,
        "mag_files": {
            item.isoformat(): {key: str(value) for key, value in files.items()}
            for item, files in mag_files.items()
        },
        "outputs": {
            "cluster_spatial_assignments_csv": str(output_dir / "cluster_spatial_assignments.csv"),
            "cluster_spatial_summary_csv": str(output_dir / "cluster_spatial_summary.csv"),
            "all_clusters_longitude_latitude_map_png": str(output_dir / "all_clusters_longitude_latitude_map.png"),
        },
    }
    (output_dir / "cluster_spatial_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_step(f"Spatial distribution outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
