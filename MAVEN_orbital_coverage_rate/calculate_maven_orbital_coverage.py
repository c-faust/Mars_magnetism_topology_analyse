from __future__ import annotations

"""
Calculate MAVEN orbital spatial coverage in the MSO frame.

The script reuses the project-level MAVEN downloader and MAG STS parser:

1. Resolve local MAG sun-state/MSO 1-second (ss1s) files for every UTC day.
2. Download only missing daily files from LASP unless auto-download is disabled.
3. Select spacecraft positions in the half-open interval [start, end).
4. Bin the positions in Cartesian and spherical MSO grids.
5. Write one per-cell CSV for each grid and one summary CSV.
"""

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from download_maven_data import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    PIPELINE_PRODUCTS,
    build_session,
    download_product_for_day,
    parse_filename,
    parse_iso_timestamp,
)
from process_maven_spectra import build_mag_times, parse_mag_sts  # noqa: E402


MARS_RADIUS_KM = 3389.5
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "MAVEN_orbital_coverage_rate"


@dataclass(frozen=True)
class PositionSeries:
    """Valid MSO position samples selected from daily MAG ss1s files."""

    times_unix: np.ndarray
    positions_rm: np.ndarray
    raw_interval_sample_count: int
    invalid_position_sample_count: int
    source_files: tuple[Path, ...]


@dataclass(frozen=True)
class GridCoverage:
    """N-dimensional grid edges and the number of samples in every cell."""

    coordinate_system: str
    axis_names: tuple[str, str, str]
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    counts: np.ndarray
    valid_sample_count: int
    in_range_sample_count: int

    @property
    def total_cell_count(self) -> int:
        return int(self.counts.size)

    @property
    def covered_cell_count(self) -> int:
        return int(np.count_nonzero(self.counts))

    @property
    def coverage_rate(self) -> float:
        if self.total_cell_count == 0:
            return float("nan")
        return self.covered_cell_count / self.total_cell_count

    @property
    def out_of_range_sample_count(self) -> int:
        return self.valid_sample_count - self.in_range_sample_count


def log(message: str) -> None:
    print(f"[orbital-coverage] {message}", flush=True)


def utc_iso_from_unix(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="milliseconds")


def iter_interval_days(start: datetime, end: datetime) -> Iterable[date]:
    """Yield UTC dates touched by the half-open interval [start, end)."""

    if end <= start:
        raise ValueError("End time must be later than start time.")
    current = start.date()
    final = (end - timedelta(microseconds=1)).date()
    while current <= final:
        yield current
        current += timedelta(days=1)


def _mag_ss1s_spec():
    matches = [
        spec
        for spec in PIPELINE_PRODUCTS
        if spec.instrument == "mag" and any("ss1s" in alias.lower() for alias in spec.aliases)
    ]
    if not matches:
        raise RuntimeError("No MAG ss1s product specification exists in download_maven_data.PIPELINE_PRODUCTS.")
    return matches[0]


def find_local_mag_ss1s(data_root: Path, day: date) -> Path | None:
    """Return the best non-empty local MAG ss1s file for one day."""

    day_code = day.strftime("%Y%m%d")
    mag_root = data_root / "mag"
    if not mag_root.exists():
        return None

    candidates: list[tuple[tuple[int, int, int, str], Path]] = []
    extension_rank = {"sts": 0, "tab": 1}
    aliases = tuple(alias.lower() for alias in _mag_ss1s_spec().aliases)
    for extension in ("sts", "tab"):
        for path in mag_root.rglob(f"mvn_mag_*_{day_code}_*.{extension}"):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            parsed = parse_filename(path.name)
            if parsed is None:
                continue
            description = parsed["description"].lower()
            parsed_day = f"{parsed['year']}{parsed['month']}{parsed['day']}"
            if (
                parsed["instrument"] != "mag"
                or not any(alias in description for alias in aliases)
                or parsed_day != day_code
            ):
                continue
            key = (
                extension_rank[parsed["extension"]],
                -int(parsed["version"]),
                -int(parsed["revision"]),
                path.name,
            )
            candidates.append((key, path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def resolve_mag_ss1s_files(
    start: datetime,
    end: datetime,
    data_root: Path,
    auto_download: bool = True,
) -> list[Path]:
    """Resolve all daily MSO position files, downloading only missing days."""

    days = list(iter_interval_days(start, end))
    resolved: list[Path] = []
    session = None
    spec = _mag_ss1s_spec()

    for index, day in enumerate(days, start=1):
        path = find_local_mag_ss1s(data_root, day)
        if path is not None:
            log(f"MAG ss1s {index}/{len(days)} {day.isoformat()}: local {path.name}")
            resolved.append(path)
            continue

        if not auto_download:
            raise FileNotFoundError(
                f"Missing MAG ss1s data for {day.isoformat()} under {data_root}. "
                "Remove --no-auto-download to download it."
            )

        if session is None:
            session = build_session()
        log(f"MAG ss1s {index}/{len(days)} {day.isoformat()}: missing locally; downloading from LASP")
        path = download_product_for_day(session=session, spec=spec, day=day, data_root=data_root)
        if not path.exists() or path.stat().st_size == 0:
            raise OSError(f"Downloaded MAG file is empty or missing: {path}")
        resolved.append(path)

    return resolved


def load_mso_position_series(
    paths: Sequence[Path],
    start: datetime,
    end: datetime,
) -> PositionSeries:
    """Load and concatenate MSO positions inside [start, end)."""

    time_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    raw_interval_sample_count = 0

    for index, path in enumerate(paths, start=1):
        log(f"Reading MAG position file {index}/{len(paths)}: {path.name}")
        parsed = parse_mag_sts(path)
        columns = parsed["columns"]
        data = np.asarray(parsed["data"], dtype=float)
        try:
            position_indices = [columns.index(name) for name in ("POSN.X", "POSN.Y", "POSN.Z")]
        except ValueError as exc:
            raise KeyError(f"POSN.X/POSN.Y/POSN.Z columns are missing in {path}") from exc

        times = build_mag_times(columns, data)
        selected = (times >= start.timestamp()) & (times < end.timestamp())
        if not np.any(selected):
            continue
        selected_times = np.asarray(times[selected], dtype=float)
        selected_positions_km = np.asarray(data[selected][:, position_indices], dtype=float)
        raw_interval_sample_count += int(selected_times.size)
        time_parts.append(selected_times)
        position_parts.append(selected_positions_km)

    if not time_parts:
        raise ValueError(
            f"No MAG ss1s samples were found in [{start.isoformat()}, {end.isoformat()})."
        )

    times = np.concatenate(time_parts)
    positions_km = np.vstack(position_parts)
    order = np.argsort(times, kind="stable")
    times = times[order]
    positions_km = positions_km[order]

    # A boundary sample can occur in two source files. Keep the first sample at
    # each timestamp so it cannot inflate cell hit counts.
    _, unique_indices = np.unique(times, return_index=True)
    unique_indices.sort()
    times = times[unique_indices]
    positions_km = positions_km[unique_indices]

    radii_km = np.linalg.norm(positions_km, axis=1)
    valid = np.all(np.isfinite(positions_km), axis=1) & np.isfinite(times) & (radii_km > 0.0)
    invalid_position_sample_count = int(np.count_nonzero(~valid))
    times = times[valid]
    positions_rm = positions_km[valid] / MARS_RADIUS_KM

    if times.size == 0:
        raise ValueError("All selected MAG spacecraft positions are invalid.")

    return PositionSeries(
        times_unix=times,
        positions_rm=positions_rm,
        raw_interval_sample_count=raw_interval_sample_count,
        invalid_position_sample_count=invalid_position_sample_count,
        source_files=tuple(paths),
    )


def regular_edges(lower: float, upper: float, cells: int, label: str) -> np.ndarray:
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError(f"{label} maximum must be finite and greater than its minimum.")
    if cells <= 0:
        raise ValueError(f"{label} cell count must be positive.")
    return np.linspace(lower, upper, cells + 1, dtype=float)


def angular_edges(
    lower: float,
    upper: float,
    *,
    cells: int | None,
    delta_degree: float,
    label: str,
) -> np.ndarray:
    """Build angular edges using a cell count or a requested angular step.

    An explicit cell count takes precedence. With delta_degree, a final
    narrower cell is appended when the selected range is not exactly divisible
    by the requested step.
    """

    if cells is not None:
        return regular_edges(lower, upper, cells, label)
    if not math.isfinite(delta_degree) or delta_degree <= 0.0:
        raise ValueError("delta-degree must be finite and positive.")
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError(f"{label} maximum must be finite and greater than its minimum.")

    span = upper - lower
    full_cells = int(math.floor(span / delta_degree + 1e-12))
    edges = lower + np.arange(full_cells + 1, dtype=float) * delta_degree
    tolerance = 1e-10 * max(1.0, abs(lower), abs(upper))
    if edges.size == 0 or edges[-1] < upper - tolerance:
        edges = np.append(edges, upper)
    else:
        edges[-1] = upper
    if edges.size < 2:
        edges = np.asarray([lower, upper], dtype=float)
    return edges


def _histogram_3d(
    points: np.ndarray,
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, int]:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) coordinate array, got shape {values.shape}.")
    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite]
    in_range = np.ones(values.shape[0], dtype=bool)
    for axis in range(3):
        # np.histogramdd includes the rightmost outer edge in the final bin.
        in_range &= (values[:, axis] >= edges[axis][0]) & (values[:, axis] <= edges[axis][-1])
    counts, _ = np.histogramdd(values[in_range], bins=edges)
    return counts.astype(np.int64), int(np.count_nonzero(in_range))


def calculate_cartesian_coverage(
    positions_rm: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    cells: tuple[int, int, int],
) -> GridCoverage:
    edges = (
        regular_edges(*x_range, cells[0], "X"),
        regular_edges(*y_range, cells[1], "Y"),
        regular_edges(*z_range, cells[2], "Z"),
    )
    counts, in_range_count = _histogram_3d(positions_rm, edges)
    return GridCoverage(
        coordinate_system="cartesian_mso",
        axis_names=("x_rm", "y_rm", "z_rm"),
        edges=edges,
        counts=counts,
        valid_sample_count=int(len(positions_rm)),
        in_range_sample_count=in_range_count,
    )


def mso_cartesian_to_spherical(positions_rm: np.ndarray, longitude_min_deg: float = -180.0) -> np.ndarray:
    """Convert MSO X/Y/Z to altitude, longitude and latitude.

    Altitude is in Mars radii above the mean surface. Longitude is measured
    from +X toward +Y and wrapped into [longitude_min_deg, longitude_min_deg +
    360). Latitude is positive toward +Z.
    """

    positions = np.asarray(positions_rm, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) position array, got shape {positions.shape}.")
    radius_rm = np.linalg.norm(positions, axis=1)
    altitude_rm = radius_rm - 1.0
    longitude_raw = np.degrees(np.arctan2(positions[:, 1], positions[:, 0]))
    longitude_deg = ((longitude_raw - longitude_min_deg) % 360.0) + longitude_min_deg
    sine_latitude = np.divide(
        positions[:, 2],
        radius_rm,
        out=np.full_like(radius_rm, np.nan),
        where=radius_rm > 0.0,
    )
    latitude_deg = np.degrees(np.arcsin(np.clip(sine_latitude, -1.0, 1.0)))
    return np.column_stack((altitude_rm, latitude_deg, longitude_deg))


def calculate_spherical_coverage(
    positions_rm: np.ndarray,
    altitude_range_rm: tuple[float, float],
    altitude_cells: int,
    latitude_range_deg: tuple[float, float],
    longitude_range_deg: tuple[float, float],
    *,
    latitude_cells: int | None,
    longitude_cells: int | None,
    delta_degree: float,
) -> GridCoverage:
    altitude_min, altitude_max = altitude_range_rm
    latitude_min, latitude_max = latitude_range_deg
    longitude_min, longitude_max = longitude_range_deg
    if altitude_min < 0.0:
        raise ValueError("Spherical altitude minimum must be >= 0 R_MARS.")
    if latitude_min < -90.0 or latitude_max > 90.0:
        raise ValueError("Spherical latitude range must stay within [-90, 90] degrees.")
    if longitude_max - longitude_min > 360.0:
        raise ValueError("Spherical longitude range cannot span more than 360 degrees.")

    edges = (
        regular_edges(altitude_min, altitude_max, altitude_cells, "altitude"),
        angular_edges(
            latitude_min,
            latitude_max,
            cells=latitude_cells,
            delta_degree=delta_degree,
            label="latitude",
        ),
        angular_edges(
            longitude_min,
            longitude_max,
            cells=longitude_cells,
            delta_degree=delta_degree,
            label="longitude",
        ),
    )
    spherical = mso_cartesian_to_spherical(positions_rm, longitude_min_deg=longitude_min)
    counts, in_range_count = _histogram_3d(spherical, edges)
    return GridCoverage(
        coordinate_system="spherical_mso",
        axis_names=("altitude_rm", "latitude_deg", "longitude_deg"),
        edges=edges,
        counts=counts,
        valid_sample_count=int(len(positions_rm)),
        in_range_sample_count=in_range_count,
    )


def _fraction_percent(count: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * count / denominator


def write_cartesian_csv(path: Path, coverage: GridCoverage) -> None:
    x_edges, y_edges, z_edges = coverage.edges
    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    dz = np.diff(z_edges)
    fields = [
        "x_index",
        "y_index",
        "z_index",
        "x_min_rm",
        "x_max_rm",
        "x_center_rm",
        "y_min_rm",
        "y_max_rm",
        "y_center_rm",
        "z_min_rm",
        "z_max_rm",
        "z_center_rm",
        "cell_volume_rm3",
        "sample_count",
        "fraction_of_in_range_samples_percent",
        "covered",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ix, iy, iz in np.ndindex(coverage.counts.shape):
            count = int(coverage.counts[ix, iy, iz])
            writer.writerow(
                {
                    "x_index": ix,
                    "y_index": iy,
                    "z_index": iz,
                    "x_min_rm": x_edges[ix],
                    "x_max_rm": x_edges[ix + 1],
                    "x_center_rm": 0.5 * (x_edges[ix] + x_edges[ix + 1]),
                    "y_min_rm": y_edges[iy],
                    "y_max_rm": y_edges[iy + 1],
                    "y_center_rm": 0.5 * (y_edges[iy] + y_edges[iy + 1]),
                    "z_min_rm": z_edges[iz],
                    "z_max_rm": z_edges[iz + 1],
                    "z_center_rm": 0.5 * (z_edges[iz] + z_edges[iz + 1]),
                    "cell_volume_rm3": dx[ix] * dy[iy] * dz[iz],
                    "sample_count": count,
                    "fraction_of_in_range_samples_percent": _fraction_percent(
                        count, coverage.in_range_sample_count
                    ),
                    "covered": int(count > 0),
                }
            )


def _spherical_cell_volume_rm3(
    altitude_lower_rm: float,
    altitude_upper_rm: float,
    latitude_lower_deg: float,
    latitude_upper_deg: float,
    longitude_lower_deg: float,
    longitude_upper_deg: float,
) -> float:
    radial_term = ((1.0 + altitude_upper_rm) ** 3 - (1.0 + altitude_lower_rm) ** 3) / 3.0
    latitude_term = math.sin(math.radians(latitude_upper_deg)) - math.sin(
        math.radians(latitude_lower_deg)
    )
    longitude_term = math.radians(longitude_upper_deg - longitude_lower_deg)
    return radial_term * latitude_term * longitude_term


def write_spherical_csv(path: Path, coverage: GridCoverage) -> None:
    altitude_edges, latitude_edges, longitude_edges = coverage.edges
    fields = [
        "altitude_index",
        "latitude_index",
        "longitude_index",
        "altitude_min_rm",
        "altitude_max_rm",
        "altitude_center_rm",
        "latitude_min_deg",
        "latitude_max_deg",
        "latitude_center_deg",
        "longitude_min_deg",
        "longitude_max_deg",
        "longitude_center_deg",
        "cell_volume_rm3",
        "sample_count",
        "fraction_of_in_range_samples_percent",
        "covered",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ia, ilat, ilon in np.ndindex(coverage.counts.shape):
            count = int(coverage.counts[ia, ilat, ilon])
            writer.writerow(
                {
                    "altitude_index": ia,
                    "latitude_index": ilat,
                    "longitude_index": ilon,
                    "altitude_min_rm": altitude_edges[ia],
                    "altitude_max_rm": altitude_edges[ia + 1],
                    "altitude_center_rm": 0.5 * (altitude_edges[ia] + altitude_edges[ia + 1]),
                    "latitude_min_deg": latitude_edges[ilat],
                    "latitude_max_deg": latitude_edges[ilat + 1],
                    "latitude_center_deg": 0.5 * (latitude_edges[ilat] + latitude_edges[ilat + 1]),
                    "longitude_min_deg": longitude_edges[ilon],
                    "longitude_max_deg": longitude_edges[ilon + 1],
                    "longitude_center_deg": 0.5 * (longitude_edges[ilon] + longitude_edges[ilon + 1]),
                    "cell_volume_rm3": _spherical_cell_volume_rm3(
                        altitude_edges[ia],
                        altitude_edges[ia + 1],
                        latitude_edges[ilat],
                        latitude_edges[ilat + 1],
                        longitude_edges[ilon],
                        longitude_edges[ilon + 1],
                    ),
                    "sample_count": count,
                    "fraction_of_in_range_samples_percent": _fraction_percent(
                        count, coverage.in_range_sample_count
                    ),
                    "covered": int(count > 0),
                }
            )


def _median_sample_interval(times_unix: np.ndarray) -> float:
    differences = np.diff(np.asarray(times_unix, dtype=float))
    positive = differences[np.isfinite(differences) & (differences > 0.0)]
    if positive.size == 0:
        return float("nan")
    return float(np.median(positive))


def _summary_row(
    coverage: GridCoverage,
    output_csv: Path,
    series: PositionSeries,
    start: datetime,
    end: datetime,
) -> dict[str, object]:
    edge_1, edge_2, edge_3 = coverage.edges
    return {
        "coordinate_system": coverage.coordinate_system,
        "start_utc_inclusive": start.isoformat(),
        "end_utc_exclusive": end.isoformat(),
        "interval_duration_seconds": (end - start).total_seconds(),
        "mars_radius_km": MARS_RADIUS_KM,
        "source_file_count": len(series.source_files),
        "raw_interval_sample_count": series.raw_interval_sample_count,
        "duplicate_timestamp_count": (
            series.raw_interval_sample_count
            - series.invalid_position_sample_count
            - len(series.times_unix)
        ),
        "invalid_position_sample_count": series.invalid_position_sample_count,
        "valid_position_sample_count": len(series.times_unix),
        "first_valid_sample_utc": utc_iso_from_unix(series.times_unix[0]),
        "last_valid_sample_utc": utc_iso_from_unix(series.times_unix[-1]),
        "median_sample_interval_seconds": _median_sample_interval(series.times_unix),
        "axis_1_name": coverage.axis_names[0],
        "axis_1_min": edge_1[0],
        "axis_1_max": edge_1[-1],
        "axis_1_cell_count": len(edge_1) - 1,
        "axis_2_name": coverage.axis_names[1],
        "axis_2_min": edge_2[0],
        "axis_2_max": edge_2[-1],
        "axis_2_cell_count": len(edge_2) - 1,
        "axis_3_name": coverage.axis_names[2],
        "axis_3_min": edge_3[0],
        "axis_3_max": edge_3[-1],
        "axis_3_cell_count": len(edge_3) - 1,
        "total_cell_count": coverage.total_cell_count,
        "covered_cell_count": coverage.covered_cell_count,
        "coverage_rate": coverage.coverage_rate,
        "coverage_percent": 100.0 * coverage.coverage_rate,
        "in_range_sample_count": coverage.in_range_sample_count,
        "out_of_range_valid_sample_count": coverage.out_of_range_sample_count,
        "grid_csv": str(output_csv),
    }


def write_summary_csv(
    path: Path,
    coverages_and_paths: Sequence[tuple[GridCoverage, Path]],
    series: PositionSeries,
    start: datetime,
    end: datetime,
) -> None:
    rows = [
        _summary_row(coverage, output_csv, series, start, end)
        for coverage, output_csv in coverages_and_paths
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_time_tag(value: datetime) -> str:
    return value.strftime("%Y%m%dT%H%M%S")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate MAVEN orbital spatial coverage in Cartesian and spherical MSO grids. "
            "The interval is [START, END)."
        )
    )
    parser.add_argument("--start", required=True, help="Inclusive UTC start, e.g. 2016-10-06T18:00:00Z.")
    parser.add_argument("--end", required=True, help="Exclusive UTC end, e.g. 2016-10-07T00:00:00Z.")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Local MAVEN data root. Missing MAG ss1s days are downloaded here.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory. Default: outputs/MAVEN_orbital_coverage_rate/"
            "<start>_<end>."
        ),
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Fail when a daily MAG ss1s file is absent instead of downloading it.",
    )

    cartesian = parser.add_argument_group("Cartesian MSO grid")
    cartesian.add_argument(
        "--cartesian-range",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        default=(-3.0, 3.0, -3.0, 3.0, -3.0, 3.0),
        help="Cartesian limits in R_MARS (default: -3 3 -3 3 -3 3).",
    )
    cartesian.add_argument(
        "--cartesian-bins",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        default=(60, 60, 60),
        help="Cartesian cell counts (default: 60 60 60).",
    )

    spherical = parser.add_argument_group("Spherical MSO grid")
    spherical.add_argument(
        "--altitude-range",
        nargs=2,
        type=float,
        metavar=("HMIN", "HMAX"),
        default=(0.0, 2.0),
        help="Altitude limits above the Mars surface in R_MARS (default: 0 2).",
    )
    spherical.add_argument(
        "--altitude-bins",
        type=int,
        default=20,
        help="Altitude cell count (default: 20).",
    )
    spherical.add_argument(
        "--latitude-range",
        nargs=2,
        type=float,
        metavar=("LATMIN", "LATMAX"),
        default=(-90.0, 90.0),
        help="MSO latitude limits in degrees (default: -90 90).",
    )
    spherical.add_argument(
        "--longitude-range",
        nargs=2,
        type=float,
        metavar=("LONMIN", "LONMAX"),
        default=(-180.0, 180.0),
        help=(
            "MSO longitude limits in degrees, with span <= 360 "
            "(default: -180 180). Ranges such as 0 360 are supported."
        ),
    )
    spherical.add_argument(
        "--delta-degree",
        type=float,
        default=5.0,
        help=(
            "Latitude/longitude cell step in degrees (default: 5). "
            "Used for an angular axis unless its explicit bin count is supplied."
        ),
    )
    spherical.add_argument(
        "--latitude-bins",
        type=int,
        help="Optional latitude cell count; overrides --delta-degree for latitude.",
    )
    spherical.add_argument(
        "--longitude-bins",
        type=int,
        help="Optional longitude cell count; overrides --delta-degree for longitude.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    start = parse_iso_timestamp(args.start)
    end = parse_iso_timestamp(args.end)
    if end <= start:
        raise ValueError("--end must be later than --start.")

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / f"{safe_time_tag(start)}_{safe_time_tag(end)}"
    )

    cartesian_range = tuple(float(value) for value in args.cartesian_range)
    cartesian_cells = tuple(int(value) for value in args.cartesian_bins)
    altitude_range = tuple(float(value) for value in args.altitude_range)
    latitude_range = tuple(float(value) for value in args.latitude_range)
    longitude_range = tuple(float(value) for value in args.longitude_range)

    paths = resolve_mag_ss1s_files(
        start=start,
        end=end,
        data_root=data_root,
        auto_download=not args.no_auto_download,
    )
    series = load_mso_position_series(paths, start=start, end=end)
    log(
        f"Selected {len(series.times_unix)} valid MSO position samples "
        f"from {len(series.source_files)} daily file(s)."
    )

    cartesian = calculate_cartesian_coverage(
        series.positions_rm,
        x_range=(cartesian_range[0], cartesian_range[1]),
        y_range=(cartesian_range[2], cartesian_range[3]),
        z_range=(cartesian_range[4], cartesian_range[5]),
        cells=cartesian_cells,
    )
    spherical = calculate_spherical_coverage(
        series.positions_rm,
        altitude_range_rm=(altitude_range[0], altitude_range[1]),
        altitude_cells=int(args.altitude_bins),
        latitude_range_deg=(latitude_range[0], latitude_range[1]),
        longitude_range_deg=(longitude_range[0], longitude_range[1]),
        latitude_cells=args.latitude_bins,
        longitude_cells=args.longitude_bins,
        delta_degree=float(args.delta_degree),
    )

    cartesian_path = output_dir / "cartesian_mso_coverage.csv"
    spherical_path = output_dir / "spherical_mso_coverage.csv"
    summary_path = output_dir / "coverage_summary.csv"
    log(f"Writing {cartesian.total_cell_count} Cartesian grid rows.")
    write_cartesian_csv(cartesian_path, cartesian)
    log(f"Writing {spherical.total_cell_count} spherical grid rows.")
    write_spherical_csv(spherical_path, spherical)
    write_summary_csv(
        summary_path,
        [(cartesian, cartesian_path), (spherical, spherical_path)],
        series,
        start,
        end,
    )

    log(
        f"Cartesian coverage: {cartesian.covered_cell_count}/{cartesian.total_cell_count} "
        f"= {100.0 * cartesian.coverage_rate:.6f}%"
    )
    log(
        f"Spherical coverage: {spherical.covered_cell_count}/{spherical.total_cell_count} "
        f"= {100.0 * spherical.coverage_rate:.6f}%"
    )
    log(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
