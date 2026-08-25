from __future__ import annotations
"""
Plot MAVEN orbit maps in planetocentric or MSO coordinates.

The plot uses MAG planetocentric (PC) positions for the spacecraft ground
track. The background is the Morschhauser et al. (2014) crustal model evaluated
on a latitude/longitude grid at a configurable altitude.

The optional MSO projection mode uses MAG sunstate (SS) positions and plots a
requested time interval in the XY, XZ, and YZ planes. Mars is drawn to scale;
the XY and XZ panels also show the configured bow shock and the Vignes et al.
(2000) average MPB.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from matplotlib.patches import Circle
from scipy.special import gammaln, lpmv

from analyze_magnetic_topology import load_mag_day
from bow_shock.models import (
    DEFAULT_MODEL_NAME,
    MARS_RADIUS_KM as MSO_MARS_RADIUS_KM,
    BowShockModel,
    get_model,
)
from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from mars_crustal_model import (
    DEFAULT_MODEL_ROOT,
    MARS_REFERENCE_RADIUS_KM,
    SphericalHarmonicCoefficients,
    ensure_morschhauser_coefficients,
    load_morschhauser_coefficients,
    mars_body_to_icrf_matrix,
)
from process_maven_spectra import infer_daily_file, locate_nearest_index
from region_id.classify_region_id import VIGNES_2000_MPB


def format_cache_number(value: float | int | None) -> str:
    if value is None:
        return "full"
    return f"{float(value):.3f}".replace("-", "m").replace(".", "p")


def crustal_cache_path(
    model_root: Path,
    lon_min: float,
    lon_max: float,
    altitude_km: float,
    grid_step_deg: float,
    model_max_degree: int | None,
) -> Path:
    cache_dir = model_root / "precomputed"
    degree_label = "full" if model_max_degree is None else f"deg{int(model_max_degree)}"
    filename = (
        "morschhauser2014_"
        f"alt{format_cache_number(altitude_km)}km_"
        f"step{format_cache_number(grid_step_deg)}deg_"
        f"{degree_label}_"
        f"lon{int(lon_min):03d}_{int(lon_max):03d}.npz"
    )
    return cache_dir / filename


def pc_position_to_lon_lat(position_pc_km: np.ndarray) -> tuple[float, float, float]:
    x, y, z = np.asarray(position_pc_km, dtype=float)
    radius = float(np.linalg.norm([x, y, z]))
    lon = float(np.degrees(np.arctan2(y, x)) % 360.0)
    lat = float(np.degrees(np.arcsin(np.clip(z / max(radius, 1e-9), -1.0, 1.0))))
    return lon, lat, radius


def choose_longitude_window(center_lon_deg: float) -> tuple[float, float]:
    return (0.0, 180.0) if center_lon_deg < 180.0 else (180.0, 360.0)


def wrap_longitudes_to_window(longitudes: np.ndarray, lon_min: float, lon_max: float) -> np.ndarray:
    lon = np.asarray(longitudes, dtype=float) % 360.0
    if lon_min == 0.0 and lon_max == 180.0:
        return lon
    return np.where(lon < lon_min, lon + 360.0, lon)


def sun_direction_pc(unix_seconds: float) -> np.ndarray:
    time = Time(unix_seconds, format="unix", scale="utc")
    sun_pos, _ = get_body_barycentric_posvel("sun", time)
    mars_pos, _ = get_body_barycentric_posvel("mars", time)
    mars_to_sun_icrf = (sun_pos.xyz - mars_pos.xyz).to_value("km")
    body_to_icrf = mars_body_to_icrf_matrix(unix_seconds)
    mars_to_sun_pc = body_to_icrf.T @ mars_to_sun_icrf
    return mars_to_sun_pc / np.linalg.norm(mars_to_sun_pc)


def surface_unit_vectors(lon_grid_deg: np.ndarray, lat_grid_deg: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(lon_grid_deg)
    lat = np.deg2rad(lat_grid_deg)
    return np.stack(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ],
        axis=-1,
    )


def truncate_coefficients(
    coeffs: SphericalHarmonicCoefficients,
    max_degree: int | None,
) -> SphericalHarmonicCoefficients:
    if max_degree is None or max_degree >= coeffs.max_degree:
        return coeffs
    mask = coeffs.degree <= max_degree
    return SphericalHarmonicCoefficients(
        degree=coeffs.degree[mask],
        order=coeffs.order[mask],
        g=coeffs.g[mask],
        h=coeffs.h[mask],
        max_degree=int(max_degree),
    )


def crustal_field_magnitude_grid(
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    altitude_km: float,
    model_root: Path,
    model_max_degree: int | None = 60,
) -> np.ndarray:
    coefficient_path = ensure_morschhauser_coefficients(model_root)
    coeffs = truncate_coefficients(load_morschhauser_coefficients(str(coefficient_path)), model_max_degree)
    radius = MARS_REFERENCE_RADIUS_KM + altitude_km
    theta = np.deg2rad(90.0 - np.asarray(lat_values, dtype=float))[:, None]
    phi = np.deg2rad(np.asarray(lon_values, dtype=float) % 360.0)[None, :]
    cos_theta = np.cos(theta)
    sin_theta = np.maximum(np.sin(theta), 1e-10)
    radial_factor_base = MARS_REFERENCE_RADIUS_KM / radius

    br = np.zeros((len(lat_values), len(lon_values)), dtype=float)
    btheta = np.zeros_like(br)
    bphi = np.zeros_like(br)

    for n_value, m_value, g_value, h_value in zip(coeffs.degree, coeffs.order, coeffs.g, coeffs.h):
        n = int(n_value)
        m = int(m_value)
        log_ratio = gammaln(n - m + 1) - gammaln(n + m + 1)
        schmidt = np.sqrt((2.0 - (1.0 if m == 0 else 0.0)) * np.exp(log_ratio))
        p_nm = ((-1) ** m) * schmidt * lpmv(m, n, cos_theta)
        if n == m:
            p_n1m = np.zeros_like(p_nm)
        else:
            prev_log_ratio = gammaln(n - m) - gammaln(n + m)
            prev_schmidt = np.sqrt((2.0 - (1.0 if m == 0 else 0.0)) * np.exp(prev_log_ratio))
            p_n1m = ((-1) ** m) * prev_schmidt * lpmv(m, n - 1, cos_theta)
        dp_dtheta = (n * cos_theta * p_nm - (n + m) * p_n1m) / sin_theta
        p_nm = np.nan_to_num(p_nm, nan=0.0, posinf=0.0, neginf=0.0)
        dp_dtheta = np.nan_to_num(dp_dtheta, nan=0.0, posinf=0.0, neginf=0.0)

        cos_mphi = np.cos(m * phi)
        sin_mphi = np.sin(m * phi)
        common = g_value * cos_mphi + h_value * sin_mphi
        radial_factor = radial_factor_base ** (n + 2)
        br += (n + 1) * radial_factor * p_nm * common
        btheta -= radial_factor * dp_dtheta * common
        if m > 0:
            bphi += radial_factor * m * (-g_value * sin_mphi + h_value * cos_mphi) * p_nm / sin_theta

    return np.sqrt(br * br + btheta * btheta + bphi * bphi)


def load_or_build_crustal_field_grid(
    lon_min: float,
    lon_max: float,
    altitude_km: float,
    grid_step_deg: float,
    model_root: Path,
    model_max_degree: int | None = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, bool]:
    lon_values = np.arange(lon_min, lon_max + grid_step_deg * 0.5, grid_step_deg, dtype=float)
    lat_values = np.arange(-90.0, 90.0 + grid_step_deg * 0.5, grid_step_deg, dtype=float)
    cache_path = crustal_cache_path(model_root, lon_min, lon_max, altitude_km, grid_step_deg, model_max_degree)

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_lon = np.asarray(cached["lon_values"], dtype=float)
                cached_lat = np.asarray(cached["lat_values"], dtype=float)
                field_mag = np.asarray(cached["field_mag_nT"], dtype=float)
            if (
                np.array_equal(cached_lon, lon_values)
                and np.array_equal(cached_lat, lat_values)
                and field_mag.shape == (len(lat_values), len(lon_values))
            ):
                return cached_lon, cached_lat, field_mag, cache_path, True
        except Exception:
            pass

    field_mag = crustal_field_magnitude_grid(
        lon_values,
        lat_values,
        altitude_km,
        model_root,
        model_max_degree=model_max_degree,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        lon_values=lon_values,
        lat_values=lat_values,
        field_mag_nT=field_mag,
        altitude_km=np.asarray([altitude_km], dtype=float),
        grid_step_deg=np.asarray([grid_step_deg], dtype=float),
        model_max_degree=np.asarray([-1 if model_max_degree is None else model_max_degree], dtype=int),
    )
    return lon_values, lat_values, field_mag, cache_path, False


def precompute_crustal_field_grids(
    model_root: Path,
    altitude_km: float = 185.0,
    grid_step_deg: float = 2.0,
    model_max_degree: int | None = 60,
) -> list[dict]:
    results = []
    for lon_min, lon_max in [(0.0, 180.0), (180.0, 360.0)]:
        lon_values, lat_values, field_mag, cache_path, cache_hit = load_or_build_crustal_field_grid(
            lon_min=lon_min,
            lon_max=lon_max,
            altitude_km=altitude_km,
            grid_step_deg=grid_step_deg,
            model_root=model_root,
            model_max_degree=model_max_degree,
        )
        results.append(
            {
                "longitude_window_deg": [lon_min, lon_max],
                "crustal_altitude_km": altitude_km,
                "grid_step_deg": grid_step_deg,
                "model_max_degree": model_max_degree,
                "shape": list(field_mag.shape),
                "lon_samples": int(len(lon_values)),
                "lat_samples": int(len(lat_values)),
                "cache_file": str(cache_path),
                "cache_hit": cache_hit,
            }
        )
    return results


def iter_utc_dates(start_time: datetime, end_time: datetime):
    """Yield every UTC calendar date touched by an inclusive time interval."""
    current = start_time.astimezone(timezone.utc).date()
    last = end_time.astimezone(timezone.utc).date()
    while current <= last:
        yield current
        current += timedelta(days=1)


def resolve_interval_mag_files(
    start_time: datetime,
    end_time: datetime,
    data_root: Path,
    product_alias: str,
    explicit_files: list[Path] | tuple[Path, ...] | None = None,
) -> list[Path]:
    """Resolve all daily MAG position files required by an interval."""
    if explicit_files:
        paths = [Path(path).expanduser().resolve() for path in explicit_files]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Explicit MAG file(s) do not exist: " + ", ".join(missing)
            )
        return list(dict.fromkeys(paths))

    paths: list[Path] = []
    for day in iter_utc_dates(start_time, end_time):
        day_time = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        paths.append(
            infer_daily_file(
                data_root,
                "mag",
                product_alias,
                day_time,
                "sts",
            )
        )
    return list(dict.fromkeys(paths))


def resolve_mso_mag_files(
    start_time: datetime,
    end_time: datetime,
    data_root: Path,
    explicit_files: list[Path] | tuple[Path, ...] | None = None,
) -> list[Path]:
    return resolve_interval_mag_files(
        start_time,
        end_time,
        data_root,
        "ss1s",
        explicit_files,
    )


def resolve_pc_mag_files(
    start_time: datetime,
    end_time: datetime,
    data_root: Path,
    explicit_files: list[Path] | tuple[Path, ...] | None = None,
) -> list[Path]:
    return resolve_interval_mag_files(
        start_time,
        end_time,
        data_root,
        "pc1s",
        explicit_files,
    )


def load_mag_position_trajectory(
    start_time: datetime,
    end_time: datetime,
    mag_files: list[Path] | tuple[Path, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Load, time-select, sort, and deduplicate MAG position samples."""
    if end_time <= start_time:
        raise ValueError("MSO projection end time must be later than start time.")

    time_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    for path in mag_files:
        mag_data = load_mag_day(Path(path))
        times = np.asarray(mag_data["times"], dtype=float)
        positions = np.asarray(
            mag_data["data"][:, mag_data["pos_indices"]],
            dtype=float,
        )
        selected = (
            (times >= start_time.timestamp())
            & (times <= end_time.timestamp())
            & np.all(np.isfinite(positions), axis=1)
        )
        if np.any(selected):
            time_parts.append(times[selected])
            position_parts.append(positions[selected])

    if not time_parts:
        raise ValueError(
            "No valid MAG position samples were found in the requested interval."
        )

    times = np.concatenate(time_parts)
    positions = np.concatenate(position_parts, axis=0)
    order = np.argsort(times, kind="stable")
    times = times[order]
    positions = positions[order]
    _, unique_indices = np.unique(times, return_index=True)
    unique_indices.sort()
    return times[unique_indices], positions[unique_indices]


def load_mso_trajectory(
    start_time: datetime,
    end_time: datetime,
    mag_ss_files: list[Path] | tuple[Path, ...],
) -> tuple[np.ndarray, np.ndarray]:
    return load_mag_position_trajectory(start_time, end_time, mag_ss_files)


def boundary_plane_curves(
    model: BowShockModel,
    plane: str,
    x_min_rm: float,
    x_max_rm: float,
    sample_count: int = 600,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the two branches of a model's XY or XZ plane intersection."""
    if plane not in {"xy", "xz"}:
        raise ValueError("Boundary curves are supported only for XY and XZ planes.")
    if sample_count < 2:
        raise ValueError("boundary sample_count must be at least 2.")
    maximum = min(float(x_max_rm), float(model.nose_x_rm()))
    if x_min_rm >= maximum:
        raise ValueError("boundary x_min_rm must be smaller than the model nose.")

    x_values = np.linspace(float(x_min_rm), maximum, int(sample_count))
    azimuths = (0.0, np.pi) if plane == "xy" else (np.pi / 2.0, 3.0 * np.pi / 2.0)
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for azimuth in azimuths:
        radius = np.asarray(model.rho_at_x_azimuth(x_values, azimuth), dtype=float)
        transverse = (
            radius * np.cos(azimuth)
            if plane == "xy"
            else radius * np.sin(azimuth)
        )
        curves.append((x_values, transverse))
    return curves


def _finite_plot_extent(
    first_values: np.ndarray,
    second_values: np.ndarray,
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    first_parts = [np.asarray(first_values, dtype=float).reshape(-1)]
    second_parts = [np.asarray(second_values, dtype=float).reshape(-1)]
    for curve_first, curve_second in curves:
        first_parts.append(np.asarray(curve_first, dtype=float).reshape(-1))
        second_parts.append(np.asarray(curve_second, dtype=float).reshape(-1))
    first = np.concatenate(first_parts + [np.asarray([-1.0, 1.0])])
    second = np.concatenate(second_parts + [np.asarray([-1.0, 1.0])])
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    first_span = max(float(np.ptp(first)), 0.5)
    second_span = max(float(np.ptp(second)), 0.5)
    return (
        (float(np.min(first) - 0.08 * first_span), float(np.max(first) + 0.08 * first_span)),
        (float(np.min(second) - 0.08 * second_span), float(np.max(second) + 0.08 * second_span)),
    )


def _draw_mars_disc(ax) -> None:
    mars = Circle(
        (0.0, 0.0),
        1.0,
        facecolor="#b85c38",
        edgecolor="#5f2c1d",
        linewidth=1.2,
        alpha=0.88,
        zorder=1,
        label="Mars",
    )
    ax.add_patch(mars)


TRAJECTORY_MARKERS_BY_LINESTYLE = {
    "-": "o",
    "--": "s",
    "-.": "D",
    ":": "^",
}


def select_trajectory_markers(
    times: np.ndarray,
    trajectory_markers: tuple[tuple[datetime, str, str], ...]
    | list[tuple[datetime, str, str]],
) -> list[dict]:
    """Match requested marker times to the nearest available trajectory samples."""
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return []

    selected: list[dict] = []
    for requested_time, linestyle, color in trajectory_markers:
        requested_unix = requested_time.timestamp()
        if requested_unix < times[0] or requested_unix > times[-1]:
            continue
        index = int(np.argmin(np.abs(times - requested_unix)))
        actual_time = datetime.fromtimestamp(float(times[index]), tz=timezone.utc)
        selected.append(
            {
                "index": index,
                "requested_time": requested_time.astimezone(timezone.utc),
                "actual_time": actual_time,
                "delta_seconds": float(times[index] - requested_unix),
                "linestyle": linestyle,
                "color": color,
                "marker": TRAJECTORY_MARKERS_BY_LINESTYLE.get(linestyle, "o"),
            }
        )
    return selected


def serialize_trajectory_markers(markers: list[dict]) -> list[dict]:
    return [
        {
            "requested_time": marker["requested_time"].isoformat(timespec="seconds"),
            "actual_time": marker["actual_time"].isoformat(timespec="seconds"),
            "delta_seconds": marker["delta_seconds"],
            "linestyle": marker["linestyle"],
            "color": marker["color"],
            "marker": marker["marker"],
        }
        for marker in markers
    ]


def plot_mso_orbit_projections(
    start_time: datetime,
    end_time: datetime,
    mag_ss_files: list[Path] | tuple[Path, ...],
    output_path: Path,
    bow_model_name: str = DEFAULT_MODEL_NAME,
    boundary_x_min_rm: float | None = None,
    boundary_sample_count: int = 600,
    trajectory_markers: tuple[tuple[datetime, str, str], ...]
    | list[tuple[datetime, str, str]] = (),
) -> dict:
    """Plot a time-selected MAVEN trajectory as separate MSO XY, XZ, and YZ PNGs."""
    times, positions_km = load_mso_trajectory(start_time, end_time, mag_ss_files)
    # Use the same 3389.5-km radius as the MSO boundary-model library so the
    # trajectory, unit Mars disc, bow shock, and MPB share one R_M definition.
    positions_rm = positions_km / MSO_MARS_RADIUS_KM
    x_rm, y_rm, z_rm = positions_rm.T
    selected_markers = select_trajectory_markers(times, trajectory_markers)

    bow_model = get_model(bow_model_name)
    mpb_model = VIGNES_2000_MPB
    automatic_x_min = min(-3.0, float(np.nanmin(x_rm)) - 0.25)
    x_min_rm = automatic_x_min if boundary_x_min_rm is None else float(boundary_x_min_rm)
    x_max_rm = max(
        float(np.nanmax(x_rm)) + 0.25,
        float(bow_model.nose_x_rm()),
        float(mpb_model.nose_x_rm()),
    )

    bow_xy = boundary_plane_curves(
        bow_model, "xy", x_min_rm, x_max_rm, boundary_sample_count
    )
    bow_xz = boundary_plane_curves(
        bow_model, "xz", x_min_rm, x_max_rm, boundary_sample_count
    )
    mpb_xy = boundary_plane_curves(
        mpb_model, "xy", x_min_rm, x_max_rm, boundary_sample_count
    )
    mpb_xz = boundary_plane_curves(
        mpb_model, "xz", x_min_rm, x_max_rm, boundary_sample_count
    )

    projections = (
        ("XY plane", x_rm, y_rm, "X_MSO (R_M)", "Y_MSO (R_M)", bow_xy, mpb_xy),
        ("XZ plane", x_rm, z_rm, "X_MSO (R_M)", "Z_MSO (R_M)", bow_xz, mpb_xz),
        ("YZ plane", y_rm, z_rm, "Y_MSO (R_M)", "Z_MSO (R_M)", [], []),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem if output_path.suffix else output_path.name
    output_paths = {
        plane: output_path.parent / f"{stem}_{plane}{suffix}"
        for plane in ("xy", "xz", "yz")
    }
    interval_title = (
        f"{start_time.astimezone(timezone.utc).isoformat(timespec='seconds')} to "
        f"{end_time.astimezone(timezone.utc).isoformat(timespec='seconds')}"
    )

    for plane, (
        title,
        horizontal,
        vertical,
        xlabel,
        ylabel,
        bow_curves,
        mpb_curves,
    ) in zip(("xy", "xz", "yz"), projections):
        fig, ax = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
        _draw_mars_disc(ax)
        for curve_index, (curve_x, curve_y) in enumerate(bow_curves):
            ax.plot(
                curve_x,
                curve_y,
                color="#20252b",
                linestyle="--",
                linewidth=1.5,
                label="Bow shock" if curve_index == 0 else None,
                zorder=2,
            )
        for curve_index, (curve_x, curve_y) in enumerate(mpb_curves):
            ax.plot(
                curve_x,
                curve_y,
                color="#2878b5",
                linestyle="-.",
                linewidth=1.5,
                label="MPB" if curve_index == 0 else None,
                zorder=2,
            )

        ax.plot(
            horizontal,
            vertical,
            color="#d23875",
            linewidth=1.8,
            label="MAVEN trajectory",
            zorder=4,
        )
        ax.scatter(
            [horizontal[0]],
            [vertical[0]],
            s=55,
            marker="o",
            color="#2e8b57",
            edgecolors="white",
            linewidths=0.7,
            label="Start",
            zorder=5,
        )
        ax.scatter(
            [horizontal[-1]],
            [vertical[-1]],
            s=62,
            marker="X",
            color="#c23b3b",
            edgecolors="white",
            linewidths=0.7,
            label="End",
            zorder=5,
        )
        for marker_index, marker in enumerate(selected_markers, start=1):
            sample_index = marker["index"]
            label_time = marker["requested_time"].strftime("%H:%M:%S")
            ax.scatter(
                [horizontal[sample_index]],
                [vertical[sample_index]],
                s=72,
                marker=marker["marker"],
                color=marker["color"],
                edgecolors="white",
                linewidths=0.9,
                label=f"Line {marker_index}: {label_time}",
                zorder=6,
            )
        all_curves = list(bow_curves) + list(mpb_curves)
        x_limits, y_limits = _finite_plot_extent(horizontal, vertical, all_curves)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0.0, color="0.65", linewidth=0.6, zorder=0)
        ax.axvline(0.0, color="0.65", linewidth=0.6, zorder=0)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_title(f"MAVEN trajectory in MSO coordinates — {title}\n{interval_title}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8.5, frameon=True)
        fig.savefig(output_paths[plane], dpi=180)
        plt.close(fig)

    return {
        "coordinate_system": "MSO",
        "position_unit": "R_M",
        "mars_radius_km": MSO_MARS_RADIUS_KM,
        "start_time": start_time.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "end_time": end_time.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "track_start_time": datetime.fromtimestamp(float(times[0]), tz=timezone.utc).isoformat(timespec="seconds"),
        "track_end_time": datetime.fromtimestamp(float(times[-1]), tz=timezone.utc).isoformat(timespec="seconds"),
        "track_samples": int(times.size),
        "source_files": [str(Path(path)) for path in mag_ss_files],
        "bow_model": bow_model.metadata(),
        "mpb_model": mpb_model.metadata(),
        "boundary_x_min_rm": x_min_rm,
        "boundary_sample_count": int(boundary_sample_count),
        "position_extent_rm": {
            "x": [float(np.nanmin(x_rm)), float(np.nanmax(x_rm))],
            "y": [float(np.nanmin(y_rm)), float(np.nanmax(y_rm))],
            "z": [float(np.nanmin(z_rm)), float(np.nanmax(z_rm))],
        },
        "output_paths": {
            plane: str(path) for plane, path in output_paths.items()
        },
        "trajectory_markers": serialize_trajectory_markers(selected_markers),
    }


def plot_orbit_map(
    target_time: datetime,
    start_time: datetime,
    end_time: datetime,
    mag_pc_file: Path | list[Path] | tuple[Path, ...],
    model_root: Path,
    output_path: Path,
    crustal_altitude_km: float = 185.0,
    grid_step_deg: float = 2.0,
    model_max_degree: int | None = 60,
    trajectory_markers: tuple[tuple[datetime, str, str], ...]
    | list[tuple[datetime, str, str]] = (),
) -> dict:
    mag_pc_files = (
        [Path(mag_pc_file)]
        if isinstance(mag_pc_file, (str, Path))
        else [Path(path) for path in mag_pc_file]
    )
    times, positions = load_mag_position_trajectory(
        start_time,
        end_time,
        mag_pc_files,
    )
    target_index = locate_nearest_index(times, target_time)
    target_lon, target_lat, target_radius = pc_position_to_lon_lat(positions[target_index])
    lon_min, lon_max = choose_longitude_window(target_lon)
    lon_values, lat_values, field_mag, cache_path, cache_hit = load_or_build_crustal_field_grid(
        lon_min,
        lon_max,
        crustal_altitude_km,
        grid_step_deg,
        model_root,
        model_max_degree=model_max_degree,
    )

    track_positions = positions
    track_times = times
    selected_markers = select_trajectory_markers(track_times, trajectory_markers)
    track_lon_lat = np.asarray([pc_position_to_lon_lat(position)[:2] for position in track_positions], dtype=float)
    track_lon = wrap_longitudes_to_window(track_lon_lat[:, 0], lon_min, lon_max)
    track_lat = track_lon_lat[:, 1]
    visible = (track_lon >= lon_min) & (track_lon <= lon_max)
    target_lon_wrapped = float(wrap_longitudes_to_window(np.array([target_lon]), lon_min, lon_max)[0])

    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    sun_pc = sun_direction_pc(target_time.timestamp())
    cos_sza = np.sum(surface_unit_vectors(lon_grid, lat_grid) * sun_pc, axis=-1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    image = ax.pcolormesh(lon_values, lat_values, field_mag, shading="auto", cmap="turbo")
    cbar = fig.colorbar(image, ax=ax, pad=0.012)
    cbar.set_label(f"|B| at {crustal_altitude_km:g} km (nT)")

    ax.contour(lon_grid, lat_grid, cos_sza, levels=[0.0], colors=["#f4e04d"], linewidths=1.7)
    ax.plot(track_lon[visible], track_lat[visible], color="#ff3df2", linewidth=2.0, label="MAVEN track")
    ax.scatter(
        [target_lon_wrapped],
        [target_lat],
        s=70,
        color="#ffffff",
        edgecolors="#222222",
        linewidths=0.8,
        zorder=5,
        label="target time",
    )
    for marker_index, marker in enumerate(selected_markers, start=1):
        sample_index = marker["index"]
        marker_lon = track_lon[sample_index]
        if not (lon_min <= marker_lon <= lon_max):
            continue
        label_time = marker["requested_time"].strftime("%H:%M:%S")
        ax.scatter(
            [marker_lon],
            [track_lat[sample_index]],
            s=72,
            marker=marker["marker"],
            color=marker["color"],
            edgecolors="white",
            linewidths=0.9,
            label=f"Line {marker_index}: {label_time}",
            zorder=6,
        )
    ax.text(target_lon_wrapped + 1.5, target_lat + 1.5, "Target", color="#ffffff", weight="bold")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel("Planetocentric longitude (deg)")
    ax.set_ylabel("Planetocentric latitude (deg)")
    ax.set_title(
        "MAVEN ground track over Mars crustal magnetic field\n"
        f"{start_time.isoformat(timespec='seconds')} to {end_time.isoformat(timespec='seconds')}"
    )
    ax.grid(True, linestyle=":", color="white", alpha=0.45)
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "target_longitude_deg": target_lon,
        "target_latitude_deg": target_lat,
        "target_altitude_km": target_radius - MARS_REFERENCE_RADIUS_KM,
        "longitude_window_deg": [lon_min, lon_max],
        "crustal_altitude_km": crustal_altitude_km,
        "grid_step_deg": grid_step_deg,
        "model_max_degree": model_max_degree,
        "crustal_cache_file": str(cache_path),
        "crustal_cache_hit": cache_hit,
        "track_samples": int(times.size),
        "visible_track_samples": int(np.count_nonzero(visible)),
        "source_files": [str(path) for path in mag_pc_files],
        "trajectory_markers": serialize_trajectory_markers(selected_markers),
        "output_path": str(output_path),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot either a MAVEN ground track over the crustal-field map or "
            "a time-interval trajectory in MSO XY/XZ/YZ projections."
        )
    )
    parser.add_argument("--time", help="UTC target time, for example 2024-11-07T02:15:00.")
    parser.add_argument("--window-minutes", type=float, default=20.0, help="Full orbit-track window centered on --time.")
    parser.add_argument("--mag-pc-file", help="MAG PC 1-second STS file.")
    parser.add_argument(
        "--mso-projections",
        action="store_true",
        help="Plot the requested --start/--end interval in MSO XY, XZ, and YZ planes.",
    )
    parser.add_argument("--start", help="MSO projection interval start time in UTC.")
    parser.add_argument("--end", help="MSO projection interval end time in UTC.")
    parser.add_argument(
        "--mag-ss-file",
        action="append",
        default=None,
        help="Explicit MAG SS 1-second STS file for MSO mode; repeat for multi-day intervals.",
    )
    parser.add_argument(
        "--bow-model",
        default=DEFAULT_MODEL_NAME,
        help="Bow-shock model used in MSO XY/XZ panels.",
    )
    parser.add_argument(
        "--boundary-x-min-rm",
        type=float,
        help="Optional tailward X limit used to sample MPB/bow-shock curves in R_M.",
    )
    parser.add_argument(
        "--boundary-samples",
        type=int,
        default=600,
        help="Number of X samples per MSO boundary branch.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for MAVEN data.")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT), help="Directory for Mars crustal model files.")
    parser.add_argument(
        "--output",
        help=(
            "Output PNG path. In MSO mode this is a base name and _xy, _xz, "
            "and _yz are appended before the extension."
        ),
    )
    parser.add_argument("--crustal-altitude-km", type=float, default=185.0)
    parser.add_argument("--grid-step-deg", type=float, default=2.0)
    parser.add_argument("--model-max-degree", type=int, default=60)
    parser.add_argument(
        "--precompute-crustal-cache",
        action="store_true",
        help="Precompute both 0-180 and 180-360 longitude crustal-field grids, then exit.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    model_root = Path(args.model_root).expanduser().resolve()
    if args.precompute_crustal_cache:
        summary = precompute_crustal_field_grids(
            model_root=model_root,
            altitude_km=args.crustal_altitude_km,
            grid_step_deg=args.grid_step_deg,
            model_max_degree=args.model_max_degree,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    mso_mode = bool(args.mso_projections or args.start or args.end or args.mag_ss_file)
    if mso_mode:
        if not args.start or not args.end:
            raise ValueError("MSO projection mode requires both --start and --end.")
        start_time = parse_iso_timestamp(args.start)
        end_time = parse_iso_timestamp(args.end)
        if end_time <= start_time:
            raise ValueError("--end must be later than --start in MSO projection mode.")
        data_root = Path(args.data_root).expanduser().resolve()
        explicit_files = (
            [Path(value) for value in args.mag_ss_file]
            if args.mag_ss_file
            else None
        )
        mag_ss_files = resolve_mso_mag_files(
            start_time=start_time,
            end_time=end_time,
            data_root=data_root,
            explicit_files=explicit_files,
        )
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else (Path("outputs") / "maven_orbit_mso.png").resolve()
        )
        summary = plot_mso_orbit_projections(
            start_time=start_time,
            end_time=end_time,
            mag_ss_files=mag_ss_files,
            output_path=output_path,
            bow_model_name=args.bow_model,
            boundary_x_min_rm=args.boundary_x_min_rm,
            boundary_sample_count=args.boundary_samples,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if not args.time:
        raise ValueError(
            "--time is required for the ground-track map; use --start and --end "
            "for MSO projection mode."
        )

    target_time = parse_iso_timestamp(args.time)
    half_window = timedelta(minutes=args.window_minutes / 2.0)
    mag_pc_file = (
        Path(args.mag_pc_file).expanduser().resolve()
        if args.mag_pc_file
        else infer_daily_file(Path(args.data_root).expanduser().resolve(), "mag", "pc1s", target_time, "sts")
    )
    summary = plot_orbit_map(
        target_time=target_time,
        start_time=target_time - half_window,
        end_time=target_time + half_window,
        mag_pc_file=mag_pc_file,
        model_root=model_root,
        output_path=(
            Path(args.output).expanduser().resolve()
            if args.output
            else (Path("outputs") / "maven_orbit_map.png").resolve()
        ),
        crustal_altitude_km=args.crustal_altitude_km,
        grid_step_deg=args.grid_step_deg,
        model_max_degree=args.model_max_degree,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
