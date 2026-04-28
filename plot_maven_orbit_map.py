from __future__ import annotations
"""
Plot a MAVEN ground track over a Mars crustal magnetic-field map.

The plot uses MAG planetocentric (PC) positions for the spacecraft ground
track. The background is the Morschhauser et al. (2014) crustal model evaluated
on a latitude/longitude grid at a configurable altitude.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from scipy.special import gammaln, lpmv

from analyze_magnetic_topology import load_mag_day
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


def plot_orbit_map(
    target_time: datetime,
    start_time: datetime,
    end_time: datetime,
    mag_pc_file: Path,
    model_root: Path,
    output_path: Path,
    crustal_altitude_km: float = 185.0,
    grid_step_deg: float = 2.0,
    model_max_degree: int | None = 60,
) -> dict:
    mag_data = load_mag_day(mag_pc_file)
    times = np.asarray(mag_data["times"], dtype=float)
    positions = np.asarray(mag_data["data"][:, mag_data["pos_indices"]], dtype=float)
    mask = (times >= start_time.timestamp()) & (times <= end_time.timestamp())
    if not np.any(mask):
        raise ValueError("No MAG PC samples were found in the requested orbit-map window.")

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

    track_positions = positions[mask]
    track_times = times[mask]
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
        "track_samples": int(np.count_nonzero(mask)),
        "visible_track_samples": int(np.count_nonzero(visible)),
        "output_path": str(output_path),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot MAVEN orbit over a Mars crustal magnetic-field map.")
    parser.add_argument("--time", help="UTC target time, for example 2024-11-07T02:15:00.")
    parser.add_argument("--window-minutes", type=float, default=20.0, help="Full orbit-track window centered on --time.")
    parser.add_argument("--mag-pc-file", help="MAG PC 1-second STS file.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for MAVEN data.")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT), help="Directory for Mars crustal model files.")
    parser.add_argument("--output", default=str(Path("outputs") / "maven_orbit_map.png"), help="Output PNG path.")
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

    if not args.time:
        raise ValueError("--time is required unless --precompute-crustal-cache is used.")

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
        output_path=Path(args.output).expanduser().resolve(),
        crustal_altitude_km=args.crustal_altitude_km,
        grid_step_deg=args.grid_step_deg,
        model_max_degree=args.model_max_degree,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
