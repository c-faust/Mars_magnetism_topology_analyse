from __future__ import annotations
"""
Plot one long multi-panel MAVEN overview figure for each complete orbit in a day.

The script uses local daily products already supported by the repository:
- STATIC c6-32e64m for ion energy and mass spectrograms
- SWEA svypad for electron energy and 20-80 eV pitch-angle spectrograms
- MAG ss1s for MSO magnetic field, position, and altitude
- MAG pc1s for planetocentric longitude/latitude
"""

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from analyze_magnetic_topology import (
    MARS_RADIUS_KM,
    build_swe_context,
    load_mag_day,
    load_static_context,
    resolve_daily_files,
    sanitize_for_json,
)
from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from plot_maven_data_panels import (
    axis_edges,
    positive_log_norm,
    prepare_heatmap,
    unix_to_matplotlib_dates,
)
from plot_maven_orbit_map import pc_position_to_lon_lat
from process_maven_spectra import load_pad_data, locate_nearest_index


LINE_COLORS = {
    "bx": "#c63f3b",
    "by": "#2775c7",
    "bz": "#2f8a4b",
    "bt": "#4b4038",
    "alt": "#8a5a28",
}
FLUX_CMAP = "magma"
PAD_CMAP = "turbo"


@dataclass(frozen=True)
class OrbitWindow:
    orbit_number: int
    start_unix: float
    end_unix: float
    periapsis_unix: float
    periapsis_altitude_km: float


def parse_day(value: str) -> date:
    if "T" in value:
        return parse_iso_timestamp(value).date()
    return date.fromisoformat(value)


def day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    end = datetime.combine(day, time.max, tzinfo=timezone.utc)
    return start, end


def finite_array(values) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=float)


def time_mask(times_unix: np.ndarray, start_unix: float, end_unix: float) -> np.ndarray:
    times = finite_array(times_unix)
    return (times >= start_unix) & (times <= end_unix)


def mdates_from_unix(times_unix: np.ndarray) -> list[datetime]:
    return [datetime.fromtimestamp(float(value), tz=timezone.utc) for value in finite_array(times_unix)]


def smooth_boxcar(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if width <= 1 or values.size < width:
        return values
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def find_periapsis_indices(
    times_unix: np.ndarray,
    altitude_km: np.ndarray,
    min_separation_minutes: float = 180.0,
    smoothing_seconds: int = 61,
) -> list[int]:
    times = finite_array(times_unix)
    altitude = finite_array(altitude_km)
    if times.size < 3 or altitude.size != times.size:
        return []

    cadence = float(np.nanmedian(np.diff(times))) if times.size > 1 else 1.0
    smooth_width = max(1, int(round(smoothing_seconds / max(cadence, 1e-6))))
    if smooth_width % 2 == 0:
        smooth_width += 1
    smoothed = smooth_boxcar(altitude, smooth_width)
    candidates = np.where((smoothed[1:-1] <= smoothed[:-2]) & (smoothed[1:-1] < smoothed[2:]))[0] + 1
    if candidates.size == 0:
        return []

    min_sep = min_separation_minutes * 60.0
    kept: list[int] = []
    for idx in candidates:
        if not kept or times[idx] - times[kept[-1]] >= min_sep:
            kept.append(int(idx))
            continue
        if altitude[idx] < altitude[kept[-1]]:
            kept[-1] = int(idx)
    return kept


def build_complete_orbits(
    times_unix: np.ndarray,
    altitude_km: np.ndarray,
    min_separation_minutes: float,
) -> list[OrbitWindow]:
    periapsis_indices = find_periapsis_indices(times_unix, altitude_km, min_separation_minutes=min_separation_minutes)
    orbits: list[OrbitWindow] = []
    for orbit_number, (left, right) in enumerate(zip(periapsis_indices[:-1], periapsis_indices[1:]), start=1):
        orbits.append(
            OrbitWindow(
                orbit_number=orbit_number,
                start_unix=float(times_unix[left]),
                end_unix=float(times_unix[right]),
                periapsis_unix=float(times_unix[left]),
                periapsis_altitude_km=float(altitude_km[left]),
            )
        )
    return orbits


def pad_band_context(pad_data: dict, start: datetime, end: datetime, energy_band_eV: tuple[float, float]) -> dict | None:
    times = finite_array(pad_data["times"])
    mask = time_mask(times, start.timestamp(), end.timestamp())
    if not np.any(mask):
        return None
    indices = np.where(mask)[0]
    flux = np.asarray(pad_data["flux"], dtype=float)[indices]
    energy = finite_array(pad_data["energy"])
    band = (energy >= energy_band_eV[0]) & (energy <= energy_band_eV[1])
    if not np.any(band):
        return None

    band_flux = flux[:, :, band]
    valid_counts = np.sum(np.isfinite(band_flux), axis=2)
    band_sum = np.nansum(band_flux, axis=2)
    pad_band = np.divide(
        band_sum,
        valid_counts,
        out=np.full_like(band_sum, np.nan, dtype=float),
        where=valid_counts > 0,
    )

    pitch = np.asarray(pad_data["pitch"], dtype=float)
    if pitch.ndim == 1:
        pitch_bins = pitch
    else:
        pitch_bins = np.nanmedian(pitch[indices][:, :, band], axis=(0, 2))
    return {
        "times_unix": times[indices],
        "pitch_deg": np.asarray(pitch_bins, dtype=float),
        "pad_eflux": pad_band,
    }


def plot_heatmap(ax, times_unix, matrix, y_values, title: str, ylabel: str, log_y: bool, cmap: str):
    times = finite_array(times_unix)
    y = finite_array(y_values)
    if times.size == 0 or y.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, loc="left")
        return None

    z, y_sorted = prepare_heatmap(matrix, y, log_y=log_y)
    if z.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, loc="left")
        return None

    mesh = ax.pcolormesh(
        axis_edges(unix_to_matplotlib_dates(times), log_scale=False),
        axis_edges(y_sorted, log_scale=log_y),
        z,
        shading="auto",
        cmap=cmap,
        norm=positive_log_norm(z),
    )
    ax.set_title(title, loc="left", fontsize=20)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    return mesh


def subset_context(context: dict | None, start_unix: float, end_unix: float, array_keys: list[str]) -> dict:
    if not context:
        return {}
    times = finite_array(context.get("times_unix"))
    mask = time_mask(times, start_unix, end_unix)
    out = {key: context.get(key) for key in context if key not in array_keys and key != "times_unix"}
    out["times_unix"] = times[mask]
    for key in array_keys:
        out[key] = np.asarray(context.get(key, []), dtype=float)[mask] if np.any(mask) else np.asarray([])
    return out


def sample_annotation_rows(
    sample_times: np.ndarray,
    mag_ss: dict,
    mag_pc: dict,
    count: int,
) -> list[tuple[float, str, str, str, str]]:
    if sample_times.size == 0:
        return []
    selected = np.linspace(sample_times[0], sample_times[-1], min(count, sample_times.size))
    rows = []
    ss_times = finite_array(mag_ss["times"])
    pc_times = finite_array(mag_pc["times"])
    for value in selected:
        ss_index = locate_nearest_index(ss_times, datetime.fromtimestamp(float(value), tz=timezone.utc))
        pc_index = locate_nearest_index(pc_times, datetime.fromtimestamp(float(value), tz=timezone.utc))
        pos_ss = np.asarray(mag_ss["data"][ss_index, mag_ss["pos_indices"]], dtype=float)
        pos_pc = np.asarray(mag_pc["data"][pc_index, mag_pc["pos_indices"]], dtype=float)
        lon, lat, _ = pc_position_to_lon_lat(pos_pc)
        alt = float(np.linalg.norm(pos_ss) - MARS_RADIUS_KM)
        rows.append(
            (
                float(value),
                datetime.fromtimestamp(float(value), tz=timezone.utc).strftime("%H:%M:%S"),
                f"{pos_ss[0] / MARS_RADIUS_KM:+.2f} {pos_ss[1] / MARS_RADIUS_KM:+.2f} {pos_ss[2] / MARS_RADIUS_KM:+.2f}",
                f"{lon:05.1f} {lat:+05.1f}",
                f"{alt:.0f}",
            )
        )
    return rows


def draw_annotation_panel(ax, rows: list[tuple[float, str, str, str, str]], start_unix: float, end_unix: float) -> None:
    ax.set_xlim(
        mdates.date2num(datetime.fromtimestamp(start_unix, tz=timezone.utc)),
        mdates.date2num(datetime.fromtimestamp(end_unix, tz=timezone.utc)),
    )
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    labels = [("UTC", 0.82), ("MSO XYZ (R_M)", 0.57), ("Lon Lat (deg)", 0.32), ("Alt (km)", 0.10)]
    for label, y in labels:
        ax.text(-0.012, y, label, transform=ax.transAxes, ha="right", va="center", fontsize=18, weight="bold")
    for value, utc, xyz, lonlat, alt in rows:
        x = mdates.date2num(datetime.fromtimestamp(value, tz=timezone.utc))
        for text, y in [(utc, 0.82), (xyz, 0.57), (lonlat, 0.32), (alt, 0.10)]:
            ax.text(x, y, text, ha="center", va="center", fontsize=18, rotation=90 if len(text) > 12 else 0)


def plot_orbit_panels(
    orbit: OrbitWindow,
    static_context: dict | None,
    swe_context: dict | None,
    pad_context: dict | None,
    mag_ss: dict,
    mag_pc: dict,
    output_path: Path,
    electron_pad_band_eV: tuple[float, float],
    annotation_count: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        7,
        1,
        figsize=(30, 23.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 1.2, 1.35, 1.2, 1.0, 0.9, 0.95]},
    )

    static = subset_context(static_context, orbit.start_unix, orbit.end_unix, ["energy_eflux", "mass_eflux"])
    mesh = plot_heatmap(
        axes[0],
        static.get("times_unix", []),
        static.get("energy_eflux", []),
        static_context.get("energy_eV", []) if static_context else [],
        "STATIC ion energy spectrogram",
        "Energy (eV)",
        True,
        FLUX_CMAP,
    )
    if mesh:
        fig.colorbar(mesh, ax=axes[0], pad=0.01, label="eflux")

    mesh = plot_heatmap(
        axes[1],
        static.get("times_unix", []),
        static.get("mass_eflux", []),
        static_context.get("mass_amu", []) if static_context else [],
        "STATIC ion mass spectrogram",
        "Mass (amu)",
        False,
        FLUX_CMAP,
    )
    if mesh:
        fig.colorbar(mesh, ax=axes[1], pad=0.01, label="eflux")

    swe = subset_context(swe_context, orbit.start_unix, orbit.end_unix, ["omni_eflux"])
    mesh = plot_heatmap(
        axes[2],
        swe.get("times_unix", []),
        swe.get("omni_eflux", []),
        swe_context.get("energy_eV", []) if swe_context else [],
        "SWEA electron energy spectrogram",
        "Energy (eV)",
        True,
        FLUX_CMAP,
    )
    if mesh:
        fig.colorbar(mesh, ax=axes[2], pad=0.01, label="eflux")

    pad = subset_context(pad_context, orbit.start_unix, orbit.end_unix, ["pad_eflux"])
    mesh = plot_heatmap(
        axes[3],
        pad.get("times_unix", []),
        pad.get("pad_eflux", []),
        pad_context.get("pitch_deg", []) if pad_context else [],
        f"SWEA pitch-angle spectrogram ({electron_pad_band_eV[0]:g}-{electron_pad_band_eV[1]:g} eV)",
        "Pitch angle (deg)",
        False,
        PAD_CMAP,
    )
    if mesh:
        fig.colorbar(mesh, ax=axes[3], pad=0.01, label="eflux")

    mag_times = finite_array(mag_ss["times"])
    mag_mask = time_mask(mag_times, orbit.start_unix, orbit.end_unix)
    mag_selected = np.asarray(mag_ss["data"], dtype=float)[mag_mask]
    x_mag = mdates_from_unix(mag_times[mag_mask])
    if mag_selected.size:
        bx = mag_selected[:, mag_ss["b_indices"][0]]
        by = mag_selected[:, mag_ss["b_indices"][1]]
        bz = mag_selected[:, mag_ss["b_indices"][2]]
        bt = np.sqrt(bx * bx + by * by + bz * bz)
        axes[4].plot(x_mag, bx, color=LINE_COLORS["bx"], linewidth=0.9, label="Bx")
        axes[4].plot(x_mag, by, color=LINE_COLORS["by"], linewidth=0.9, label="By")
        axes[4].plot(x_mag, bz, color=LINE_COLORS["bz"], linewidth=0.9, label="Bz")
        axes[4].plot(x_mag, bt, color=LINE_COLORS["bt"], linewidth=1.1, label="|B|")
        axes[4].legend(loc="upper right", ncol=4, frameon=False)
    axes[4].set_title("MAG magnetic field in MSO", loc="left", fontsize=20)
    axes[4].set_ylabel("nT")

    if mag_selected.size:
        pos = mag_selected[:, mag_ss["pos_indices"]]
        altitude = np.linalg.norm(pos, axis=1) - MARS_RADIUS_KM
        axes[5].plot(x_mag, altitude, color=LINE_COLORS["alt"], linewidth=1.1)
        axes[5].axvline(datetime.fromtimestamp(orbit.periapsis_unix, tz=timezone.utc), color="black", linestyle="--", linewidth=1.0)
    axes[5].set_title("Spacecraft altitude", loc="left", fontsize=20)
    axes[5].set_ylabel("km")

    annotation_rows = sample_annotation_rows(mag_times[mag_mask], mag_ss, mag_pc, annotation_count)
    draw_annotation_panel(axes[6], annotation_rows, orbit.start_unix, orbit.end_unix)

    x_start = mdates.date2num(datetime.fromtimestamp(orbit.start_unix, tz=timezone.utc))
    x_end = mdates.date2num(datetime.fromtimestamp(orbit.end_unix, tz=timezone.utc))
    for ax in axes[:6]:
        ax.set_xlim(x_start, x_end)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.grid(True, linestyle=":", alpha=0.25)
    for ax in axes[:5]:
        ax.tick_params(labelbottom=False)

    start_label = datetime.fromtimestamp(orbit.start_unix, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    end_label = datetime.fromtimestamp(orbit.end_unix, tz=timezone.utc).strftime("%H:%M:%S")
    fig.suptitle(
        f"MAVEN orbit {orbit.orbit_number:02d} | {start_label} to {end_label} UTC | "
        f"periapsis altitude {orbit.periapsis_altitude_km:.0f} km",
        fontsize=18,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "orbit_number": orbit.orbit_number,
        "start_time": datetime.fromtimestamp(orbit.start_unix, tz=timezone.utc).isoformat(timespec="seconds"),
        "end_time": datetime.fromtimestamp(orbit.end_unix, tz=timezone.utc).isoformat(timespec="seconds"),
        "periapsis_time": datetime.fromtimestamp(orbit.periapsis_unix, tz=timezone.utc).isoformat(timespec="seconds"),
        "periapsis_altitude_km": orbit.periapsis_altitude_km,
        "output_path": str(output_path),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot daily MAVEN multi-panel figures for each complete orbit.")
    parser.add_argument("--day", required=True, help="UTC day, for example 2024-11-07.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for MAVEN data.")
    parser.add_argument(
        "--output-root",
        default=str(Path("outputs") / "maven_daily_orbit_panels"),
        help="Directory used for output figures and summary JSON.",
    )
    parser.add_argument("--no-auto-download", action="store_true", help="Disable automatic download for missing files.")
    parser.add_argument(
        "--electron-pad-band",
        nargs=2,
        type=float,
        default=(20.0, 80.0),
        metavar=("LOW_EV", "HIGH_EV"),
        help="Electron energy band averaged into the pitch-angle panel.",
    )
    parser.add_argument(
        "--min-orbit-separation-minutes",
        type=float,
        default=180.0,
        help="Minimum time separation between periapsis candidates.",
    )
    parser.add_argument(
        "--annotation-count",
        type=int,
        default=9,
        help="Number of bottom annotation columns per orbit.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    target_day = parse_day(args.day)
    start, end = day_bounds(target_day)
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_root).expanduser().resolve() / target_day.strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_daily_files(
        start=start,
        end=end,
        data_root=data_root,
        pad_file=None,
        mag_file=None,
        auto_download_missing_data=not args.no_auto_download,
    )
    files = resolved[target_day]

    mag_ss = load_mag_day(files["mag_ss"])
    mag_pc = load_mag_day(files["mag_pc"])
    pad_data = load_pad_data(files["pad"])
    static_context = load_static_context(files["sta_c6"], start, end)
    swe_context = build_swe_context(pad_data, start, end)
    pad_context = pad_band_context(pad_data, start, end, tuple(args.electron_pad_band))

    mag_times = finite_array(mag_ss["times"])
    positions = np.asarray(mag_ss["data"][:, mag_ss["pos_indices"]], dtype=float)
    altitude = np.linalg.norm(positions, axis=1) - MARS_RADIUS_KM
    day_mask = time_mask(mag_times, start.timestamp(), end.timestamp())
    orbits = build_complete_orbits(
        mag_times[day_mask],
        altitude[day_mask],
        min_separation_minutes=args.min_orbit_separation_minutes,
    )
    if not orbits:
        raise ValueError(f"No complete orbits were detected on {target_day.isoformat()}.")

    results = []
    for orbit in orbits:
        output_path = output_dir / f"maven_{target_day.strftime('%Y%m%d')}_orbit_{orbit.orbit_number:02d}.png"
        results.append(
            plot_orbit_panels(
                orbit=orbit,
                static_context=static_context,
                swe_context=swe_context,
                pad_context=pad_context,
                mag_ss=mag_ss,
                mag_pc=mag_pc,
                output_path=output_path,
                electron_pad_band_eV=tuple(args.electron_pad_band),
                annotation_count=max(2, int(args.annotation_count)),
            )
        )

    summary = sanitize_for_json(
        {
            "day": target_day.isoformat(),
            "input_files": {key: str(path) for key, path in files.items()},
            "electron_pad_band_eV": list(args.electron_pad_band),
            "complete_orbit_count": len(results),
            "orbits": results,
        }
    )
    summary_path = output_dir / "orbit_panel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
