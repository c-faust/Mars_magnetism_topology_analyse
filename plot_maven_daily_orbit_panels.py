from __future__ import annotations
"""
Plot one MAVEN data-panel figure for each complete orbit in a day.

This script only detects orbit windows and delegates the actual panel rendering
to plot_maven_data_panels.py so the daily and single-event figures share one
plotting implementation.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np

from analyze_magnetic_topology import (
    MARS_RADIUS_KM,
    load_mag_day,
    resolve_daily_files,
    sanitize_for_json,
)
from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from plot_maven_data_panels import (
    DEFAULT_PAD_ENERGY_BANDS_EV,
    build_data_panel_summary_from_data,
    flatten_energy_bands,
    plot_data_panels,
    validate_energy_bands,
)


@dataclass(frozen=True)
class OrbitWindow:
    orbit_number: int
    start_unix: float
    end_unix: float
    periapsis_unix: float
    periapsis_altitude_km: float

    @property
    def center_datetime(self) -> datetime:
        return datetime.fromtimestamp((self.start_unix + self.end_unix) / 2.0, tz=timezone.utc)

    @property
    def duration_minutes(self) -> float:
        return (self.end_unix - self.start_unix) / 60.0


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


def plot_orbit_panels(
    orbit: OrbitWindow,
    data_root: Path,
    output_path: Path,
    pad_energy_bands_eV: tuple[tuple[float, float], ...],
    auto_download_missing_data: bool,
    step_seconds: int,
    figure_width: float,
) -> dict:
    summary = build_data_panel_summary_from_data(
        target_time=orbit.center_datetime,
        window_minutes=orbit.duration_minutes,
        step_seconds=step_seconds,
        data_root=data_root,
        auto_download_missing_data=auto_download_missing_data,
        pad_energy_bands_eV=pad_energy_bands_eV,
    )
    result = plot_data_panels(
        summary=summary,
        target_time=orbit.center_datetime,
        output_path=output_path,
        window_minutes=orbit.duration_minutes,
        pad_energy_bands_eV=pad_energy_bands_eV,
        figure_size=(figure_width, 20.0),
        center_on_target_time=True,
    )
    return {
        "orbit_number": orbit.orbit_number,
        "start_time": datetime.fromtimestamp(orbit.start_unix, tz=timezone.utc).isoformat(timespec="seconds"),
        "end_time": datetime.fromtimestamp(orbit.end_unix, tz=timezone.utc).isoformat(timespec="seconds"),
        "center_time": orbit.center_datetime.isoformat(timespec="seconds"),
        "duration_minutes": orbit.duration_minutes,
        "periapsis_time": datetime.fromtimestamp(orbit.periapsis_unix, tz=timezone.utc).isoformat(timespec="seconds"),
        "periapsis_altitude_km": orbit.periapsis_altitude_km,
        "selected_time": result.get("selected_time"),
        "pad_energy_bands_eV": result.get("pad_energy_bands_eV"),
        "output_path": str(output_path),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot daily MAVEN data-panel figures for each complete orbit.")
    parser.add_argument("--day", required=True, help="UTC day, for example 2024-11-07.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for MAVEN data.")
    parser.add_argument(
        "--output-root",
        default=str(Path("outputs") / "maven_daily_orbit_panels"),
        help="Directory used for output figures and summary JSON.",
    )
    parser.add_argument("--no-auto-download", action="store_true", help="Disable automatic download for missing files.")
    parser.add_argument(
        "--pad-energy-bands",
        "--pad-energy-band",
        "--electron-pad-band",
        nargs="+",
        type=float,
        default=flatten_energy_bands(DEFAULT_PAD_ENERGY_BANDS_EV),
        metavar="EV",
        help="Electron PAD energy bands as LOW HIGH pairs, for example: 20 80 111 140.",
    )
    parser.add_argument(
        "--min-orbit-separation-minutes",
        type=float,
        default=180.0,
        help="Minimum time separation between periapsis candidates.",
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=60,
        help="Cadence for altitude samples passed to plot_maven_data_panels.py.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=30.0,
        help="Daily orbit figure width in inches; height follows plot_maven_data_panels.py.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    target_day = parse_day(args.day)
    start, end = day_bounds(target_day)
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_root).expanduser().resolve() / target_day.strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    auto_download_missing_data = not args.no_auto_download
    pad_energy_bands_eV = validate_energy_bands(args.pad_energy_bands)

    resolved = resolve_daily_files(
        start=start,
        end=end,
        data_root=data_root,
        pad_file=None,
        mag_file=None,
        auto_download_missing_data=auto_download_missing_data,
    )
    files = resolved[target_day]

    mag_ss = load_mag_day(files["mag_ss"])
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
                data_root=data_root,
                output_path=output_path,
                pad_energy_bands_eV=pad_energy_bands_eV,
                auto_download_missing_data=auto_download_missing_data,
                step_seconds=max(1, int(args.step_seconds)),
                figure_width=float(args.figure_width),
            )
        )

    summary = sanitize_for_json(
        {
            "day": target_day.isoformat(),
            "input_files": {key: str(path) for key, path in files.items()},
            "pad_energy_bands_eV": [list(band) for band in pad_energy_bands_eV],
            "complete_orbit_count": len(results),
            "orbits": results,
        }
    )
    summary_path = output_dir / "orbit_panel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
