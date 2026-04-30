from __future__ import annotations
"""
Controller for single-event MAVEN figure generation.

Given one target time, this script checks that all required daily products are
available, downloads missing files when allowed, then generates:
- directional electron spectrum
- orbit/crustal-field map
- magnetic-topology data panels
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from analyze_magnetic_topology import (
    MARS_RADIUS_KM,
    build_context_overview,
    load_mag_day,
    resolve_daily_files,
    sanitize_for_json,
    select_time_indices,
)
from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from mars_crustal_model import DEFAULT_MODEL_ROOT
from plot_maven_data_panels import plot_data_panels
from plot_maven_orbit_map import plot_orbit_map
from process_maven_spectra import process_target_time


CONFIG = {
    "target_time": "2024-11-07T02:15:00",
    "window_minutes": 20.0,
    "step_seconds": 30,
    "forward_pitch_max_deg": 60.0,
    "backward_pitch_min_deg": 120.0,
    "crustal_altitude_km": 185.0,
    "map_grid_step_deg": 2.0,
    "model_max_degree": 60,
    "auto_download_missing_data": True,
    "data_root": str(DEFAULT_DATA_ROOT),
    "model_root": str(DEFAULT_MODEL_ROOT),
    "output_root": str(Path("outputs") / "maven_event_figures"),
}


def log_step(message: str) -> None:
    print(f"[event] {datetime.now().isoformat(timespec='seconds')} | {message}", flush=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate all MAVEN event figures for one target time.")
    parser.add_argument("--time", help="UTC target time, for example 2024-11-07T02:15:00.")
    parser.add_argument("--window-minutes", type=float, help="Full orbit/data-panel time window centered on --time.")
    parser.add_argument("--step-seconds", type=int, help="Sampling cadence for topology/data-panel samples.")
    parser.add_argument("--forward-pitch-max", type=float, help="Forward spectrum pitch-angle upper bound in degrees.")
    parser.add_argument("--backward-pitch-min", type=float, help="Backward spectrum pitch-angle lower bound in degrees.")
    parser.add_argument("--crustal-altitude-km", type=float, help="Altitude for the crustal-field background map.")
    parser.add_argument("--map-grid-step-deg", type=float, help="Latitude/longitude grid spacing for the crustal map.")
    parser.add_argument("--model-max-degree", type=int, help="Maximum crustal-model spherical-harmonic degree for the map.")
    parser.add_argument("--data-root", help="Root directory for MAVEN data.")
    parser.add_argument("--model-root", help="Directory for Mars crustal model files.")
    parser.add_argument("--output-root", help="Directory used for all generated event products.")
    parser.add_argument("--no-auto-download", action="store_true", help="Disable automatic download for missing data.")
    return parser


def runtime_config(args: argparse.Namespace) -> dict:
    target_time = parse_iso_timestamp(args.time or CONFIG["target_time"])
    window_minutes = args.window_minutes if args.window_minutes is not None else CONFIG["window_minutes"]
    step_seconds = args.step_seconds if args.step_seconds is not None else CONFIG["step_seconds"]
    forward_pitch_max = (
        args.forward_pitch_max if args.forward_pitch_max is not None else CONFIG["forward_pitch_max_deg"]
    )
    backward_pitch_min = (
        args.backward_pitch_min if args.backward_pitch_min is not None else CONFIG["backward_pitch_min_deg"]
    )
    crustal_altitude_km = (
        args.crustal_altitude_km if args.crustal_altitude_km is not None else CONFIG["crustal_altitude_km"]
    )
    map_grid_step_deg = (
        args.map_grid_step_deg if args.map_grid_step_deg is not None else CONFIG["map_grid_step_deg"]
    )
    model_max_degree = args.model_max_degree if args.model_max_degree is not None else CONFIG["model_max_degree"]
    auto_download = False if args.no_auto_download else bool(CONFIG["auto_download_missing_data"])
    if window_minutes <= 0:
        raise ValueError("window-minutes must be positive.")
    if step_seconds <= 0:
        raise ValueError("step-seconds must be positive.")
    if not (0.0 < forward_pitch_max < backward_pitch_min < 180.0):
        raise ValueError("Pitch-angle bounds must satisfy 0 < forward < backward < 180.")
    if map_grid_step_deg <= 0:
        raise ValueError("map-grid-step-deg must be positive.")
    if model_max_degree is not None and model_max_degree <= 0:
        raise ValueError("model-max-degree must be positive.")

    return {
        "target_time": target_time,
        "window_minutes": float(window_minutes),
        "step_seconds": int(step_seconds),
        "forward_pitch_max_deg": float(forward_pitch_max),
        "backward_pitch_min_deg": float(backward_pitch_min),
        "crustal_altitude_km": float(crustal_altitude_km),
        "map_grid_step_deg": float(map_grid_step_deg),
        "model_max_degree": int(model_max_degree) if model_max_degree is not None else None,
        "auto_download_missing_data": auto_download,
        "data_root": Path(args.data_root or CONFIG["data_root"]).expanduser().resolve(),
        "model_root": Path(args.model_root or CONFIG["model_root"]).expanduser().resolve(),
        "output_root": Path(args.output_root or CONFIG["output_root"]).expanduser().resolve(),
    }


def build_panel_summary_without_topology(
    start_time,
    end_time,
    resolved_files: dict,
    model_root: Path,
    step_seconds: int,
) -> dict:
    samples = []
    for day, files in resolved_files.items():
        mag_ss = load_mag_day(files["mag_ss"])
        mag_times = np.asarray(mag_ss["times"], dtype=float)
        day_start = max(start_time, datetime.combine(day, datetime.min.time(), tzinfo=start_time.tzinfo))
        day_end = min(end_time, datetime.combine(day, datetime.max.time(), tzinfo=end_time.tzinfo))
        indices = select_time_indices(mag_times, day_start, day_end, step_seconds)
        for index in indices:
            sample_time = datetime.fromtimestamp(float(mag_times[index]), tz=start_time.tzinfo)
            position_km = np.asarray(mag_ss["data"][index, mag_ss["pos_indices"]], dtype=float)
            position_rm = position_km / MARS_RADIUS_KM
            altitude_km = float(np.linalg.norm(position_km) - MARS_RADIUS_KM)
            samples.append(
                {
                    "target_time": sample_time.isoformat(timespec="seconds"),
                    "topology": "not_computed",
                    "altitude_km": altitude_km,
                    "altitude_rm": altitude_km / MARS_RADIUS_KM,
                    "position_km": position_km.tolist(),
                    "position_rm": position_rm.tolist(),
                    "positions_by_frame_km": {"ss": position_km.tolist()},
                    "positions_by_frame_rm": {"ss": position_rm.tolist()},
                }
            )

    if not samples:
        raise ValueError("No MAG samples were found for the requested event window.")

    return sanitize_for_json(
        {
            "start_time": start_time.isoformat(timespec="seconds"),
            "end_time": end_time.isoformat(timespec="seconds"),
            "step_seconds": step_seconds,
            "topology_computed": False,
            "context_overview": build_context_overview(start_time, end_time, resolved_files, model_root),
            "samples": samples,
        }
    )


def main() -> None:
    log_step("Starting MAVEN event figure pipeline.")
    args = build_argument_parser().parse_args()
    log_step("Parsing runtime configuration.")
    cfg = runtime_config(args)
    target_time = cfg["target_time"]
    half_window = timedelta(minutes=cfg["window_minutes"] / 2.0)
    start_time = target_time - half_window
    end_time = target_time + half_window
    log_step(
        "Configured event: "
        f"target={target_time.isoformat(timespec='seconds')}, "
        f"window={start_time.isoformat(timespec='seconds')} to {end_time.isoformat(timespec='seconds')}, "
        f"step={cfg['step_seconds']}s."
    )
    log_step(f"Data root: {cfg['data_root']}")
    log_step(f"Model root: {cfg['model_root']}")
    log_step(f"Auto-download missing data: {cfg['auto_download_missing_data']}")

    event_dir = cfg["output_root"] / target_time.strftime("%Y%m%dT%H%M%S")
    event_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Event output directory ready: {event_dir}")

    log_step("Resolving required daily MAVEN files; missing files may be downloaded here.")
    resolved_files = resolve_daily_files(
        start=start_time,
        end=end_time,
        data_root=cfg["data_root"],
        pad_file=None,
        mag_file=None,
        auto_download_missing_data=cfg["auto_download_missing_data"],
    )
    target_files = resolved_files[target_time.date()]
    for key, path in target_files.items():
        log_step(f"Resolved input file [{key}]: {path}")

    log_step("Generating directional electron spectrum.")
    spectrum = process_target_time(
        target_time=target_time,
        pad_file=target_files["pad"],
        mag_file=target_files["mag_ss"],
        output_root=event_dir / "spectra",
        forward_pitch_max_deg=cfg["forward_pitch_max_deg"],
        backward_pitch_min_deg=cfg["backward_pitch_min_deg"],
    )
    log_step("Directional electron spectrum complete.")

    log_step("Generating orbit/crustal-field map.")
    orbit_result = plot_orbit_map(
        target_time=target_time,
        start_time=start_time,
        end_time=end_time,
        mag_pc_file=target_files["mag_pc"],
        model_root=cfg["model_root"],
        output_path=event_dir / "orbit_crustal_map.png",
        crustal_altitude_km=cfg["crustal_altitude_km"],
        grid_step_deg=cfg["map_grid_step_deg"],
        model_max_degree=cfg["model_max_degree"],
    )
    log_step(f"Orbit/crustal-field map complete: {orbit_result['output_path']}")

    log_step("Building data-panel context without magnetic-topology classification.")
    topology_summary = build_panel_summary_without_topology(
        start_time=start_time,
        end_time=end_time,
        resolved_files=resolved_files,
        model_root=cfg["model_root"],
        step_seconds=cfg["step_seconds"],
    )
    topology_summary_path = event_dir / "data_panel_context_summary.json"
    topology_summary_path.write_text(
        json.dumps(topology_summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    log_step(f"Data-panel context summary written: {topology_summary_path}")
    log_step("Generating MAVEN data panels.")
    panel_result = plot_data_panels(
        summary=topology_summary,
        target_time=target_time,
        output_path=event_dir / "maven_data_panels.png",
        window_minutes=cfg["window_minutes"],
    )
    log_step(f"MAVEN data panels complete: {panel_result['output_path']}")

    log_step("Writing event pipeline summary.")
    summary = sanitize_for_json(
        {
            "target_time": target_time.isoformat(timespec="seconds"),
            "start_time": start_time.isoformat(timespec="seconds"),
            "end_time": end_time.isoformat(timespec="seconds"),
            "parameters": {
                "window_minutes": cfg["window_minutes"],
                "step_seconds": cfg["step_seconds"],
                "forward_pitch_max_deg": cfg["forward_pitch_max_deg"],
                "backward_pitch_min_deg": cfg["backward_pitch_min_deg"],
                "crustal_altitude_km": cfg["crustal_altitude_km"],
                "map_grid_step_deg": cfg["map_grid_step_deg"],
                "model_max_degree": cfg["model_max_degree"],
                "auto_download_missing_data": cfg["auto_download_missing_data"],
            },
            "input_files": {key: str(value) for key, value in target_files.items()},
            "outputs": {
                "event_dir": str(event_dir),
                "spectrum_summary": str(event_dir / "spectra" / target_time.strftime("%Y%m%dT%H%M%S") / "spectrum_summary.json"),
                "spectrum_png": str(event_dir / "spectra" / target_time.strftime("%Y%m%dT%H%M%S") / "directional_electron_spectra.png"),
                "orbit_map_png": orbit_result["output_path"],
                "topology_summary_json": None,
                "topology_trajectory_png": None,
                "data_panel_context_summary_json": str(topology_summary_path),
                "data_panels_png": panel_result["output_path"],
            },
            "spectrum": spectrum.__dict__,
            "orbit_map": orbit_result,
            "data_panels": panel_result,
        }
    )
    summary_path = event_dir / "event_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    log_step(f"Event pipeline summary written: {summary_path}")
    log_step("MAVEN event figure pipeline finished.")
    print(json.dumps(summary["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
