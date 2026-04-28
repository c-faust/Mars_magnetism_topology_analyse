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
from datetime import timedelta
from pathlib import Path

from analyze_magnetic_topology import TOPOLOGY_RULES, analyze_interval, resolve_daily_files, sanitize_for_json
from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from mars_crustal_model import DEFAULT_MODEL_ROOT
from plot_maven_data_panels import plot_data_panels
from plot_maven_orbit_map import plot_orbit_map
from process_maven_spectra import process_target_time


CONFIG = {
    "target_time": "2024-11-07T02:15:00",
    "window_minutes": 20.0,
    "step_seconds": 30,
    "forward_pitch_max_deg": 30.0,
    "backward_pitch_min_deg": 150.0,
    "crustal_altitude_km": 185.0,
    "map_grid_step_deg": 2.0,
    "model_max_degree": 60,
    "auto_download_missing_data": True,
    "data_root": str(DEFAULT_DATA_ROOT),
    "model_root": str(DEFAULT_MODEL_ROOT),
    "output_root": str(Path("outputs") / "maven_event_figures"),
}


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


def main() -> None:
    args = build_argument_parser().parse_args()
    cfg = runtime_config(args)
    target_time = cfg["target_time"]
    half_window = timedelta(minutes=cfg["window_minutes"] / 2.0)
    start_time = target_time - half_window
    end_time = target_time + half_window

    event_dir = cfg["output_root"] / target_time.strftime("%Y%m%dT%H%M%S")
    event_dir.mkdir(parents=True, exist_ok=True)

    resolved_files = resolve_daily_files(
        start=start_time,
        end=end_time,
        data_root=cfg["data_root"],
        pad_file=None,
        mag_file=None,
        auto_download_missing_data=cfg["auto_download_missing_data"],
    )
    target_files = resolved_files[target_time.date()]

    spectrum = process_target_time(
        target_time=target_time,
        pad_file=target_files["pad"],
        mag_file=target_files["mag_ss"],
        output_root=event_dir / "spectra",
        forward_pitch_max_deg=cfg["forward_pitch_max_deg"],
        backward_pitch_min_deg=cfg["backward_pitch_min_deg"],
    )

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

    TOPOLOGY_RULES["pitch_angle"]["parallel_max_deg"] = cfg["forward_pitch_max_deg"]
    TOPOLOGY_RULES["pitch_angle"]["anti_parallel_min_deg"] = cfg["backward_pitch_min_deg"]
    _, topology_figure = analyze_interval(
        start=start_time,
        end=end_time,
        data_root=cfg["data_root"],
        model_root=cfg["model_root"],
        output_root=event_dir / "topology_context",
        step_seconds=cfg["step_seconds"],
        auto_download_missing_data=cfg["auto_download_missing_data"],
    )
    topology_summary_path = topology_figure.parent / "topology_summary.json"
    topology_summary = json.loads(topology_summary_path.read_text(encoding="utf-8"))
    panel_result = plot_data_panels(
        summary=topology_summary,
        target_time=target_time,
        output_path=event_dir / "maven_data_panels.png",
        window_minutes=cfg["window_minutes"],
    )

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
                "topology_summary_json": str(topology_summary_path),
                "topology_trajectory_png": str(topology_figure),
                "data_panels_png": panel_result["output_path"],
            },
            "spectrum": spectrum.__dict__,
            "orbit_map": orbit_result,
            "data_panels": panel_result,
        }
    )
    summary_path = event_dir / "event_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
