from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bow_shock.data_interface import get_bow_shock_context, get_bow_shock_surface
from bow_shock.models import DEFAULT_MODEL_NAME, MARS_RADIUS_KM, get_model, list_models
from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _draw_mars_2d(axis: plt.Axes) -> None:
    mars = plt.Circle((0.0, 0.0), 1.0, facecolor="#b85c38", edgecolor="#6f2f1e", alpha=0.9)
    axis.add_patch(mars)


def _draw_cross_section(
    axis: plt.Axes,
    model,
    x_axis: np.ndarray,
    positive_azimuth: float,
    negative_azimuth: float,
    spacecraft_x: float,
    spacecraft_transverse: float,
    boundary_position_rm: np.ndarray | None,
    transverse_index: int,
    title: str,
    transverse_label: str,
) -> None:
    positive = model.rho_at_x_azimuth(x_axis, positive_azimuth)
    negative = model.rho_at_x_azimuth(x_axis, negative_azimuth)
    axis.plot(x_axis, positive, color="#1565c0", linewidth=2.2)
    axis.plot(x_axis, -negative, color="#1565c0", linewidth=2.2, label=model.display_name)
    _draw_mars_2d(axis)
    axis.scatter(
        spacecraft_x,
        spacecraft_transverse,
        marker="*",
        s=130,
        color="#d62828",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="MAVEN",
    )
    if boundary_position_rm is not None:
        axis.scatter(
            boundary_position_rm[0],
            boundary_position_rm[transverse_index],
            marker="o",
            s=42,
            color="#f9c74f",
            edgecolor="#5f4b00",
            linewidth=0.8,
            zorder=5,
            label="Boundary on MAVEN ray",
        )
        axis.plot(
            [0.0, boundary_position_rm[0]],
            [0.0, boundary_position_rm[transverse_index]],
            color="#777777",
            linewidth=0.9,
            linestyle=":",
        )
    axis.axhline(0.0, color="#777777", linewidth=0.6)
    axis.axvline(0.0, color="#777777", linewidth=0.6)
    axis.set_xlabel(r"X MSO ($R_M$)")
    axis.set_ylabel(f"{transverse_label} MSO ($R_M$)")
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)


def _draw_3d(
    axis,
    surface: dict,
    position_rm: np.ndarray,
    boundary_position_rm: np.ndarray | None,
) -> None:
    x = np.asarray(surface["x_rm"], dtype=float)
    y = np.asarray(surface["y_rm"], dtype=float)
    z = np.asarray(surface["z_rm"], dtype=float)
    axis.plot_surface(
        x,
        y,
        z,
        color="#4d9de0",
        alpha=0.28,
        linewidth=0.0,
        antialiased=True,
    )

    polar = np.linspace(0.0, np.pi, 30)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 48)
    mars_x = np.outer(np.cos(polar), np.ones_like(azimuth))
    mars_y = np.outer(np.sin(polar), np.cos(azimuth))
    mars_z = np.outer(np.sin(polar), np.sin(azimuth))
    axis.plot_surface(mars_x, mars_y, mars_z, color="#b85c38", alpha=0.9, linewidth=0.0)
    axis.scatter(*position_rm, marker="*", s=110, color="#d62828", edgecolor="white", linewidth=0.8)
    if boundary_position_rm is not None:
        axis.scatter(
            *boundary_position_rm,
            marker="o",
            s=36,
            color="#f9c74f",
            edgecolor="#5f4b00",
            linewidth=0.8,
        )
        axis.plot(
            [0.0, boundary_position_rm[0]],
            [0.0, boundary_position_rm[1]],
            [0.0, boundary_position_rm[2]],
            color="#777777",
            linewidth=0.9,
            linestyle=":",
        )
    axis.set_xlabel(r"X MSO ($R_M$)")
    axis.set_ylabel(r"Y MSO ($R_M$)")
    axis.set_zlabel(r"Z MSO ($R_M$)")
    axis.set_title("3-D bow-shock surface")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.any(finite):
        x_limits = (float(np.nanmin(x[finite])), float(np.nanmax(x[finite])))
        transverse_limit = float(max(np.nanmax(np.abs(y[finite])), np.nanmax(np.abs(z[finite])), 1.0))
        axis.set_xlim(*x_limits)
        axis.set_ylim(-transverse_limit, transverse_limit)
        axis.set_zlim(-transverse_limit, transverse_limit)
        axis.set_box_aspect(
            (max(x_limits[1] - x_limits[0], 1.0), 2.0 * transverse_limit, 2.0 * transverse_limit)
        )
        axis.set_xticks(np.linspace(x_limits[0], x_limits[1], 5))
        axis.set_yticks(np.linspace(-transverse_limit, transverse_limit, 5))
        axis.set_zticks(np.linspace(-transverse_limit, transverse_limit, 5))
        axis.tick_params(labelsize=8)
    axis.view_init(elev=22.0, azim=-62.0)


def plot_bow_shock(
    time_utc: str,
    model_name: str = DEFAULT_MODEL_NAME,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_path: str | Path | None = None,
    context_json_path: str | Path | None = None,
    x_min_rm: float = -3.0,
    x_max_rm: float | None = None,
    max_mag_delta_seconds: float = 5.0,
    boundary_tolerance_km: float = 10.0,
) -> tuple[Path, Path, dict]:
    model = get_model(model_name)
    context = get_bow_shock_context(
        time_utc,
        model_name=model.name,
        data_root=data_root,
        max_mag_delta_seconds=max_mag_delta_seconds,
        boundary_tolerance_km=boundary_tolerance_km,
    )
    surface = get_bow_shock_surface(
        time_utc,
        model_name=model.name,
        x_min_rm=x_min_rm,
        x_max_rm=x_max_rm,
    )

    requested_time = parse_iso_timestamp(time_utc)
    stamp = requested_time.strftime("%Y%m%dT%H%M%S")
    default_dir = Path("outputs") / "bow_shock"
    image_path = (
        Path(output_path)
        if output_path is not None
        else default_dir / f"{stamp}_{model.name}.png"
    )
    json_path = (
        Path(context_json_path)
        if context_json_path is not None
        else image_path.with_suffix(".json")
    )
    image_path = image_path.expanduser().resolve()
    json_path = json_path.expanduser().resolve()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    position_rm = np.asarray(context.position_mso_rm, dtype=float)
    boundary_rm = (
        None
        if context.boundary_position_mso_rm is None
        else np.asarray(context.boundary_position_mso_rm, dtype=float)
    )
    maximum_x = model.nose_x_rm() if x_max_rm is None else min(float(x_max_rm), model.nose_x_rm())
    x_axis = np.linspace(float(x_min_rm), maximum_x, 900)

    figure = plt.figure(figsize=(16.0, 5.6))
    xy_axis = figure.add_subplot(1, 3, 1)
    xz_axis = figure.add_subplot(1, 3, 2)
    surface_axis = figure.add_subplot(1, 3, 3, projection="3d")
    _draw_cross_section(
        xy_axis,
        model,
        x_axis,
        0.0,
        np.pi,
        position_rm[0],
        position_rm[1],
        boundary_rm,
        1,
        "Equatorial cross-section",
        "Y",
    )
    _draw_cross_section(
        xz_axis,
        model,
        x_axis,
        np.pi / 2.0,
        3.0 * np.pi / 2.0,
        position_rm[0],
        position_rm[2],
        boundary_rm,
        2,
        "Meridional cross-section",
        "Z",
    )
    _draw_3d(surface_axis, surface, position_rm, boundary_rm)

    handles, labels = xy_axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    xy_axis.legend(unique.values(), unique.keys(), loc="lower left", frameon=False, fontsize=8)
    offset_text = (
        f"{context.radial_offset_km:+.0f} km"
        if np.isfinite(context.radial_offset_km)
        else "not intersected on radial ray"
    )
    figure.suptitle(
        f"Mars bow shock at {context.sample_time_utc} | {model.display_name}\n"
        f"MAVEN: {context.location}, SZA={context.sza_deg:.1f} deg, radial offset={offset_text}",
        fontsize=12,
    )
    figure.subplots_adjust(left=0.055, right=0.975, bottom=0.12, top=0.80, wspace=0.28)
    figure.savefig(image_path, dpi=180)
    plt.close(figure)

    payload = context.to_dict()
    payload["model"] = model.metadata()
    payload["image_path"] = str(image_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)

    return image_path, json_path, payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a Mars bow-shock model and the nearest MAVEN MSO position at a UTC time."
    )
    parser.add_argument("--time", required=True, help="UTC time, for example 2024-11-07T02:15:00.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        choices=[item["name"] for item in list_models()],
        help="Bow-shock model.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Local MAVEN data root.")
    parser.add_argument("--output", help="PNG output path.")
    parser.add_argument("--context-json", help="JSON output path. Defaults to the PNG path with .json.")
    parser.add_argument("--x-min-rm", type=float, default=-3.0, help="Tailward plot limit in Mars radii.")
    parser.add_argument("--x-max-rm", type=float, help="Sunward plot limit, capped at the model nose.")
    parser.add_argument("--max-mag-delta-seconds", type=float, default=5.0)
    parser.add_argument("--boundary-tolerance-km", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    image_path, json_path, context = plot_bow_shock(
        time_utc=args.time,
        model_name=args.model,
        data_root=args.data_root,
        output_path=args.output,
        context_json_path=args.context_json,
        x_min_rm=args.x_min_rm,
        x_max_rm=args.x_max_rm,
        max_mag_delta_seconds=args.max_mag_delta_seconds,
        boundary_tolerance_km=args.boundary_tolerance_km,
    )
    print(f"Wrote image: {image_path}")
    print(f"Wrote context: {json_path}")
    print(
        "Bow-shock context: "
        f"location={context['location']}, "
        f"inside_bow_shock={context['inside_bow_shock']}, "
        f"radial_offset_km={context['radial_offset_km']}"
    )


if __name__ == "__main__":
    main()
