from __future__ import annotations
"""
Render the magnetic-topology data panels as a static PNG.

This is the Python counterpart of `magnetic_topology_data_panels.html`: it reads
`topology_summary.json`, picks a target time, and draws the same science context
panels around that time.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from download_maven_data import parse_iso_timestamp


MARS_RADIUS_KM = 3389.5
LINE_COLORS = {"bx": "#cc4338", "by": "#2674c8", "bz": "#3a8a53", "bmag": "#6e5b4f"}
PAD_CMAP = "turbo"
FLUX_CMAP = "magma"


def unix_to_matplotlib_dates(values: np.ndarray) -> np.ndarray:
    return mdates.date2num([datetime.fromtimestamp(float(value), tz=timezone.utc) for value in values])


def iso_to_unix(value: str) -> float:
    return parse_iso_timestamp(value).timestamp()


def finite_array(values) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=float)


def nearest_sample_index(samples: list[dict], target_time: datetime) -> int:
    if not samples:
        raise ValueError("topology_summary.json does not contain any samples.")
    sample_times = np.asarray([iso_to_unix(sample["target_time"]) for sample in samples], dtype=float)
    return int(np.argmin(np.abs(sample_times - target_time.timestamp())))


def window_indices(times_unix, center_unix: float, window_seconds: float) -> np.ndarray:
    times = finite_array(times_unix)
    return np.where((times >= center_unix - window_seconds / 2.0) & (times <= center_unix + window_seconds / 2.0))[0]


def axis_edges(values: np.ndarray, log_scale: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if values.size == 1:
        delta = values[0] * 0.05 if log_scale and values[0] > 0 else 0.5
        return np.array([values[0] - delta, values[0] + delta], dtype=float)
    if log_scale:
        safe = np.clip(values, 1e-12, None)
        log_values = np.log10(safe)
        mids = (log_values[:-1] + log_values[1:]) / 2.0
        first = log_values[0] - (mids[0] - log_values[0])
        last = log_values[-1] + (log_values[-1] - mids[-1])
        return 10.0 ** np.concatenate([[first], mids, [last]])
    mids = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def prepare_heatmap(matrix, y_values, log_y: bool = False) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(matrix, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if z.ndim != 2 or y.size == 0:
        return np.empty((0, 0)), y
    if z.shape[1] != y.size and z.shape[0] == y.size:
        z = z.T
    order = np.argsort(y)
    if log_y:
        order = order[y[order] > 0]
    y_sorted = y[order]
    return z[:, order].T, y_sorted


def sample_altitude_km(sample: dict) -> float:
    if sample.get("altitude_km") is not None:
        return float(sample["altitude_km"])
    if sample.get("position_km"):
        position = np.asarray(sample["position_km"], dtype=float)
        return float(np.linalg.norm(position) - MARS_RADIUS_KM)
    if sample.get("position_rm"):
        position = np.asarray(sample["position_rm"], dtype=float)
        return float((np.linalg.norm(position) - 1.0) * MARS_RADIUS_KM)
    return float("nan")


def plot_heatmap(
    ax,
    matrix,
    times_unix,
    y_values,
    title: str,
    ylabel: str,
    log_y: bool = False,
    norm=None,
    cmap: str = FLUX_CMAP,
):
    times = finite_array(times_unix)
    if len(times) == 0 or len(y_values) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return None
    z, y_sorted = prepare_heatmap(matrix, y_values, log_y=log_y)
    if z.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return None

    time_edges = axis_edges(unix_to_matplotlib_dates(times), log_scale=False)
    y_edges = axis_edges(y_sorted, log_scale=log_y)
    mesh = ax.pcolormesh(time_edges, y_edges, z, shading="auto", cmap=cmap, norm=norm)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    return mesh


def plot_line_panel(ax, times_unix, traces: list[tuple[str, str, np.ndarray]], title: str, ylabel: str, y_range=None):
    times = finite_array(times_unix)
    if len(times) == 0 or not traces:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, loc="left", fontsize=10)
        return
    x = [datetime.fromtimestamp(float(value), tz=timezone.utc) for value in times]
    for label, color, values in traces:
        ax.plot(x, values, color=color, linewidth=1.2, label=label)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylabel(ylabel)
    if y_range:
        ax.set_ylim(*y_range)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=min(3, len(traces)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))


def mark_target_time(ax, target_unix: float) -> None:
    ax.axvline(
        mdates.date2num(datetime.fromtimestamp(float(target_unix), tz=timezone.utc)),
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
        zorder=10,
    )


def plot_data_panels(
    summary: dict,
    target_time: datetime,
    output_path: Path,
    window_minutes: float = 20.0,
) -> dict:
    samples = summary.get("samples", [])
    selected_index = nearest_sample_index(samples, target_time)
    selected = samples[selected_index]
    center_unix = iso_to_unix(selected["target_time"])
    target_unix = target_time.timestamp()
    window_seconds = window_minutes * 60.0
    window_start = mdates.date2num(datetime.fromtimestamp(center_unix - window_seconds / 2.0, tz=timezone.utc))
    window_end = mdates.date2num(datetime.fromtimestamp(center_unix + window_seconds / 2.0, tz=timezone.utc))
    overview = summary.get("context_overview", {})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        7,
        1,
        figsize=(12.5, 17.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 0.8, 0.8, 0.75, 1.05, 0.55]},
    )
    axes_flat = axes.ravel()

    static = overview.get("static") or {}
    static_indices = window_indices(static.get("times_unix"), center_unix, window_seconds)
    mesh = plot_heatmap(
        axes_flat[0],
        np.asarray(static.get("energy_eflux", []), dtype=float)[static_indices] if len(static_indices) else [],
        np.asarray(static.get("times_unix", []), dtype=float)[static_indices] if len(static_indices) else [],
        static.get("energy_eV", []),
        "STATIC Energy",
        "Energy (eV)",
        log_y=True,
        norm=LogNorm(vmin=1e3, vmax=1e9),
    )
    if mesh:
        fig.colorbar(mesh, ax=axes_flat[0], pad=0.01, label="eflux")

    mesh = plot_heatmap(
        axes_flat[1],
        np.asarray(static.get("mass_eflux", []), dtype=float)[static_indices] if len(static_indices) else [],
        np.asarray(static.get("times_unix", []), dtype=float)[static_indices] if len(static_indices) else [],
        static.get("mass_amu", []),
        "STATIC Mass",
        "Mass (amu)",
        norm=LogNorm(vmin=1e3, vmax=1e9),
    )
    if mesh:
        fig.colorbar(mesh, ax=axes_flat[1], pad=0.01, label="eflux")

    mag = overview.get("mag") or {}
    mag_indices = window_indices(mag.get("times_unix"), center_unix, window_seconds)
    mag_times = np.asarray(mag.get("times_unix", []), dtype=float)[mag_indices] if len(mag_indices) else []
    plot_line_panel(
        axes_flat[2],
        mag_times,
        [("|B|", LINE_COLORS["bmag"], np.asarray(mag.get("bmag_nT", []), dtype=float)[mag_indices])],
        "|B|",
        "nT",
        y_range=(0.0, 50.0),
    )
    plot_line_panel(
        axes_flat[3],
        mag_times,
        [
            ("Bx", LINE_COLORS["bx"], np.asarray(mag.get("bx_nT", []), dtype=float)[mag_indices]),
            ("By", LINE_COLORS["by"], np.asarray(mag.get("by_nT", []), dtype=float)[mag_indices]),
            ("Bz", LINE_COLORS["bz"], np.asarray(mag.get("bz_nT", []), dtype=float)[mag_indices]),
        ] if len(mag_indices) else [],
        "B_MSO",
        "nT",
        y_range=(-50.0, 50.0),
    )

    sample_times = np.asarray([iso_to_unix(sample["target_time"]) for sample in samples], dtype=float)
    sample_indices = window_indices(sample_times, center_unix, window_seconds)
    plot_line_panel(
        axes_flat[4],
        sample_times[sample_indices] if len(sample_indices) else [],
        [("Altitude", "#9a5f2f", np.asarray([sample_altitude_km(samples[i]) for i in sample_indices], dtype=float))],
        "Altitude",
        "km",
    )

    swe = overview.get("swe") or {}
    swe_indices = window_indices(swe.get("times_unix"), center_unix, window_seconds)
    swe_times = np.asarray(swe.get("times_unix", []), dtype=float)[swe_indices] if len(swe_indices) else []
    pad_matrix = np.asarray(swe.get("pad_111_140_eflux", []), dtype=float)[swe_indices] if len(swe_indices) else []
    mesh = plot_heatmap(
        axes_flat[5],
        pad_matrix,
        swe_times,
        swe.get("pitch_deg", []),
        "SWE PAD (111-140 eV)",
        "Pitch angle (deg)",
        norm=LogNorm(vmin=1e3, vmax=1e9),
        cmap=PAD_CMAP,
    )
    if mesh:
        fig.colorbar(mesh, ax=axes_flat[5], pad=0.01, label="eflux")

    axes_flat[6].axis("off")
    axes_flat[6].text(
        0.0,
        0.95,
        "Selected sample\n"
        f"{selected.get('target_time')}\n\n"
        f"Topology: {selected.get('topology', 'n/a')}\n"
        f"Altitude: {sample_altitude_km(selected):.1f} km\n"
        f"Window: {window_minutes:g} min",
        ha="left",
        va="top",
        fontsize=12,
        transform=axes_flat[6].transAxes,
    )

    for ax in axes_flat[:6]:
        ax.set_xlim(window_start, window_end)
        mark_target_time(ax, target_unix)
        ax.grid(True, linestyle=":", alpha=0.25)
    for ax in axes_flat[:5]:
        ax.tick_params(labelbottom=False)
    fig.suptitle("MAVEN Magnetic Topology Data Panels", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "selected_index": selected_index,
        "selected_time": selected.get("target_time"),
        "output_path": str(output_path),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render data panels from topology_summary.json.")
    parser.add_argument("--summary-json", required=True, help="Path to topology_summary.json.")
    parser.add_argument("--time", required=True, help="UTC target time.")
    parser.add_argument("--window-minutes", type=float, default=20.0)
    parser.add_argument("--output", default=str(Path("outputs") / "maven_data_panels.png"))
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    result = plot_data_panels(
        summary=summary,
        target_time=parse_iso_timestamp(args.time),
        output_path=Path(args.output).expanduser().resolve(),
        window_minutes=args.window_minutes,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
