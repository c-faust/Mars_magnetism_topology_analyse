from __future__ import annotations
"""
Render the magnetic-topology data panels as a static PNG.

This is the Python counterpart of `magnetic_topology_data_panels.html`: it reads
either a `topology_summary.json`-shaped dictionary or the local MAVEN daily data,
picks a target time, and draws the same science context panels around that time.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from analyze_magnetic_topology import (
    build_mag_context,
    derive_pitch_bins,
    iter_days,
    load_mag_day,
    load_static_context,
    resolve_daily_files,
    select_time_indices,
)
from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from process_maven_spectra import load_pad_data


MARS_RADIUS_KM = 3389.5
LINE_COLORS = {"bx": "#cc4338", "by": "#2674c8", "bz": "#3a8a53", "bmag": "#6e5b4f"}
PAD_CMAP = "turbo"
FLUX_CMAP = "magma"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "maven_data_panels"
DEFAULT_PAD_ENERGY_BAND_EV = (111.0, 140.0)


def log_step(message: str) -> None:
    print(f"[data-panels] {datetime.now().isoformat(timespec='seconds')} | {message}", flush=True)


def unix_to_matplotlib_dates(values: np.ndarray) -> np.ndarray:
    return mdates.date2num([datetime.fromtimestamp(float(value), tz=timezone.utc) for value in values])


def iso_to_unix(value: str) -> float:
    return parse_iso_timestamp(value).timestamp()


def finite_array(values) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=float)


def nearest_sample_index(samples: list[dict], target_time: datetime) -> int:
    if not samples:
        raise ValueError("The data-panel summary does not contain any samples.")
    sample_times = np.asarray([iso_to_unix(sample["target_time"]) for sample in samples], dtype=float)
    return int(np.argmin(np.abs(sample_times - target_time.timestamp())))


def validate_energy_band(energy_band_eV: tuple[float, float]) -> tuple[float, float]:
    low, high = (float(energy_band_eV[0]), float(energy_band_eV[1]))
    if low <= 0.0 or high <= 0.0 or low >= high:
        raise ValueError("pad-energy-band must satisfy 0 < LOW_EV < HIGH_EV.")
    return low, high


def energy_band_label(energy_band_eV: tuple[float, float]) -> str:
    low, high = validate_energy_band(energy_band_eV)
    return f"{low:g}-{high:g} eV"


def default_pad_energy_band(energy: np.ndarray, requested_band_eV: tuple[float, float]) -> tuple[float, float]:
    low, high = validate_energy_band(requested_band_eV)
    band_mask = (energy >= low) & (energy <= high)
    if np.any(band_mask):
        return low, high
    fallback = (100.0, 150.0)
    fallback_mask = (energy >= fallback[0]) & (energy <= fallback[1])
    if np.any(fallback_mask):
        return fallback
    raise ValueError(
        f"No SWE energy bins were found in {energy_band_label((low, high))} "
        f"or fallback {energy_band_label(fallback)}."
    )


def build_swe_context_for_band(
    pad_data: dict,
    start: datetime,
    end: datetime,
    pad_energy_band_eV: tuple[float, float] = DEFAULT_PAD_ENERGY_BAND_EV,
) -> dict | None:
    times = np.asarray(pad_data["times"], dtype=float)
    time_mask = (times >= start.timestamp()) & (times <= end.timestamp())
    if not np.any(time_mask):
        return None

    indices = np.where(time_mask)[0]
    flux = np.asarray(pad_data["flux"], dtype=float)[indices]
    energy = np.asarray(pad_data["energy"], dtype=float)
    actual_band_eV = default_pad_energy_band(energy, pad_energy_band_eV)
    band_mask_energy = (energy >= actual_band_eV[0]) & (energy <= actual_band_eV[1])

    omni_spectrum = np.nanmean(flux, axis=1)
    band_flux = flux[:, :, band_mask_energy]
    valid_counts = np.sum(np.isfinite(band_flux), axis=2)
    band_sum = np.nansum(band_flux, axis=2)
    pad_band = np.divide(
        band_sum,
        valid_counts,
        out=np.full_like(band_sum, np.nan, dtype=float),
        where=valid_counts > 0,
    )
    pitch_bins = derive_pitch_bins(pad_data, indices, band_mask_energy)
    context = {
        "times_unix": times[indices].tolist(),
        "energy_eV": energy.tolist(),
        "pitch_deg": np.asarray(pitch_bins, dtype=float).tolist(),
        "omni_eflux": omni_spectrum.tolist(),
        "pad_eflux": pad_band.tolist(),
        "pad_energy_band_eV": list(actual_band_eV),
    }
    if actual_band_eV == DEFAULT_PAD_ENERGY_BAND_EV:
        context["pad_111_140_eflux"] = pad_band.tolist()
    return context


def concat_timeseries(parts: list[dict], keys: list[str]) -> dict | None:
    if not parts:
        return None
    merged: dict[str, list] = {key: [] for key in keys}
    static_keys = [key for key in parts[0].keys() if key not in merged]
    for part in parts:
        for key in keys:
            merged[key].extend(part[key])
    for key in static_keys:
        merged[key] = parts[0][key]
    return merged


def sample_altitude_entries(
    mag_data_ss: dict,
    start: datetime,
    end: datetime,
    step_seconds: int,
) -> list[dict]:
    times = np.asarray(mag_data_ss["times"], dtype=float)
    samples: list[dict] = []
    for index in select_time_indices(times, start, end, step_seconds):
        sample_time = datetime.fromtimestamp(float(times[index]), tz=timezone.utc)
        position_km = np.asarray(mag_data_ss["data"][index, mag_data_ss["pos_indices"]], dtype=float)
        altitude_km = float(np.linalg.norm(position_km) - MARS_RADIUS_KM)
        samples.append(
            {
                "target_time": sample_time.isoformat(timespec="seconds"),
                "topology": "not_computed",
                "altitude_km": altitude_km,
                "altitude_rm": altitude_km / MARS_RADIUS_KM,
                "position_km": position_km.tolist(),
                "position_rm": (position_km / MARS_RADIUS_KM).tolist(),
            }
        )
    return samples


def build_data_panel_summary_from_data(
    target_time: datetime,
    window_minutes: float,
    step_seconds: int,
    data_root: Path = DEFAULT_DATA_ROOT,
    auto_download_missing_data: bool = False,
    pad_energy_band_eV: tuple[float, float] = DEFAULT_PAD_ENERGY_BAND_EV,
) -> dict:
    if window_minutes <= 0:
        raise ValueError("window-minutes must be positive.")
    if step_seconds <= 0:
        raise ValueError("step-seconds must be positive.")

    half_window = timedelta(minutes=window_minutes / 2.0)
    start = target_time - half_window
    end = target_time + half_window
    resolved_files = resolve_daily_files(
        start=start,
        end=end,
        data_root=data_root,
        pad_file=None,
        mag_file=None,
        auto_download_missing_data=auto_download_missing_data,
    )

    static_parts: list[dict] = []
    swe_parts: list[dict] = []
    mag_parts: list[dict] = []
    samples: list[dict] = []
    input_files: dict[str, dict[str, str]] = {}

    for day in iter_days(start, end):
        files = resolved_files[day]
        input_files[day.isoformat()] = {key: str(path) for key, path in files.items()}
        day_start = max(start, datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc))
        day_end = min(end, datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc))

        pad_data = load_pad_data(files["pad"])
        swe_context = build_swe_context_for_band(pad_data, day_start, day_end, pad_energy_band_eV)
        if swe_context:
            swe_parts.append(swe_context)

        mag_data_ss = load_mag_day(files["mag_ss"])
        mag_context = build_mag_context(mag_data_ss, day_start, day_end)
        if mag_context:
            mag_parts.append(mag_context)
        samples.extend(sample_altitude_entries(mag_data_ss, day_start, day_end, step_seconds))

        static_context = load_static_context(files["sta_c6"], day_start, day_end)
        if static_context:
            static_parts.append(static_context)

    if not samples:
        raise ValueError("No MAG samples were found in the requested event window.")

    return {
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": end.isoformat(timespec="seconds"),
        "step_seconds": step_seconds,
        "topology_computed": False,
        "source": "local_data",
        "input_files": input_files,
        "context_overview": {
            "window_seconds": window_minutes * 60.0,
            "static": concat_timeseries(static_parts, ["times_unix", "energy_eflux", "mass_eflux"]),
            "mag": concat_timeseries(mag_parts, ["times_unix", "bx_nT", "by_nT", "bz_nT", "bmag_nT"]),
            "swe": concat_timeseries(swe_parts, ["times_unix", "omni_eflux", "pad_eflux"]),
        },
        "samples": samples,
    }


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


def positive_log_norm(matrix, lower_percentile: float = 2.0, upper_percentile: float = 98.0) -> LogNorm | None:
    values = np.asarray(matrix, dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return None
    vmin = float(np.nanpercentile(positive, lower_percentile))
    vmax = float(np.nanpercentile(positive, upper_percentile))
    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = float(np.nanmin(positive))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(positive))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def resolve_local_source(path_value: str | None, data_root: Path = DEFAULT_DATA_ROOT) -> Path | None:
    if not path_value:
        return None
    source = Path(path_value).expanduser()
    if source.exists():
        return source.resolve()
    matches = sorted(data_root.rglob(source.name))
    return matches[0].resolve() if matches else None


def reload_static_context_for_window(static: dict, center_unix: float, window_seconds: float) -> dict:
    source = resolve_local_source(static.get("source_file"))
    if source is None:
        return static
    start = datetime.fromtimestamp(center_unix - window_seconds / 2.0, tz=timezone.utc)
    end = datetime.fromtimestamp(center_unix + window_seconds / 2.0, tz=timezone.utc)
    reloaded = load_static_context(source, start, end)
    return reloaded or static


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


def resolve_pad_panel_data(
    swe: dict,
    requested_band_eV: tuple[float, float] = DEFAULT_PAD_ENERGY_BAND_EV,
) -> tuple[np.ndarray, tuple[float, float]]:
    if "pad_eflux" in swe:
        band = swe.get("pad_energy_band_eV") or requested_band_eV
        return np.asarray(swe.get("pad_eflux", []), dtype=float), validate_energy_band(tuple(band))
    if "pad_111_140_eflux" in swe:
        return np.asarray(swe.get("pad_111_140_eflux", []), dtype=float), DEFAULT_PAD_ENERGY_BAND_EV
    return np.asarray([], dtype=float), validate_energy_band(requested_band_eV)


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
    ax.set_title(title, loc="left", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if log_y:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    return mesh


def plot_line_panel(ax, times_unix, traces: list[tuple[str, str, np.ndarray]], title: str, ylabel: str, y_range=None):
    times = finite_array(times_unix)
    if len(times) == 0 or not traces:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, loc="left", fontsize=18)
        return
    x = [datetime.fromtimestamp(float(value), tz=timezone.utc) for value in times]
    for label, color, values in traces:
        ax.plot(x, values, color=color, linewidth=1.2, label=label)
    ax.set_title(title, loc="left", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if y_range:
        ax.set_ylim(*y_range)
    ax.legend(loc="upper right", fontsize=18, frameon=False, ncol=min(3, len(traces)))
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
    pad_energy_band_eV: tuple[float, float] = DEFAULT_PAD_ENERGY_BAND_EV,
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
        8,
        1,
        figsize=(12.5, 19.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 1.15, 0.8, 0.8, 0.75, 1.05, 0.55]},
    )
    axes_flat = axes.ravel()

    static = reload_static_context_for_window(overview.get("static") or {}, center_unix, window_seconds)
    static_indices = window_indices(static.get("times_unix"), center_unix, window_seconds)
    static_energy_matrix = np.asarray(static.get("energy_eflux", []), dtype=float)[static_indices] if len(static_indices) else []
    mesh = plot_heatmap(
        axes_flat[0],
        static_energy_matrix,
        np.asarray(static.get("times_unix", []), dtype=float)[static_indices] if len(static_indices) else [],
        static.get("energy_eV", []),
        "STATIC Energy",
        "Energy (eV)",
        log_y=True,
        norm=positive_log_norm(static_energy_matrix),
    )
    if mesh:
        fig.colorbar(mesh, ax=axes_flat[0], pad=0.01, label="eflux")

    static_mass_matrix = np.asarray(static.get("mass_eflux", []), dtype=float)[static_indices] if len(static_indices) else []
    mesh = plot_heatmap(
        axes_flat[1],
        static_mass_matrix,
        np.asarray(static.get("times_unix", []), dtype=float)[static_indices] if len(static_indices) else [],
        static.get("mass_amu", []),
        "STATIC Mass",
        "Mass (amu)",
        norm=positive_log_norm(static_mass_matrix),
    )
    if mesh:
        fig.colorbar(mesh, ax=axes_flat[1], pad=0.01, label="eflux")

    swe = overview.get("swe") or {}
    swe_indices = window_indices(swe.get("times_unix"), center_unix, window_seconds)
    swe_times = np.asarray(swe.get("times_unix", []), dtype=float)[swe_indices] if len(swe_indices) else []
    electron_energy_matrix = np.asarray(swe.get("omni_eflux", []), dtype=float)[swe_indices] if len(swe_indices) else []
    mesh = plot_heatmap(
        axes_flat[2],
        electron_energy_matrix,
        swe_times,
        swe.get("energy_eV", []),
        "SWE Electron Energy",
        "Energy (eV)",
        log_y=True,
        norm=LogNorm(vmin=1e3, vmax=1e9),
        cmap=FLUX_CMAP,
    )
    if mesh:
        fig.colorbar(mesh, ax=axes_flat[2], pad=0.01, label="eflux")

    mag = overview.get("mag") or {}
    mag_indices = window_indices(mag.get("times_unix"), center_unix, window_seconds)
    mag_times = np.asarray(mag.get("times_unix", []), dtype=float)[mag_indices] if len(mag_indices) else []
    plot_line_panel(
        axes_flat[3],
        mag_times,
        [("|B|", LINE_COLORS["bmag"], np.asarray(mag.get("bmag_nT", []), dtype=float)[mag_indices])],
        "|B|",
        "nT",
        y_range=(0.0, 50.0),
    )
    plot_line_panel(
        axes_flat[4],
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
        axes_flat[5],
        sample_times[sample_indices] if len(sample_indices) else [],
        [("Altitude", "#9a5f2f", np.asarray([sample_altitude_km(samples[i]) for i in sample_indices], dtype=float))],
        "Altitude",
        "km",
    )

    pad_all, resolved_pad_band_eV = resolve_pad_panel_data(swe, pad_energy_band_eV)
    pad_matrix = pad_all[swe_indices] if len(swe_indices) and pad_all.size else []
    mesh = plot_heatmap(
        axes_flat[6],
        pad_matrix,
        swe_times,
        swe.get("pitch_deg", []),
        f"SWE PAD ({energy_band_label(resolved_pad_band_eV)})",
        "Pitch angle (deg)",
        norm=positive_log_norm(pad_matrix),
        cmap=PAD_CMAP,
    )
    if mesh:
        fig.colorbar(mesh, ax=axes_flat[6], pad=0.01, label="eflux")

    axes_flat[7].axis("off")
    axes_flat[7].text(
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
        transform=axes_flat[7].transAxes,
    )

    for ax in axes_flat[:7]:
        ax.set_xlim(window_start, window_end)
        mark_target_time(ax, target_unix)
        ax.grid(True, linestyle=":", alpha=0.25)
    for ax in axes_flat[:6]:
        ax.tick_params(labelbottom=False)
    fig.suptitle("MAVEN Magnetic Topology Data Panels", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "selected_index": selected_index,
        "selected_time": selected.get("target_time"),
        "pad_energy_band_eV": list(resolved_pad_band_eV),
        "output_path": str(output_path),
    }


def default_event_output_path(output_root: Path, target_time: datetime) -> Path:
    return output_root / target_time.strftime("%Y%m%dT%H%M%S") / "maven_data_panels.png"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render MAVEN data panels for one event time.")
    parser.add_argument(
        "--summary-json",
        help="Optional path to topology_summary.json or data_panel_context_summary.json. If omitted, local data files are used directly.",
    )
    parser.add_argument("--time", required=True, help="UTC target time.")
    parser.add_argument("--window-minutes", type=float, default=20.0)
    parser.add_argument("--step-seconds", type=int, default=60, help="Cadence for altitude samples when reading local data.")
    parser.add_argument(
        "--pad-energy-band",
        nargs=2,
        type=float,
        default=DEFAULT_PAD_ENERGY_BAND_EV,
        metavar=("LOW_EV", "HIGH_EV"),
        help="Electron energy band averaged into the SWE PAD panel.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for local MAVEN data.")
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Download missing SWE/STATIC/MAG daily files. By default only local files are used.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory used for per-event output folders when --output is not supplied.",
    )
    parser.add_argument("--output", help="Explicit PNG output path. Overrides --output-root.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    target_time = parse_iso_timestamp(args.time)
    pad_energy_band_eV = validate_energy_band(tuple(args.pad_energy_band))
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        log_step(f"Loading summary context: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        data_root = Path(args.data_root).expanduser().resolve()
        log_step(f"Building panel context directly from local data: {data_root}")
        summary = build_data_panel_summary_from_data(
            target_time=target_time,
            window_minutes=args.window_minutes,
            step_seconds=args.step_seconds,
            data_root=data_root,
            auto_download_missing_data=args.auto_download,
            pad_energy_band_eV=pad_energy_band_eV,
        )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_event_output_path(Path(args.output_root).expanduser().resolve(), target_time)
    )
    log_step(f"Writing data panels to: {output_path}")
    result = plot_data_panels(
        summary=summary,
        target_time=target_time,
        output_path=output_path,
        window_minutes=args.window_minutes,
        pad_energy_band_eV=pad_energy_band_eV,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
