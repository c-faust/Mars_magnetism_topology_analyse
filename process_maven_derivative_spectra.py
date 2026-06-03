from __future__ import annotations
"""
Single-time MAVEN energy-derivative spectrum processing.

This script mirrors `process_maven_spectra.py`, but plots normalized
d(log Flux)/d(log Energy) spectra instead of raw directional flux spectra. Outputs are written
to the same timestamped directory used by `process_maven_spectra.py`.

Example:
python process_maven_derivative_spectra.py --time 2024-11-07T02:15:00
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from process_maven_spectra import (
    DEFAULT_SCPOT_MIN_FLAG,
    build_output_dir,
    compute_directional_spectra,
    format_unix_time,
    infer_daily_file,
    load_pad_data,
    nearest_mag_vector,
    optional_nearest_spacecraft_potential,
)


DEFAULT_NORMALIZATION = "zscore"


@dataclass(frozen=True)
class DerivativeSpectrumResult:
    """Compact output object written to JSON by the derivative pipeline."""

    target_time: str
    swe_time: str
    mag_time: str
    lpw_time: str | None
    pad_file: str
    mag_file: str
    lpw_file: str | None
    magnetic_field_nT: list[float]
    spacecraft_potential_V: float | None
    spacecraft_potential_marker_eV: float | None
    forward_pitch_max_deg: float
    backward_pitch_min_deg: float
    normalization: str
    derivative_method: str
    energy_eV: list[float]
    forward_flux: list[float | None]
    backward_flux: list[float | None]
    forward_dlogflux_dlogenergy: list[float | None]
    backward_dlogflux_dlogenergy: list[float | None]
    forward_normalized_dlogflux_dlogenergy: list[float | None]
    backward_normalized_dlogflux_dlogenergy: list[float | None]
    forward_pitch_bins_deg: list[float]
    backward_pitch_bins_deg: list[float]


def finite_or_none(values: np.ndarray) -> list[float | None]:
    result: list[float | None] = []
    for value in np.asarray(values, dtype=float):
        result.append(float(value) if np.isfinite(value) else None)
    return result


def flux_energy_derivative(energy_eV: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return d(log Flux)/d(log Energy) on the sorted energy grid."""

    energy = np.asarray(energy_eV, dtype=float)
    flux = np.asarray(flux, dtype=float)
    order = np.argsort(energy)
    sorted_energy = energy[order]
    sorted_flux = flux[order]
    derivative = np.full(sorted_flux.shape, np.nan, dtype=float)

    valid = np.isfinite(sorted_energy) & np.isfinite(sorted_flux) & (sorted_energy > 0.0) & (sorted_flux > 0.0)
    if np.count_nonzero(valid) < 2:
        return sorted_energy, derivative

    valid_log_energy = np.log10(sorted_energy[valid])
    valid_log_flux = np.log10(sorted_flux[valid])
    unique_log_energy, unique_indices = np.unique(valid_log_energy, return_index=True)
    if unique_log_energy.size < 2:
        return sorted_energy, derivative

    unique_log_flux = valid_log_flux[unique_indices]
    derivative[valid] = np.gradient(unique_log_flux, unique_log_energy, edge_order=1)
    return sorted_energy, derivative


def normalize_one_spectrum(values: np.ndarray, method: str) -> np.ndarray:
    """Normalize one derivative spectrum without inventing values for missing bins."""

    data = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(data)
    normalized = np.full(data.shape, np.nan, dtype=float)
    if not np.any(valid):
        return normalized

    selected = data[valid]
    if method == "none":
        normalized[valid] = selected
        return normalized
    if method == "zscore":
        scale = float(np.nanstd(selected))
        normalized[valid] = (selected - float(np.nanmean(selected))) / (scale if scale > 0.0 else 1.0)
        return normalized
    if method == "minmax":
        lower = float(np.nanmin(selected))
        upper = float(np.nanmax(selected))
        span = upper - lower
        normalized[valid] = (selected - lower) / (span if span > 0.0 else 1.0)
        return normalized
    if method == "l2":
        norm = float(np.linalg.norm(selected))
        normalized[valid] = selected / (norm if norm > 0.0 else 1.0)
        return normalized
    if method == "maxabs":
        scale = float(np.nanmax(np.abs(selected)))
        normalized[valid] = selected / (scale if scale > 0.0 else 1.0)
        return normalized
    raise ValueError(f"Unsupported derivative normalization method: {method}")


def plot_derivative_spectra(
    energy: np.ndarray,
    forward_normalized_derivative: np.ndarray,
    backward_normalized_derivative: np.ndarray,
    output_path: Path,
    spacecraft_potential_marker_eV: float | None = None,
    spacecraft_potential_V: float | None = None,
    forward_pitch_max_deg: float = 30.0,
    backward_pitch_min_deg: float = 150.0,
    normalization: str = DEFAULT_NORMALIZATION,
) -> None:
    energy = np.asarray(energy, dtype=float)
    forward = np.asarray(forward_normalized_derivative, dtype=float)
    backward = np.asarray(backward_normalized_derivative, dtype=float)
    positive_energy = energy[np.isfinite(energy) & (energy > 0.0)]
    x_limits = (float(np.nanmin(positive_energy)), float(np.nanmax(positive_energy))) if positive_energy.size else None

    plt.figure(figsize=(8, 5))
    if np.any(np.isfinite(forward)):
        plt.semilogx(
            energy,
            forward,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"Parallel d(logF)/d(logE), pitch < {forward_pitch_max_deg:g} deg",
        )
    if np.any(np.isfinite(backward)):
        plt.semilogx(
            energy,
            backward,
            marker="s",
            markersize=3,
            linewidth=1.2,
            label=f"Anti-parallel d(logF)/d(logE), pitch > {backward_pitch_min_deg:g} deg",
        )
    if (
        spacecraft_potential_marker_eV is not None
        and np.isfinite(spacecraft_potential_marker_eV)
        and spacecraft_potential_marker_eV > 0.0
    ):
        label_value = (
            spacecraft_potential_marker_eV
            if spacecraft_potential_V is None or not np.isfinite(spacecraft_potential_V)
            else spacecraft_potential_V
        )
        plt.axvline(
            spacecraft_potential_marker_eV,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"LPW Vsc = {label_value:.2f} V",
        )
    if x_limits is not None:
        plt.xlim(*x_limits)
    plt.axhline(0.0, color="0.35", linewidth=0.8, alpha=0.7)
    plt.xlabel("Energy (eV)")
    plt.ylabel(f"Normalized d(logF)/d(logE) ({normalization})")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def process_target_time(
    target_time: datetime,
    pad_file: Path,
    mag_file: Path,
    lpw_file: Path | None,
    output_root: Path,
    forward_pitch_max_deg: float = 30.0,
    backward_pitch_min_deg: float = 150.0,
    spacecraft_potential_min_flag: float = DEFAULT_SCPOT_MIN_FLAG,
    normalization: str = DEFAULT_NORMALIZATION,
) -> DerivativeSpectrumResult:
    pad_data = load_pad_data(pad_file)
    forward_flux, backward_flux, pad_index, forward_bins, backward_bins = compute_directional_spectra(
        pad_data,
        target_time,
        forward_pitch_max_deg=forward_pitch_max_deg,
        backward_pitch_min_deg=backward_pitch_min_deg,
    )
    magnetic_field, mag_time = nearest_mag_vector(mag_file, target_time)
    spacecraft_potential, lpw_time, spacecraft_potential_marker_eV = optional_nearest_spacecraft_potential(
        lpw_file,
        target_time,
        min_flag=spacecraft_potential_min_flag,
    )

    energy, forward_derivative = flux_energy_derivative(pad_data["energy"], forward_flux)
    _, backward_derivative = flux_energy_derivative(pad_data["energy"], backward_flux)
    forward_normalized = normalize_one_spectrum(forward_derivative, normalization)
    backward_normalized = normalize_one_spectrum(backward_derivative, normalization)

    output_dir = build_output_dir(target_time, output_root)
    plot_path = output_dir / "directional_electron_derivative_spectra.png"
    plot_derivative_spectra(
        energy,
        forward_normalized,
        backward_normalized,
        plot_path,
        spacecraft_potential_marker_eV=spacecraft_potential_marker_eV,
        spacecraft_potential_V=spacecraft_potential,
        forward_pitch_max_deg=forward_pitch_max_deg,
        backward_pitch_min_deg=backward_pitch_min_deg,
        normalization=normalization,
    )

    result = DerivativeSpectrumResult(
        target_time=target_time.isoformat(timespec="seconds"),
        swe_time=format_unix_time(pad_data["times"][pad_index]),
        mag_time=mag_time,
        lpw_time=lpw_time,
        pad_file=str(pad_file),
        mag_file=str(mag_file),
        lpw_file=str(lpw_file) if lpw_file is not None else None,
        magnetic_field_nT=magnetic_field.tolist(),
        spacecraft_potential_V=float(spacecraft_potential) if spacecraft_potential is not None else None,
        spacecraft_potential_marker_eV=(
            float(spacecraft_potential_marker_eV) if spacecraft_potential_marker_eV is not None else None
        ),
        forward_pitch_max_deg=float(forward_pitch_max_deg),
        backward_pitch_min_deg=float(backward_pitch_min_deg),
        normalization=normalization,
        derivative_method="np.gradient(log10(flux), log10(energy_eV)), edge_order=1",
        energy_eV=energy.tolist(),
        forward_flux=finite_or_none(np.asarray(forward_flux, dtype=float)[np.argsort(pad_data["energy"])]),
        backward_flux=finite_or_none(np.asarray(backward_flux, dtype=float)[np.argsort(pad_data["energy"])]),
        forward_dlogflux_dlogenergy=finite_or_none(forward_derivative),
        backward_dlogflux_dlogenergy=finite_or_none(backward_derivative),
        forward_normalized_dlogflux_dlogenergy=finite_or_none(forward_normalized),
        backward_normalized_dlogflux_dlogenergy=finite_or_none(backward_normalized),
        forward_pitch_bins_deg=forward_bins.tolist(),
        backward_pitch_bins_deg=backward_bins.tolist(),
    )

    (output_dir / "derivative_spectrum_summary.json").write_text(
        json.dumps(result.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process MAVEN PAD and MAG files into a normalized d(logF)/d(logE) spectrum.")
    parser.add_argument("--time", required=True, help="Target timestamp, for example 2024-11-07T12:00:00.")
    parser.add_argument("--pad-file", help="Path to the SWE PAD CDF file.")
    parser.add_argument("--mag-file", help="Path to the MAG STS file.")
    parser.add_argument("--lpw-file", help="Path to the LPW mrgscpot CDF file.")
    parser.add_argument(
        "--spacecraft-potential-min-flag",
        type=float,
        default=DEFAULT_SCPOT_MIN_FLAG,
        help="Minimum LPW mrgscpot quality flag used for spacecraft-potential samples.",
    )
    parser.add_argument(
        "--forward-pitch-max",
        type=float,
        default=30.0,
        help="Upper pitch-angle bound, in degrees, used for the forward spectrum.",
    )
    parser.add_argument(
        "--backward-pitch-min",
        type=float,
        default=150.0,
        help="Lower pitch-angle bound, in degrees, used for the backward spectrum.",
    )
    parser.add_argument(
        "--normalization",
        choices=("zscore", "minmax", "l2", "maxabs", "none"),
        default=DEFAULT_NORMALIZATION,
        help="Per-direction normalization applied after d(logF)/d(logE) is computed.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for downloaded data.")
    parser.add_argument(
        "--output-root",
        default=str(Path("outputs") / "maven_spectra"),
        help="Directory used to store figures and summary files.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    target_time = parse_iso_timestamp(args.time)
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    day = target_time

    pad_file = Path(args.pad_file).expanduser().resolve() if args.pad_file else infer_daily_file(
        data_root=data_root,
        instrument="swe",
        datatype_alias="svypad",
        day=day,
        extension="cdf",
    )
    mag_file = Path(args.mag_file).expanduser().resolve() if args.mag_file else infer_daily_file(
        data_root=data_root,
        instrument="mag",
        datatype_alias="ss1s",
        day=day,
        extension="sts",
    )
    if args.lpw_file:
        lpw_file = Path(args.lpw_file).expanduser().resolve()
    else:
        try:
            lpw_file = infer_daily_file(
                data_root=data_root,
                instrument="lpw",
                datatype_alias="mrgscpot",
                day=day,
                extension="cdf",
            )
        except FileNotFoundError as exc:
            print(f"[derivative_spectra] {exc}", flush=True)
            lpw_file = None

    result = process_target_time(
        target_time=target_time,
        pad_file=pad_file,
        mag_file=mag_file,
        lpw_file=lpw_file,
        output_root=output_root,
        forward_pitch_max_deg=args.forward_pitch_max,
        backward_pitch_min_deg=args.backward_pitch_min,
        spacecraft_potential_min_flag=args.spacecraft_potential_min_flag,
        normalization=args.normalization,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
