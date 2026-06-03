from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from process_maven_spectra import infer_daily_file
from identify_magnetic_topology.PAD_score_method import (
    DEFAULT_ENERGY_RANGE_EV,
    coadd_pads,
    integrate_energy_band,
    load_pad_with_sigma,
    pad_pitch_centers_for_group,
)


DEFAULT_OUTPUT_ROOT = Path("outputs") / "identify_magnetic_topology" / "coadded_pad_results"


def iter_utc_days(start: datetime, end: datetime) -> list[datetime]:
    first = datetime.combine(start.date(), time.min, tzinfo=timezone.utc)
    last = datetime.combine(end.date(), time.min, tzinfo=timezone.utc)
    days = []
    current = first
    while current <= last:
        days.append(current)
        current += timedelta(days=1)
    return days


def format_unix_time(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")


def default_output_dir(output_root: Path, start: datetime, end: datetime) -> Path:
    return output_root / f"{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}"


def build_coadded_pad_tables(
    start: datetime,
    end: datetime,
    data_root: Path,
    energy_range_eV: tuple[float, float],
    energy_method: str,
    group_size: int,
    keep_partial: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pitch_rows = []
    group_rows = []

    for day in iter_utc_days(start, end):
        try:
            pad_file = infer_daily_file(data_root, "swe", "svypad", day, "cdf")
            pad_data = load_pad_with_sigma(pad_file)
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            print(f"[coadded-pad] skip SWE day {day.date()}: {exc}", flush=True)
            continue

        times = np.asarray(pad_data["times"], dtype=float)
        indices = np.where((times >= start.timestamp()) & (times <= end.timestamp()))[0]
        if indices.size == 0:
            continue

        daily_flux = np.asarray(pad_data["flux"], dtype=float)[indices]
        daily_sigma = None if pad_data["sigma"] is None else np.asarray(pad_data["sigma"], dtype=float)[indices]
        sigma_source = str(pad_data.get("sigma_source", "missing"))
        daily_times = times[indices]
        pad_flux, pad_sigma, selected_energy = integrate_energy_band(
            daily_flux,
            daily_sigma,
            np.asarray(pad_data["energy"], dtype=float),
            energy_range_eV=energy_range_eV,
            method=energy_method,
        )
        co_flux, co_sigma, co_times = coadd_pads(
            pad_flux,
            pad_sigma,
            daily_times,
            group_size=group_size,
            keep_partial=keep_partial,
        )

        for group_index, center_unix in enumerate(co_times):
            source_start = group_index * group_size
            source_end = min(source_start + group_size, indices.size)
            if source_end - source_start < group_size and not keep_partial:
                continue

            source_indices = indices[source_start:source_end]
            source_times = times[source_indices]
            pitch_centers = pad_pitch_centers_for_group(pad_data, source_indices, selected_energy)
            flux_row = co_flux[group_index]
            sigma_row = co_sigma[group_index]
            variance_row = sigma_row * sigma_row
            valid_bins = np.isfinite(flux_row) & np.isfinite(sigma_row)

            group_rows.append(
                {
                    "group_index": len(group_rows),
                    "center_time_unix": float(center_unix),
                    "center_time_utc": format_unix_time(center_unix),
                    "source_time_utc": "|".join(format_unix_time(value) for value in source_times),
                    "source_time_unix": "|".join(f"{float(value):.6f}" for value in source_times),
                    "source_sample_count": int(source_times.size),
                    "valid_pitch_bin_count": int(np.count_nonzero(valid_bins)),
                    "mean_flux_100_300": float(np.nanmean(flux_row)) if np.any(np.isfinite(flux_row)) else float("nan"),
                    "mean_sigma_100_300": float(np.nanmean(sigma_row)) if np.any(np.isfinite(sigma_row)) else float("nan"),
                    "mean_variance_100_300": float(np.nanmean(variance_row)) if np.any(np.isfinite(variance_row)) else float("nan"),
                    "sigma_source": sigma_source,
                    "source_file": str(pad_file),
                }
            )

            for pitch_index, (pitch, flux_value, sigma_value, variance_value) in enumerate(
                zip(pitch_centers, flux_row, sigma_row, variance_row)
            ):
                pitch_rows.append(
                    {
                        "group_index": len(group_rows) - 1,
                        "center_time_unix": float(center_unix),
                        "center_time_utc": format_unix_time(center_unix),
                        "pitch_bin_index": int(pitch_index),
                        "pitch_angle_deg": float(pitch),
                        "flux4_100_300": float(flux_value),
                        "sigma4_100_300": float(sigma_value),
                        "variance4_100_300": float(variance_value),
                        "sigma_source": sigma_source,
                        "source_sample_count": int(source_times.size),
                        "source_time_utc": "|".join(format_unix_time(value) for value in source_times),
                        "source_file": str(pad_file),
                    }
                )

    return pd.DataFrame(group_rows), pd.DataFrame(pitch_rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print 4-sample coadded MAVEN SWE PAD intermediate results.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--energy-range", nargs=2, type=float, default=DEFAULT_ENERGY_RANGE_EV, metavar=("LOW_EV", "HIGH_EV"))
    parser.add_argument("--energy-method", choices=("sum", "mean"), default="mean")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--keep-partial", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    start = parse_iso_timestamp(args.start).astimezone(timezone.utc)
    end = parse_iso_timestamp(args.end).astimezone(timezone.utc)
    if end <= start:
        raise ValueError("--end must be later than --start.")

    energy_range = (float(args.energy_range[0]), float(args.energy_range[1]))
    if not (0.0 < energy_range[0] < energy_range[1]):
        raise ValueError("--energy-range must satisfy 0 < LOW_EV < HIGH_EV.")

    output_dir = default_output_dir(Path(args.output_root).expanduser().resolve(), start, end)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_df, pitch_df = build_coadded_pad_tables(
        start=start,
        end=end,
        data_root=Path(args.data_root).expanduser().resolve(),
        energy_range_eV=energy_range,
        energy_method=args.energy_method,
        group_size=int(args.group_size),
        keep_partial=bool(args.keep_partial),
    )

    group_csv = output_dir / "coadded_pad_group_summary.csv"
    pitch_csv = output_dir / "coadded_pad_by_pitch.csv"
    group_df.to_csv(group_csv, index=False)
    pitch_df.to_csv(pitch_csv, index=False)

    summary = {
        "start": start.isoformat(timespec="seconds"),
        "end": end.isoformat(timespec="seconds"),
        "energy_range_eV": list(energy_range),
        "energy_method": args.energy_method,
        "group_size": int(args.group_size),
        "keep_partial": bool(args.keep_partial),
        "group_rows": int(len(group_df)),
        "pitch_rows": int(len(pitch_df)),
        "outputs": {
            "group_summary_csv": str(group_csv),
            "by_pitch_csv": str(pitch_csv),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
