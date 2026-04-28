from __future__ import annotations
"""
Download controller for MAVEN SWE electron spectra.

This script intentionally delegates network and file-selection details to the
project-level `download_maven_data.py` module. The local job here is only to
describe which dates should be downloaded for machine-learning work.
"""

import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from download_maven_data import PIPELINE_PRODUCTS, build_session, download_product_for_day, fetch_filenames
from process_maven_spectra import format_unix_time, load_pad_data


ML_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ML_ROOT / "data" / "maven"
DEFAULT_OUTPUT_ROOT = ML_ROOT / "outputs" / "downloads"
SWE_PRODUCTS = tuple(spec for spec in PIPELINE_PRODUCTS if spec.instrument == "swe" and spec.datatype == "svypad")


def log_step(message: str) -> None:
    print(f"[download] {datetime.now().isoformat(timespec='seconds')} | {message}", flush=True)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_days(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def inspect_time_range(path: Path) -> dict:
    try:
        pad_data = load_pad_data(path)
        times = pad_data["times"]
        if len(times) == 0:
            return {"time_start_utc": None, "time_end_utc": None, "time_sample_count": 0, "read_status": "empty"}
        return {
            "time_start_utc": format_unix_time(float(times[0])),
            "time_end_utc": format_unix_time(float(times[-1])),
            "time_sample_count": int(len(times)),
            "read_status": "ok",
        }
    except Exception as exc:
        return {
            "time_start_utc": None,
            "time_end_utc": None,
            "time_sample_count": 0,
            "read_status": f"error: {type(exc).__name__}: {exc}",
        }


def write_csv_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "date",
        "instrument",
        "datatype",
        "time_start_utc",
        "time_end_utc",
        "time_sample_count",
        "read_status",
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download MAVEN SWE svypad electron spectra for ML analysis.")
    parser.add_argument(
        "--check-connection",
        action="store_true",
        help="Query the LASP file-list API for the requested interval without downloading files.",
    )
    parser.add_argument("--year", type=int, help="Download one full UTC year, for example 2024.")
    parser.add_argument("--start-date", help="UTC start date, for example 2024-11-07.")
    parser.add_argument("--end-date", help="UTC end date, for example 2024-11-08.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for MAVEN data.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where the download manifest will be written.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.year is not None:
        start_date = date(args.year, 1, 1)
        end_date = date(args.year, 12, 31)
    elif args.start_date and args.end_date:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
    else:
        raise ValueError("Use either --year or both --start-date and --end-date.")

    if end_date < start_date:
        raise ValueError("--end-date must be on or after --start-date.")

    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    total_days = (end_date - start_date).days + 1
    log_step(f"Date range ready: {start_date} to {end_date} ({total_days} day(s)).")
    log_step(f"Data root: {data_root}")
    log_step(f"Output root: {output_root}")

    log_step("Building HTTP session.")
    session = build_session()

    if args.check_connection:
        log_step("Checking LASP file-list connectivity; no files will be downloaded.")
        for spec in SWE_PRODUCTS:
            filenames = fetch_filenames(
                session=session,
                instrument=spec.instrument,
                start_date=start_date,
                end_date=end_date,
                level=spec.level,
            )
            matching = [name for name in filenames if any(alias in name.lower() for alias in spec.aliases)]
            print(
                f"Connection OK: LASP returned {len(filenames)} {spec.instrument} filenames, "
                f"{len(matching)} matching {spec.datatype}, for {start_date} to {end_date}."
            )
            for name in matching[:5]:
                print(f"  {name}")
        log_step("Connection check finished.")
        return

    manifest = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "data_root": str(data_root),
        "products": [],
    }

    for day_index, day in enumerate(iter_days(start_date, end_date), start=1):
        log_step(f"Processing day {day_index}/{total_days}: {day.isoformat()}")
        for spec in SWE_PRODUCTS:
            log_step(f"Resolving and downloading {spec.instrument}/{spec.datatype} for {day.isoformat()}.")
            local_path = download_product_for_day(session=session, spec=spec, day=day, data_root=data_root)
            log_step(f"Inspecting time coverage in {local_path.name}.")
            manifest["products"].append(
                {
                    "date": day.isoformat(),
                    "instrument": spec.instrument,
                    "datatype": spec.datatype,
                    "path": str(local_path),
                    **inspect_time_range(local_path),
                }
            )
            print(f"{day.isoformat()} {spec.instrument}/{spec.datatype}: {local_path}")

    manifest_path = output_root / "download_manifest.json"
    csv_manifest_path = output_root / "download_manifest.csv"
    log_step("Writing download manifests.")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv_manifest(csv_manifest_path, manifest["products"])
    print(f"Download manifest written to: {manifest_path}")
    print(f"CSV manifest written to: {csv_manifest_path}")
    log_step("Download workflow finished.")


if __name__ == "__main__":
    main()
