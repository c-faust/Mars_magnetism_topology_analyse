from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

from download_maven_data import (
    ProductSpec,
    build_session,
    fetch_filenames,
    choose_best_filename,
)


LPW_MRG_SCPOT = ProductSpec(
    instrument="lpw",
    datatype="mrgscpot",
    aliases=("mrgscpot", "scpot"),
    level="l2",
    format_preference=("cdf",),
)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def daterange(start_day: date, end_day: date):
    current = start_day
    while current <= end_day:
        yield current
        current += timedelta(days=1)


def check_one_day(session, day: date) -> dict:
    try:
        filenames = fetch_filenames(
            session=session,
            instrument="lpw",
            start_date=day,
            end_date=day,
            level="l2",
        )

        filename, parsed = choose_best_filename(
            filenames=filenames,
            spec=LPW_MRG_SCPOT,
            day=day,
        )

        if filename and parsed:
            return {
                "date": day.isoformat(),
                "exists": "yes",
                "filename": filename,
                "version": parsed["version"],
                "revision": parsed["revision"],
                "n_lpw_l2_files": len(filenames),
                "status": "ok",
            }

        return {
            "date": day.isoformat(),
            "exists": "no",
            "filename": "",
            "version": "",
            "revision": "",
            "n_lpw_l2_files": len(filenames),
            "status": "no_mrgscpot",
        }

    except Exception as exc:
        return {
            "date": day.isoformat(),
            "exists": "error",
            "filename": "",
            "version": "",
            "revision": "",
            "n_lpw_l2_files": "",
            "status": str(exc),
        }


def main() -> None:
    start_day = parse_date("2014-10-01")
    end_day = parse_date("2024-12-31")

    output_csv = Path("lpw_mrgscpot_date_availability.csv")

    session = build_session()

    rows = []

    for day in daterange(start_day, end_day):
        print(f"[check] {day}")
        row = check_one_day(session, day)
        rows.append(row)

    fieldnames = [
        "date",
        "exists",
        "filename",
        "version",
        "revision",
        "n_lpw_l2_files",
        "status",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_yes = sum(1 for r in rows if r["exists"] == "yes")
    n_no = sum(1 for r in rows if r["exists"] == "no")
    n_error = sum(1 for r in rows if r["exists"] == "error")

    print()
    print(f"Saved table to: {output_csv.resolve()}")
    print(f"Available days: {n_yes}")
    print(f"Missing days: {n_no}")
    print(f"Error days: {n_error}")


if __name__ == "__main__":
    main()