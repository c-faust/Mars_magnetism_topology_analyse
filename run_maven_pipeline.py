from __future__ import annotations
"""
Small batch runner for the single-time spectrum workflow.

Use this file when you want to process a list of explicit timestamps rather
than a continuous interval. It simply glues together:
- `download_maven_data.py`
- `process_maven_spectra.py`
"""

import json
from datetime import datetime
from pathlib import Path

from download_maven_data import DEFAULT_DATA_ROOT, download_products_for_timestamp, parse_iso_timestamp
from process_maven_spectra import process_target_time


CONFIG = {
    # Edit this list when you want to batch-process several specific times.
    "target_times": [
        "2024-11-07T02:15:00",
    ],
    "data_root": str(DEFAULT_DATA_ROOT),
    "output_root": str(Path("outputs") / "maven_spectra"),
}


def main() -> None:
    """Run the download + single-time processing chain for every target time."""
    data_root = Path(CONFIG["data_root"]).expanduser().resolve()
    output_root = Path(CONFIG["output_root"]).expanduser().resolve()
    all_results = []

    for raw_time in CONFIG["target_times"]:
        target_time = parse_iso_timestamp(raw_time)
        downloaded = download_products_for_timestamp(target_time=target_time, data_root=data_root)
        result = process_target_time(
            target_time=target_time,
            pad_file=downloaded["swe_svypad"],
            mag_file=downloaded["mag_sunstate-1sec"],
            output_root=output_root,
        )
        all_results.append(result.__dict__)

    summary_path = output_root / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Pipeline summary written to: {summary_path}")


if __name__ == "__main__":
    main()
