from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from process_maven_spectra import build_mag_times, infer_daily_file, locate_nearest_index, parse_mag_sts


@dataclass(frozen=True)
class MagneticFieldDirection:
    time_unix: float
    time_utc: str
    magnetic_field_nT: np.ndarray
    position_km: np.ndarray
    dot_b_r: float
    angle_deg: float
    field_direction: str


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


def load_mag_sunstate_day(path: Path) -> dict:
    parsed = parse_mag_sts(path)
    columns = parsed["columns"]
    data = np.asarray(parsed["data"], dtype=float)
    return {
        "path": path,
        "times": build_mag_times(columns, data),
        "data": data,
        "columns": columns,
        "b_indices": [columns.index("OB_B.X"), columns.index("OB_B.Y"), columns.index("OB_B.Z")],
        "pos_indices": [columns.index("POSN.X"), columns.index("POSN.Y"), columns.index("POSN.Z")],
    }


def load_magnetic_geometry_interval(
    data_root: Path | tuple[Path, ...] | list[Path],
    start: datetime,
    end: datetime,
) -> dict | None:
    times: list[np.ndarray] = []
    fields: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    source_files: list[str] = []

    for day in iter_utc_days(start, end):
        try:
            mag_file = infer_daily_file(data_root, "mag", "ss1s", day, "sts")
            mag_day = load_mag_sunstate_day(mag_file)
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            print(f"[mag-field-direction] skip MAG SS day {day.date()}: {exc}", flush=True)
            continue

        day_times = np.asarray(mag_day["times"], dtype=float)
        day_data = np.asarray(mag_day["data"], dtype=float)
        window = (day_times >= start.timestamp()) & (day_times <= end.timestamp())
        if not np.any(window):
            continue

        times.append(day_times[window])
        fields.append(day_data[window][:, mag_day["b_indices"]])
        positions.append(day_data[window][:, mag_day["pos_indices"]])
        source_files.append(str(mag_file))

    if not times:
        return None

    merged_times = np.concatenate(times)
    order = np.argsort(merged_times)
    return {
        "times": merged_times[order],
        "magnetic_field_nT": np.vstack(fields)[order],
        "position_km": np.vstack(positions)[order],
        "source_files": source_files,
    }


def classify_field_direction(magnetic_field_nT: np.ndarray, position_km: np.ndarray) -> tuple[str, float, float]:
    field = np.asarray(magnetic_field_nT, dtype=float)
    position = np.asarray(position_km, dtype=float)
    if field.size != 3 or position.size != 3:
        return "invalid", float("nan"), float("nan")
    if not np.all(np.isfinite(field)) or not np.all(np.isfinite(position)):
        return "invalid", float("nan"), float("nan")

    field_norm = float(np.linalg.norm(field))
    position_norm = float(np.linalg.norm(position))
    if field_norm <= 0.0 or position_norm <= 0.0:
        return "invalid", float("nan"), float("nan")

    dot_b_r = float(np.dot(field, position))
    cos_angle = float(np.clip(dot_b_r / (field_norm * position_norm), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    if angle_deg > 90.0:
        field_direction = "toward_surface"
    elif angle_deg < 90.0:
        field_direction = "away_from_surface"
    else:
        field_direction = "perpendicular"
    return field_direction, dot_b_r, angle_deg


def nearest_magnetic_field_direction(
    magnetic_geometry: dict,
    target_time: datetime,
    max_delta_seconds: float,
) -> MagneticFieldDirection | None:
    times = np.asarray(magnetic_geometry.get("times", []), dtype=float)
    if times.size == 0:
        return None

    index = locate_nearest_index(times, target_time)
    delta = abs(float(times[index]) - target_time.timestamp())
    if delta > max_delta_seconds:
        return None

    field = np.asarray(magnetic_geometry["magnetic_field_nT"][index], dtype=float)
    position = np.asarray(magnetic_geometry["position_km"][index], dtype=float)
    field_direction, dot_b_r, angle_deg = classify_field_direction(field, position)
    if field_direction == "invalid":
        return None

    return MagneticFieldDirection(
        time_unix=float(times[index]),
        time_utc=format_unix_time(times[index]),
        magnetic_field_nT=field,
        position_km=position,
        dot_b_r=dot_b_r,
        angle_deg=angle_deg,
        field_direction=field_direction,
    )


def map_parallel_antiparallel_to_toward_away(
    parallel_value: float,
    antiparallel_value: float,
    field_direction: str,
) -> tuple[float, float, str, str]:
    if field_direction == "toward_surface":
        return parallel_value, antiparallel_value, "parallel", "antiparallel"
    if field_direction == "away_from_surface":
        return antiparallel_value, parallel_value, "antiparallel", "parallel"
    return float("nan"), float("nan"), "undefined", "undefined"


def write_direction_csv(path: Path, magnetic_geometry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(magnetic_geometry.get("times", []), dtype=float)
    fields = np.asarray(magnetic_geometry.get("magnetic_field_nT", []), dtype=float)
    positions = np.asarray(magnetic_geometry.get("position_km", []), dtype=float)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time_unix",
                "time_utc",
                "bx_nT",
                "by_nT",
                "bz_nT",
                "x_mso_km",
                "y_mso_km",
                "z_mso_km",
                "dot_b_r",
                "field_angle_deg",
                "field_direction",
            ],
        )
        writer.writeheader()
        for unix_time, field, position in zip(times, fields, positions):
            field_direction, dot_b_r, angle_deg = classify_field_direction(field, position)
            writer.writerow(
                {
                    "time_unix": float(unix_time),
                    "time_utc": format_unix_time(unix_time),
                    "bx_nT": float(field[0]),
                    "by_nT": float(field[1]),
                    "bz_nT": float(field[2]),
                    "x_mso_km": float(position[0]),
                    "y_mso_km": float(position[1]),
                    "z_mso_km": float(position[2]),
                    "dot_b_r": dot_b_r,
                    "field_angle_deg": angle_deg,
                    "field_direction": field_direction,
                }
            )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify whether MAVEN MAG B points toward or away from the Mars surface.")
    parser.add_argument("--start", required=True, help="UTC interval start, for example 2024-11-07T02:00:00.")
    parser.add_argument("--end", required=True, help="UTC interval end, for example 2024-11-07T02:30:00.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for MAVEN data.")
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Defaults to outputs/identify_magnetic_topology/magnetic_field_direction/<start>_<end>.csv.",
    )
    return parser


def default_output_path(start: datetime, end: datetime) -> Path:
    name = f"{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}.csv"
    return Path("outputs") / "identify_magnetic_topology" / "magnetic_field_direction" / name


def main() -> None:
    args = build_argument_parser().parse_args()
    start = parse_iso_timestamp(args.start).astimezone(timezone.utc)
    end = parse_iso_timestamp(args.end).astimezone(timezone.utc)
    if end <= start:
        raise ValueError("--end must be later than --start.")

    data_root = Path(args.data_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(start, end).resolve()
    magnetic_geometry = load_magnetic_geometry_interval(data_root, start, end)
    if magnetic_geometry is None:
        raise FileNotFoundError("No usable MAG sunstate-1sec samples were found in the requested interval.")

    write_direction_csv(output_path, magnetic_geometry)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
