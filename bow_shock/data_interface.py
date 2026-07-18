from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from download_maven_data import DEFAULT_DATA_ROOT, parse_iso_timestamp
from identify_magnetic_topology.magnetic_field_direction import (
    format_unix_time,
    load_magnetic_geometry_interval,
)
from process_maven_spectra import locate_nearest_index

from bow_shock.models import (
    DEFAULT_MODEL_NAME,
    MARS_RADIUS_KM,
    BowShockEvaluation,
    evaluate_position,
    get_model,
    sample_surface,
)


@dataclass(frozen=True)
class SpacecraftPosition:
    requested_time_utc: str
    sample_time_utc: str
    sample_time_unix: float
    time_delta_seconds: float
    position_mso_km: np.ndarray
    position_mso_rm: np.ndarray
    altitude_km: float
    sza_deg: float
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class BowShockContext:
    requested_time_utc: str
    sample_time_utc: str
    sample_time_unix: float
    time_delta_seconds: float
    model_name: str
    model_display_name: str
    model_type: str
    coordinate_system: str
    position_mso_km: np.ndarray
    position_mso_rm: np.ndarray
    altitude_km: float
    sza_deg: float
    inside_bow_shock: bool
    location: str
    model_value: float
    boundary_position_mso_km: np.ndarray | None
    boundary_position_mso_rm: np.ndarray | None
    spacecraft_radius_rm: float
    boundary_radius_rm: float
    radial_offset_rm: float
    radial_offset_km: float
    source_files: tuple[str, ...]

    def to_dict(self) -> dict:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return result


def _as_utc_datetime(value: str | datetime) -> datetime:
    parsed = parse_iso_timestamp(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _altitude_and_sza(position_mso_km: np.ndarray) -> tuple[float, float]:
    position = np.asarray(position_mso_km, dtype=float)
    radius = float(np.linalg.norm(position))
    if radius <= 0.0:
        return float("nan"), float("nan")
    altitude_km = radius - MARS_RADIUS_KM
    sza_deg = float(np.degrees(np.arccos(np.clip(position[0] / radius, -1.0, 1.0))))
    return altitude_km, sza_deg


def get_maven_position(
    time_utc: str | datetime,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    max_mag_delta_seconds: float = 5.0,
) -> SpacecraftPosition:
    target = _as_utc_datetime(time_utc)
    half_window = timedelta(seconds=max(10.0, float(max_mag_delta_seconds) + 2.0))
    geometry = load_magnetic_geometry_interval(
        Path(data_root).expanduser().resolve(),
        target - half_window,
        target + half_window,
    )
    if geometry is None:
        raise FileNotFoundError(
            f"No local MAG ss1s data were found near {target.isoformat(timespec='seconds')}."
        )

    times = np.asarray(geometry["times"], dtype=float)
    index = locate_nearest_index(times, target)
    sample_unix = float(times[index])
    delta_seconds = abs(sample_unix - target.timestamp())
    if delta_seconds > max_mag_delta_seconds:
        raise LookupError(
            f"Nearest MAG sample is {delta_seconds:.3f} s from the requested time; "
            f"limit is {max_mag_delta_seconds:.3f} s."
        )

    position_km = np.asarray(geometry["position_km"][index], dtype=float)
    altitude_km, sza_deg = _altitude_and_sza(position_km)
    return SpacecraftPosition(
        requested_time_utc=target.isoformat(timespec="seconds"),
        sample_time_utc=format_unix_time(sample_unix),
        sample_time_unix=sample_unix,
        time_delta_seconds=delta_seconds,
        position_mso_km=position_km,
        position_mso_rm=position_km / MARS_RADIUS_KM,
        altitude_km=altitude_km,
        sza_deg=sza_deg,
        source_files=tuple(str(path) for path in geometry.get("source_files", [])),
    )


def get_bow_shock_context(
    time_utc: str | datetime,
    model_name: str = DEFAULT_MODEL_NAME,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    spacecraft_position_mso_km: np.ndarray | None = None,
    max_mag_delta_seconds: float = 5.0,
    boundary_tolerance_km: float = 10.0,
) -> BowShockContext:
    target = _as_utc_datetime(time_utc)
    if spacecraft_position_mso_km is None:
        spacecraft = get_maven_position(
            target,
            data_root=data_root,
            max_mag_delta_seconds=max_mag_delta_seconds,
        )
    else:
        position_km = np.asarray(spacecraft_position_mso_km, dtype=float)
        altitude_km, sza_deg = _altitude_and_sza(position_km)
        spacecraft = SpacecraftPosition(
            requested_time_utc=target.isoformat(timespec="seconds"),
            sample_time_utc=target.isoformat(timespec="seconds"),
            sample_time_unix=target.timestamp(),
            time_delta_seconds=0.0,
            position_mso_km=position_km,
            position_mso_rm=position_km / MARS_RADIUS_KM,
            altitude_km=altitude_km,
            sza_deg=sza_deg,
            source_files=(),
        )

    model = get_model(model_name)
    evaluation: BowShockEvaluation = evaluate_position(
        spacecraft.position_mso_km,
        model=model,
        boundary_tolerance_km=boundary_tolerance_km,
    )
    return BowShockContext(
        requested_time_utc=spacecraft.requested_time_utc,
        sample_time_utc=spacecraft.sample_time_utc,
        sample_time_unix=spacecraft.sample_time_unix,
        time_delta_seconds=spacecraft.time_delta_seconds,
        model_name=model.name,
        model_display_name=model.display_name,
        model_type=model.model_type,
        coordinate_system=model.coordinate_system,
        position_mso_km=spacecraft.position_mso_km,
        position_mso_rm=spacecraft.position_mso_rm,
        altitude_km=spacecraft.altitude_km,
        sza_deg=spacecraft.sza_deg,
        inside_bow_shock=evaluation.inside_bow_shock,
        location=evaluation.location,
        model_value=evaluation.model_value,
        boundary_position_mso_km=evaluation.boundary_position_mso_km,
        boundary_position_mso_rm=evaluation.boundary_position_mso_rm,
        spacecraft_radius_rm=evaluation.spacecraft_radius_rm,
        boundary_radius_rm=evaluation.boundary_radius_rm,
        radial_offset_rm=evaluation.radial_offset_rm,
        radial_offset_km=evaluation.radial_offset_km,
        source_files=spacecraft.source_files,
    )


def get_bow_shock_surface(
    time_utc: str | datetime | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    x_min_rm: float = -3.0,
    x_max_rm: float | None = None,
    n_x: int = 180,
    n_azimuth: int = 96,
) -> dict:
    surface = sample_surface(
        model=model_name,
        x_min_rm=x_min_rm,
        x_max_rm=x_max_rm,
        n_x=n_x,
        n_azimuth=n_azimuth,
    )
    model = get_model(model_name)
    surface["time_utc"] = (
        _as_utc_datetime(time_utc).isoformat(timespec="seconds")
        if time_utc is not None
        else None
    )
    surface["model"] = model.metadata()
    return surface
