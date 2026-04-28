from __future__ import annotations
"""
Morschhauser crustal magnetic field model utilities.

This module is intentionally separate from the main analysis script because it
contains the model-specific logic:
- download/load spherical harmonic coefficients
- evaluate the crustal field in Mars body-fixed coordinates
- rotate the result into MSO for comparison with MAG observations
"""

import gzip
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from scipy.special import gammaln, lpmv


MARS_REFERENCE_RADIUS_KM = 3393.5
MARS_MODEL_URL = "https://zenodo.org/records/3876495/files/Morschhauser2014.txt.gz?download=1"
DEFAULT_MODEL_ROOT = Path("data") / "models" / "mars_crustal"
MARS_MODEL_FILENAME = "Morschhauser2014.txt.gz"

# These constants match the NAIF BODY499 values commonly used for Mars body-fixed transforms.
# BODY499_POLE_RA  = (317.68143 -0.1061 0.0) deg
# BODY499_POLE_DEC = ( 52.88650 -0.0609 0.0) deg
# BODY499_PM       = (176.630   350.89198226 0.0) deg
MARS_POLE_RA_DEG = (317.68143, -0.1061)
MARS_POLE_DEC_DEG = (52.88650, -0.0609)
MARS_PM_DEG = (176.630, 350.89198226)


@dataclass(frozen=True)
class SphericalHarmonicCoefficients:
    """Container for one fully parsed spherical harmonic coefficient table."""
    degree: np.ndarray
    order: np.ndarray
    g: np.ndarray
    h: np.ndarray
    max_degree: int


def ensure_morschhauser_coefficients(model_root: Path = DEFAULT_MODEL_ROOT) -> Path:
    """Download the model coefficient file the first time it is needed."""
    model_root.mkdir(parents=True, exist_ok=True)
    target = model_root / MARS_MODEL_FILENAME
    if target.exists():
        return target

    response = requests.get(MARS_MODEL_URL, stream=True, timeout=(20, 180))
    response.raise_for_status()
    with target.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return target


@lru_cache(maxsize=2)
def load_morschhauser_coefficients(path_str: str) -> SphericalHarmonicCoefficients:
    """Parse the Morschhauser coefficient text file into vectorized arrays.

    In this file format:
    - `m >= 0` stores `g(n,m)`
    - `m < 0` stores `h(n,|m|)`
    """
    path = Path(path_str)
    opener = gzip.open if path.suffix == ".gz" else open
    coefficient_map: dict[tuple[int, int], dict[str, float]] = {}
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            values: list[float] = []
            for token in parts:
                try:
                    values.append(float(token))
                except ValueError:
                    continue
            if len(values) < 3:
                continue
            n = int(values[0])
            m = int(values[1])
            coeff = float(values[2])
            if n < 1 or abs(m) > n:
                continue
            key = (n, abs(m))
            if key not in coefficient_map:
                coefficient_map[key] = {"g": 0.0, "h": 0.0}
            if m >= 0:
                coefficient_map[key]["g"] = coeff
            else:
                coefficient_map[key]["h"] = coeff

    if not coefficient_map:
        raise ValueError(f"No valid spherical harmonic coefficients were parsed from {path}.")

    ordered_rows = sorted((n, m, values["g"], values["h"]) for (n, m), values in coefficient_map.items())
    degree = np.asarray([item[0] for item in ordered_rows], dtype=int)
    order = np.asarray([item[1] for item in ordered_rows], dtype=int)
    g = np.asarray([item[2] for item in ordered_rows], dtype=float)
    h = np.asarray([item[3] for item in ordered_rows], dtype=float)
    return SphericalHarmonicCoefficients(
        degree=degree,
        order=order,
        g=g,
        h=h,
        max_degree=int(np.max(degree)),
    )


def julian_centuries_tdb(unix_seconds: float) -> tuple[float, float]:
    """Return `(days, centuries)` since J2000 in TDB-like form."""
    time = Time(unix_seconds, format="unix", scale="utc")
    jd_tdb = float(time.tdb.jd)
    d = jd_tdb - 2451545.0
    t = d / 36525.0
    return d, t


def mars_body_to_icrf_matrix(unix_seconds: float) -> np.ndarray:
    """Build the Mars body-fixed -> ICRF rotation matrix for one epoch."""
    d, t = julian_centuries_tdb(unix_seconds)
    alpha = np.deg2rad(MARS_POLE_RA_DEG[0] + MARS_POLE_RA_DEG[1] * t)
    delta = np.deg2rad(MARS_POLE_DEC_DEG[0] + MARS_POLE_DEC_DEG[1] * t)
    w = np.deg2rad((MARS_PM_DEG[0] + MARS_PM_DEG[1] * d) % 360.0)

    z_axis = np.array(
        [np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)],
        dtype=float,
    )
    q_axis = np.array([-np.sin(alpha), np.cos(alpha), 0.0], dtype=float)
    q_axis /= np.linalg.norm(q_axis)
    p_axis = np.cross(z_axis, q_axis)
    p_axis /= np.linalg.norm(p_axis)

    x_axis = q_axis * np.cos(w) + p_axis * np.sin(w)
    y_axis = -q_axis * np.sin(w) + p_axis * np.cos(w)
    return np.column_stack([x_axis, y_axis, z_axis])


def icrf_to_mso_matrix(unix_seconds: float) -> np.ndarray:
    """Build the ICRF -> MSO rotation matrix using Sun and Mars ephemerides."""
    time = Time(unix_seconds, format="unix", scale="utc")
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", time)
    mars_pos, mars_vel = get_body_barycentric_posvel("mars", time)

    r_ms = (sun_pos.xyz - mars_pos.xyz).to_value("km")
    v_mars = (mars_vel.xyz - sun_vel.xyz).to_value("km/s")

    x_axis = r_ms / np.linalg.norm(r_ms)
    z_axis = np.cross(x_axis, v_mars)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.vstack([x_axis, y_axis, z_axis])


def schmidt_semi_normalized_pnm(n: int, m: int, x: float) -> float:
    """Evaluate Schmidt semi-normalized associated Legendre functions.

    This is one of the core mathematical pieces inside the spherical harmonic
    field model.
    """
    x = float(np.clip(x, -1.0, 1.0))
    base = lpmv(m, n, x)
    log_ratio = gammaln(n - m + 1) - gammaln(n + m + 1)
    schmidt = np.sqrt((2.0 - (1.0 if m == 0 else 0.0)) * np.exp(log_ratio))
    with np.errstate(invalid="ignore", over="ignore"):
        value = ((-1) ** m) * schmidt * base
    if not np.isfinite(value):
        return 0.0
    return float(value)


def dtheta_schmidt_pnm(n: int, m: int, theta: float) -> float:
    x = np.cos(theta)
    sin_theta = max(np.sin(theta), 1e-10)
    p_nm = schmidt_semi_normalized_pnm(n, m, x)
    if n == m:
        p_n1m = 0.0
    else:
        p_n1m = schmidt_semi_normalized_pnm(n - 1, m, x)
    value = (n * x * p_nm - (n + m) * p_n1m) / sin_theta
    if not np.isfinite(value):
        return 0.0
    return float(value)


def evaluate_morschhauser_field_pc(position_pc_km: np.ndarray, coeffs: SphericalHarmonicCoefficients) -> np.ndarray:
    """Evaluate the crustal magnetic field at one point in PC coordinates."""
    x, y, z = np.asarray(position_pc_km, dtype=float)
    r = float(np.linalg.norm(position_pc_km))
    theta = float(np.arccos(np.clip(z / max(r, 1e-9), -1.0, 1.0)))
    phi = float(np.arctan2(y, x))
    sin_theta = max(np.sin(theta), 1e-10)

    br = 0.0
    btheta = 0.0
    bphi = 0.0
    a = MARS_REFERENCE_RADIUS_KM
    radial_factor_base = a / r

    for n, m, g, h in zip(coeffs.degree, coeffs.order, coeffs.g, coeffs.h):
        p_nm = schmidt_semi_normalized_pnm(int(n), int(m), np.cos(theta))
        dp_dtheta = dtheta_schmidt_pnm(int(n), int(m), theta)
        if not np.isfinite(p_nm) or not np.isfinite(dp_dtheta):
            continue
        cos_mphi = np.cos(m * phi)
        sin_mphi = np.sin(m * phi)
        common = g * cos_mphi + h * sin_mphi
        radial_factor = radial_factor_base ** (n + 2)
        if not np.isfinite(radial_factor):
            continue
        br += (n + 1) * radial_factor * common * p_nm
        btheta -= radial_factor * common * dp_dtheta
        if m > 0:
            bphi += radial_factor * m * (-g * sin_mphi + h * cos_mphi) * p_nm / sin_theta

    e_r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], dtype=float)
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)], dtype=float)
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=float)
    return br * e_r + btheta * e_theta + bphi * e_phi


def evaluate_morschhauser_field_mso(
    position_pc_km: np.ndarray,
    unix_seconds: float,
    coeffs: SphericalHarmonicCoefficients,
) -> np.ndarray:
    """Evaluate the model in PC and rotate the field vector into MSO."""
    field_pc = evaluate_morschhauser_field_pc(position_pc_km, coeffs)
    body_to_icrf = mars_body_to_icrf_matrix(unix_seconds)
    icrf_to_mso = icrf_to_mso_matrix(unix_seconds)
    field_icrf = body_to_icrf @ field_pc
    return icrf_to_mso @ field_icrf
