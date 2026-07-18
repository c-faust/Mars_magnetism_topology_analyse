from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

MARS_RADIUS_KM = 3389.5
DEFAULT_MODEL_NAME = "gruesbeck2018_mso"


@dataclass(frozen=True)
class BowShockEvaluation:
    model_name: str
    position_mso_rm: np.ndarray
    position_mso_km: np.ndarray
    model_value: float
    inside_bow_shock: bool
    location: str
    boundary_position_mso_rm: np.ndarray | None
    boundary_position_mso_km: np.ndarray | None
    spacecraft_radius_rm: float
    boundary_radius_rm: float
    radial_offset_rm: float
    radial_offset_km: float


class BowShockModel(Protocol):
    name: str
    display_name: str
    model_type: str
    coordinate_system: str
    source_doi: str
    source_url: str

    def implicit_value(self, position_mso_rm: np.ndarray) -> float: ...

    def boundary_on_ray(self, direction_mso: np.ndarray) -> np.ndarray | None: ...

    def rho_at_x_azimuth(self, x_rm: np.ndarray, azimuth_rad: float) -> np.ndarray: ...

    def nose_position_rm(self) -> np.ndarray: ...

    def nose_x_rm(self) -> float: ...

    def metadata(self) -> dict: ...


def _positive_real_roots(coefficients: list[float], tolerance: float = 1e-10) -> list[float]:
    values = np.asarray(coefficients, dtype=float)
    first_nonzero = np.flatnonzero(np.abs(values) > tolerance)
    if first_nonzero.size == 0:
        return []
    roots = np.roots(values[first_nonzero[0] :])
    result = [
        float(root.real)
        for root in roots
        if abs(float(root.imag)) <= tolerance and float(root.real) > tolerance
    ]
    return sorted(result)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(-1)
    if value.size != 3 or not np.all(np.isfinite(value)):
        raise ValueError("A finite three-component MSO vector is required.")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("The MSO vector magnitude must be positive.")
    return value / norm


@dataclass(frozen=True)
class AxisymmetricConicModel:
    name: str
    display_name: str
    focus_x_rm: float
    eccentricity: float
    semilatus_rectum_rm: float
    source_doi: str
    source_url: str
    notes: str
    model_type: str = "axisymmetric_conic"
    coordinate_system: str = "MSO"

    def implicit_value(self, position_mso_rm: np.ndarray) -> float:
        x, y, z = np.asarray(position_mso_rm, dtype=float)
        relative_x = x - self.focus_x_rm
        focus_distance = float(np.sqrt(relative_x**2 + y**2 + z**2))
        return focus_distance + self.eccentricity * relative_x - self.semilatus_rectum_rm

    def boundary_on_ray(self, direction_mso: np.ndarray) -> np.ndarray | None:
        direction = _unit_vector(direction_mso)
        mu = float(direction[0])
        focus = self.focus_x_rm
        epsilon = self.eccentricity
        shifted_latus = self.semilatus_rectum_rm + epsilon * focus
        coefficients = [
            1.0 - (epsilon * mu) ** 2,
            2.0 * mu * (epsilon * shifted_latus - focus),
            focus**2 - shifted_latus**2,
        ]
        for radius in _positive_real_roots(coefficients):
            point = radius * direction
            relative_x = float(point[0] - focus)
            if self.semilatus_rectum_rm - epsilon * relative_x < 0.0:
                continue
            if abs(self.implicit_value(point)) <= 1e-7:
                return point
        return None

    def rho_at_x_azimuth(self, x_rm: np.ndarray, azimuth_rad: float) -> np.ndarray:
        del azimuth_rad
        x = np.asarray(x_rm, dtype=float)
        relative_x = x - self.focus_x_rm
        radius_squared = (
            self.semilatus_rectum_rm - self.eccentricity * relative_x
        ) ** 2 - relative_x**2
        valid = (
            (radius_squared >= 0.0)
            & (self.semilatus_rectum_rm - self.eccentricity * relative_x >= 0.0)
        )
        return np.where(valid, np.sqrt(np.maximum(radius_squared, 0.0)), np.nan)

    def nose_x_rm(self) -> float:
        return self.focus_x_rm + self.semilatus_rectum_rm / (1.0 + self.eccentricity)

    def nose_position_rm(self) -> np.ndarray:
        return np.asarray([self.nose_x_rm(), 0.0, 0.0], dtype=float)

    def metadata(self) -> dict:
        result = asdict(self)
        result["time_dependence"] = "statistical_average"
        result["equation"] = "r = L / (1 + epsilon*cos(theta)); focus=(x0,0,0)"
        return result


@dataclass(frozen=True)
class QuadraticSurfaceModel:
    name: str
    display_name: str
    a_xx: float
    b_yy: float
    c_zz: float
    d_xy: float
    e_yz: float
    f_xz: float
    g_x: float
    h_y: float
    i_z: float
    source_doi: str
    source_url: str
    notes: str
    model_type: str = "three_dimensional_quadratic"
    coordinate_system: str = "MSO"

    def implicit_value(self, position_mso_rm: np.ndarray) -> float:
        x, y, z = np.asarray(position_mso_rm, dtype=float)
        return float(
            self.a_xx * x**2
            + self.b_yy * y**2
            + self.c_zz * z**2
            + self.d_xy * x * y
            + self.e_yz * y * z
            + self.f_xz * x * z
            + self.g_x * x
            + self.h_y * y
            + self.i_z * z
            - 1.0
        )

    def boundary_on_ray(self, direction_mso: np.ndarray) -> np.ndarray | None:
        x, y, z = _unit_vector(direction_mso)
        quadratic = (
            self.a_xx * x**2
            + self.b_yy * y**2
            + self.c_zz * z**2
            + self.d_xy * x * y
            + self.e_yz * y * z
            + self.f_xz * x * z
        )
        linear = self.g_x * x + self.h_y * y + self.i_z * z
        roots = _positive_real_roots([quadratic, linear, -1.0])
        if not roots:
            return None
        return roots[0] * np.asarray([x, y, z], dtype=float)

    def rho_at_x_azimuth(self, x_rm: np.ndarray, azimuth_rad: float) -> np.ndarray:
        x = np.asarray(x_rm, dtype=float)
        cos_phi = float(np.cos(azimuth_rad))
        sin_phi = float(np.sin(azimuth_rad))
        quadratic = (
            self.b_yy * cos_phi**2
            + self.c_zz * sin_phi**2
            + self.e_yz * cos_phi * sin_phi
        )
        linear = (
            (self.d_xy * x + self.h_y) * cos_phi
            + (self.f_xz * x + self.i_z) * sin_phi
        )
        constant = self.a_xx * x**2 + self.g_x * x - 1.0
        discriminant = linear**2 - 4.0 * quadratic * constant
        valid = discriminant >= 0.0
        sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0.0))
        root_a = (-linear + sqrt_discriminant) / (2.0 * quadratic)
        root_b = (-linear - sqrt_discriminant) / (2.0 * quadratic)
        positive_a = np.where(root_a >= 0.0, root_a, np.nan)
        positive_b = np.where(root_b >= 0.0, root_b, np.nan)
        radius = np.fmax(positive_a, positive_b)
        return np.where(valid, radius, np.nan)

    def nose_position_rm(self) -> np.ndarray:
        quadratic_matrix = np.asarray(
            [
                [self.a_xx, self.d_xy / 2.0, self.f_xz / 2.0],
                [self.d_xy / 2.0, self.b_yy, self.e_yz / 2.0],
                [self.f_xz / 2.0, self.e_yz / 2.0, self.c_zz],
            ],
            dtype=float,
        )
        linear = np.asarray([self.g_x, self.h_y, self.i_z], dtype=float)
        inverse = np.linalg.inv(quadratic_matrix)
        center = -0.5 * inverse @ linear
        completed_square = 1.0 + 0.25 * float(linear @ inverse @ linear)
        x_direction = np.asarray([1.0, 0.0, 0.0])
        support_vector = inverse @ x_direction
        scale = np.sqrt(completed_square / float(x_direction @ support_vector))
        return center + scale * support_vector

    def nose_x_rm(self) -> float:
        return float(self.nose_position_rm()[0])

    def metadata(self) -> dict:
        result = asdict(self)
        result["time_dependence"] = "statistical_average"
        result["equation"] = (
            "A*x^2+B*y^2+C*z^2+D*x*y+E*y*z+F*x*z+G*x+H*y+I*z=1"
        )
        return result


VIGNES_2000 = AxisymmetricConicModel(
    name="vignes2000",
    display_name="Vignes et al. (2000)",
    focus_x_rm=0.64,
    eccentricity=1.03,
    semilatus_rectum_rm=2.04,
    source_doi="10.1029/1999GL010703",
    source_url="https://doi.org/10.1029/1999GL010703",
    notes="Widely used average MGS bow-shock fit; no upstream-condition adjustment.",
)

TROTIGNON_2006 = AxisymmetricConicModel(
    name="trotignon2006",
    display_name="Trotignon et al. (2006)",
    focus_x_rm=0.60,
    eccentricity=1.026,
    semilatus_rectum_rm=2.081,
    source_doi="10.1016/j.pss.2006.01.003",
    source_url="https://doi.org/10.1016/j.pss.2006.01.003",
    notes="Axisymmetric MGS and Phobos-2 statistical bow-shock fit.",
)

GRUESBECK_2018_MSO = QuadraticSurfaceModel(
    name="gruesbeck2018_mso",
    display_name="Gruesbeck et al. (2018), all-points MSO",
    a_xx=0.049,
    b_yy=0.157,
    c_zz=0.153,
    d_xy=0.026,
    e_yz=0.012,
    f_xz=0.051,
    g_x=0.566,
    h_y=-0.031,
    i_z=0.019,
    source_doi="10.1029/2018JA025366",
    source_url="https://doi.org/10.1029/2018JA025366",
    notes="Fully 3-D MAVEN statistical surface, Table 1 all-points MSO coefficients.",
)

MODELS: dict[str, BowShockModel] = {
    model.name: model
    for model in (
        VIGNES_2000,
        TROTIGNON_2006,
        GRUESBECK_2018_MSO,
    )
}


def get_model(name: str = DEFAULT_MODEL_NAME) -> BowShockModel:
    normalized = str(name).strip().lower()
    try:
        return MODELS[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown bow-shock model {name!r}. Choose from: {choices}.") from exc


def list_models() -> list[dict]:
    return [MODELS[name].metadata() for name in sorted(MODELS)]


def evaluate_position(
    position_mso_km: np.ndarray,
    model: str | BowShockModel = DEFAULT_MODEL_NAME,
    boundary_tolerance_km: float = 10.0,
) -> BowShockEvaluation:
    selected_model = get_model(model) if isinstance(model, str) else model
    position_km = np.asarray(position_mso_km, dtype=float).reshape(-1)
    if position_km.size != 3 or not np.all(np.isfinite(position_km)):
        raise ValueError("position_mso_km must contain three finite values.")

    position_rm = position_km / MARS_RADIUS_KM
    spacecraft_radius_rm = float(np.linalg.norm(position_rm))
    if spacecraft_radius_rm <= 0.0:
        raise ValueError("position_mso_km must not be the zero vector.")

    model_value = float(selected_model.implicit_value(position_rm))
    boundary_rm = selected_model.boundary_on_ray(position_rm)
    if boundary_rm is None:
        boundary_radius_rm = float("nan")
        radial_offset_rm = float("nan")
        radial_offset_km = float("nan")
        boundary_km = None
    else:
        boundary_radius_rm = float(np.linalg.norm(boundary_rm))
        radial_offset_rm = spacecraft_radius_rm - boundary_radius_rm
        radial_offset_km = radial_offset_rm * MARS_RADIUS_KM
        boundary_km = boundary_rm * MARS_RADIUS_KM

    if np.isfinite(radial_offset_km) and abs(radial_offset_km) <= boundary_tolerance_km:
        location = "on_boundary"
    elif model_value < 0.0:
        location = "inside"
    else:
        location = "outside"

    return BowShockEvaluation(
        model_name=selected_model.name,
        position_mso_rm=position_rm,
        position_mso_km=position_km,
        model_value=model_value,
        inside_bow_shock=location in {"inside", "on_boundary"},
        location=location,
        boundary_position_mso_rm=boundary_rm,
        boundary_position_mso_km=boundary_km,
        spacecraft_radius_rm=spacecraft_radius_rm,
        boundary_radius_rm=boundary_radius_rm,
        radial_offset_rm=radial_offset_rm,
        radial_offset_km=radial_offset_km,
    )


def sample_surface(
    model: str | BowShockModel = DEFAULT_MODEL_NAME,
    x_min_rm: float = -3.0,
    x_max_rm: float | None = None,
    n_x: int = 180,
    n_azimuth: int = 96,
) -> dict[str, np.ndarray]:
    selected_model = get_model(model) if isinstance(model, str) else model
    if n_x < 2 or n_azimuth < 4:
        raise ValueError("n_x must be >= 2 and n_azimuth must be >= 4.")
    nose = selected_model.nose_x_rm()
    maximum = float(nose if x_max_rm is None else min(x_max_rm, nose))
    if x_min_rm >= maximum:
        raise ValueError("x_min_rm must be less than the model nose position.")

    x_axis = np.linspace(float(x_min_rm), maximum, int(n_x))
    azimuth = np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)
    x_grid = np.broadcast_to(x_axis[:, None], (x_axis.size, azimuth.size)).copy()
    radius_grid = np.column_stack(
        [selected_model.rho_at_x_azimuth(x_axis, phi) for phi in azimuth]
    )
    y_grid = radius_grid * np.cos(azimuth)[None, :]
    z_grid = radius_grid * np.sin(azimuth)[None, :]
    return {
        "x_rm": x_grid,
        "y_rm": y_grid,
        "z_rm": z_grid,
        "rho_rm": radius_grid,
        "x_km": x_grid * MARS_RADIUS_KM,
        "y_km": y_grid * MARS_RADIUS_KM,
        "z_km": z_grid * MARS_RADIUS_KM,
        "rho_km": radius_grid * MARS_RADIUS_KM,
        "azimuth_rad": azimuth,
        "model_name": np.asarray(selected_model.name),
    }
