from __future__ import annotations

import unittest

import numpy as np

from bow_shock.data_interface import get_bow_shock_context
from bow_shock.models import (
    DEFAULT_MODEL_NAME,
    GRUESBECK_2018_MSO,
    MARS_RADIUS_KM,
    TROTIGNON_2006,
    VIGNES_2000,
    evaluate_position,
    get_model,
    list_models,
    sample_surface,
)


class BowShockModelTests(unittest.TestCase):
    def test_model_registry(self) -> None:
        self.assertEqual(
            {item["name"] for item in list_models()},
            {"vignes2000", "trotignon2006", "gruesbeck2018_mso"},
        )
        self.assertEqual(DEFAULT_MODEL_NAME, "gruesbeck2018_mso")
        self.assertIs(get_model(), GRUESBECK_2018_MSO)
        self.assertIs(get_model("vignes2000"), VIGNES_2000)

    def test_vignes_nose_is_on_surface(self) -> None:
        point = VIGNES_2000.nose_position_rm()
        self.assertAlmostEqual(VIGNES_2000.implicit_value(point), 0.0, places=10)
        self.assertAlmostEqual(VIGNES_2000.nose_x_rm(), 1.6449261084, places=8)

    def test_trotignon_nose_is_on_surface(self) -> None:
        point = TROTIGNON_2006.nose_position_rm()
        self.assertAlmostEqual(TROTIGNON_2006.implicit_value(point), 0.0, places=10)

    def test_gruesbeck_nose_is_on_surface(self) -> None:
        point = GRUESBECK_2018_MSO.nose_position_rm()
        self.assertAlmostEqual(GRUESBECK_2018_MSO.implicit_value(point), 0.0, places=10)
        self.assertAlmostEqual(point[0], 1.57951702895, places=8)

    def test_vignes_inside_and_outside(self) -> None:
        inside = evaluate_position(
            np.asarray([1.0, 0.0, 0.0]) * MARS_RADIUS_KM,
            model="vignes2000",
        )
        outside = evaluate_position(
            np.asarray([2.0, 0.0, 0.0]) * MARS_RADIUS_KM,
            model="vignes2000",
        )
        self.assertTrue(inside.inside_bow_shock)
        self.assertEqual(inside.location, "inside")
        self.assertFalse(outside.inside_bow_shock)
        self.assertEqual(outside.location, "outside")

    def test_surface_sampling_shapes(self) -> None:
        for model_name in ("vignes2000", "trotignon2006", "gruesbeck2018_mso"):
            surface = sample_surface(model_name, n_x=20, n_azimuth=12)
            self.assertEqual(surface["x_rm"].shape, (20, 12))
            self.assertEqual(surface["y_rm"].shape, (20, 12))
            self.assertGreater(np.count_nonzero(np.isfinite(surface["z_rm"])), 200)

    def test_context_with_supplied_position_does_not_require_mag(self) -> None:
        context = get_bow_shock_context(
            "2024-11-07T02:15:00Z",
            spacecraft_position_mso_km=np.asarray([2.0, 0.0, 0.0]) * MARS_RADIUS_KM,
        )
        self.assertEqual(context.model_name, "gruesbeck2018_mso")
        self.assertEqual(context.sample_time_utc, "2024-11-07T02:15:00+00:00")
        self.assertEqual(context.location, "outside")
        self.assertAlmostEqual(context.sza_deg, 0.0)


if __name__ == "__main__":
    unittest.main()
