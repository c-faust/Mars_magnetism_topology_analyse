from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bow_shock.models import get_model
from plot_maven_orbit_map import (
    boundary_plane_curves,
    plot_mso_orbit_projections,
    plot_orbit_map,
)


class MsoOrbitProjectionTests(unittest.TestCase):
    def test_boundary_plane_curves_support_xy_and_xz(self) -> None:
        model = get_model("gruesbeck2018_mso")
        for plane in ("xy", "xz"):
            curves = boundary_plane_curves(
                model,
                plane,
                x_min_rm=-3.0,
                x_max_rm=2.0,
                sample_count=80,
            )
            self.assertEqual(len(curves), 2)
            for x_values, transverse in curves:
                self.assertEqual(x_values.shape, (80,))
                self.assertEqual(transverse.shape, (80,))
                self.assertTrue(np.any(np.isfinite(transverse)))

    @patch("plot_maven_orbit_map.load_mag_day")
    def test_mso_projection_plot_writes_three_plane_files(self, load_mag_day) -> None:
        start = datetime(2017, 3, 22, 12, 0, tzinfo=timezone.utc)
        times = start.timestamp() + np.arange(5, dtype=float)
        positions_rm = np.asarray(
            [
                [1.8, -0.8, -0.4],
                [1.5, -0.4, -0.2],
                [1.2, 0.0, 0.0],
                [0.9, 0.4, 0.2],
                [0.6, 0.8, 0.4],
            ],
            dtype=float,
        )
        load_mag_day.return_value = {
            "times": times,
            "data": positions_rm * 3389.5,
            "pos_indices": [0, 1, 2],
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mso_projections.png"
            result = plot_mso_orbit_projections(
                start_time=start,
                end_time=datetime.fromtimestamp(times[-1], tz=timezone.utc),
                mag_ss_files=[Path("synthetic_ss1s.sts")],
                output_path=output,
                boundary_sample_count=100,
                trajectory_markers=[
                    (
                        datetime.fromtimestamp(times[2] + 0.2, tz=timezone.utc),
                        "--",
                        "red",
                    )
                ],
            )
            self.assertFalse(output.exists())
            self.assertEqual(set(result["output_paths"]), {"xy", "xz", "yz"})
            for plane, path_value in result["output_paths"].items():
                plane_output = Path(path_value)
                self.assertEqual(plane_output.name, f"mso_projections_{plane}.png")
                self.assertTrue(plane_output.exists())
                self.assertGreater(plane_output.stat().st_size, 0)
            self.assertEqual(result["coordinate_system"], "MSO")
            self.assertEqual(result["track_samples"], 5)
            self.assertEqual(result["mars_radius_km"], 3389.5)
            self.assertEqual(result["bow_model"]["name"], "gruesbeck2018_mso")
            self.assertEqual(result["mpb_model"]["name"], "vignes2000_mpb")
            self.assertEqual(len(result["trajectory_markers"]), 1)
            self.assertEqual(result["trajectory_markers"][0]["marker"], "s")
            self.assertAlmostEqual(
                result["trajectory_markers"][0]["delta_seconds"],
                -0.2,
                places=5,
            )

    @patch("plot_maven_orbit_map.sun_direction_pc")
    @patch("plot_maven_orbit_map.load_or_build_crustal_field_grid")
    @patch("plot_maven_orbit_map.load_mag_day")
    def test_ground_track_accepts_multiple_files_and_marks_line_time(
        self,
        load_mag_day,
        load_crustal_grid,
        sun_direction,
    ) -> None:
        start = datetime(2017, 3, 22, 12, 0, tzinfo=timezone.utc)
        times = start.timestamp() + np.arange(5, dtype=float)
        load_mag_day.return_value = {
            "times": times,
            "data": np.asarray(
                [
                    [3600.0, 100.0, -80.0],
                    [3550.0, 150.0, -40.0],
                    [3500.0, 200.0, 0.0],
                    [3450.0, 250.0, 40.0],
                    [3400.0, 300.0, 80.0],
                ]
            ),
            "pos_indices": [0, 1, 2],
        }
        lon_values = np.asarray([0.0, 90.0, 180.0])
        lat_values = np.asarray([-90.0, 0.0, 90.0])
        load_crustal_grid.return_value = (
            lon_values,
            lat_values,
            np.full((3, 3), 10.0),
            Path("synthetic_cache.npz"),
            True,
        )
        sun_direction.return_value = np.asarray([1.0, 0.0, 0.0])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ground.png"
            result = plot_orbit_map(
                target_time=datetime.fromtimestamp(times[2], tz=timezone.utc),
                start_time=start,
                end_time=datetime.fromtimestamp(times[-1], tz=timezone.utc),
                mag_pc_file=[Path("first_pc1s.sts"), Path("second_pc1s.sts")],
                model_root=Path(directory),
                output_path=output,
                trajectory_markers=[
                    (
                        datetime.fromtimestamp(times[3] + 0.1, tz=timezone.utc),
                        ":",
                        "blue",
                    )
                ],
            )
            self.assertTrue(output.exists())
            self.assertEqual(result["track_samples"], 5)
            self.assertEqual(len(result["source_files"]), 2)
            self.assertEqual(result["trajectory_markers"][0]["marker"], "^")


if __name__ == "__main__":
    unittest.main()
