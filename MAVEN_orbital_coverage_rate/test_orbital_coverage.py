from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from MAVEN_orbital_coverage_rate.calculate_maven_orbital_coverage import (
    GridCoverage,
    angular_edges,
    calculate_cartesian_coverage,
    calculate_spherical_coverage,
    iter_interval_days,
    mso_cartesian_to_spherical,
    write_cartesian_csv,
)


class OrbitalCoverageTests(unittest.TestCase):
    def test_interval_days_use_exclusive_end(self) -> None:
        start = datetime(2016, 1, 1, 23, 59, tzinfo=timezone.utc)
        end = datetime(2016, 1, 2, 0, 0, tzinfo=timezone.utc)
        self.assertEqual([day.isoformat() for day in iter_interval_days(start, end)], ["2016-01-01"])

    def test_cartesian_rightmost_edge_is_included(self) -> None:
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 0.0, 0.0],
            ]
        )
        result = calculate_cartesian_coverage(
            positions,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            z_range=(0.0, 1.0),
            cells=(1, 1, 1),
        )
        self.assertEqual(result.in_range_sample_count, 2)
        self.assertEqual(int(result.counts[0, 0, 0]), 2)
        self.assertEqual(result.covered_cell_count, 1)
        self.assertEqual(result.coverage_rate, 1.0)

    def test_mso_spherical_axes_and_longitude_wrapping(self) -> None:
        positions = np.asarray(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        spherical = mso_cartesian_to_spherical(positions, longitude_min_deg=0.0)
        np.testing.assert_allclose(spherical[:, 0], 1.0)
        np.testing.assert_allclose(spherical[:, 2], [0.0, 90.0, 270.0, 0.0])
        np.testing.assert_allclose(spherical[:, 1], [0.0, 0.0, 0.0, 90.0])

    def test_spherical_counts(self) -> None:
        positions = np.asarray(
            [
                [1.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [-1.5, 0.0, 0.0],
                [0.0, -1.5, 0.0],
            ]
        )
        result = calculate_spherical_coverage(
            positions,
            altitude_range_rm=(0.0, 1.0),
            altitude_cells=1,
            latitude_range_deg=(-90.0, 90.0),
            longitude_range_deg=(-180.0, 180.0),
            latitude_cells=1,
            longitude_cells=4,
            delta_degree=5.0,
        )
        self.assertEqual(result.in_range_sample_count, 4)
        self.assertEqual(result.covered_cell_count, 4)
        self.assertEqual(result.total_cell_count, 4)

    def test_delta_degree_allows_final_partial_cell(self) -> None:
        edges = angular_edges(-10.0, 10.0, cells=None, delta_degree=6.0, label="latitude")
        np.testing.assert_allclose(edges, [-10.0, -4.0, 2.0, 8.0, 10.0])

    def test_cartesian_csv_has_one_row_per_cell(self) -> None:
        coverage = GridCoverage(
            coordinate_system="cartesian_mso",
            axis_names=("x_rm", "y_rm", "z_rm"),
            edges=(
                np.asarray([0.0, 1.0, 2.0]),
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, 1.0]),
            ),
            counts=np.asarray([[[1]], [[0]]], dtype=np.int64),
            valid_sample_count=1,
            in_range_sample_count=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.csv"
            write_cartesian_csv(path, coverage)
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["covered"] for row in rows], ["1", "0"])


if __name__ == "__main__":
    unittest.main()

