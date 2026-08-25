from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from plot_maven_data_panels import (
    build_argument_parser,
    parse_vertical_line_specs,
    plot_data_panels,
    plot_region_id_panel,
    plot_topology_id_panel,
    validate_panel_ids,
)

import matplotlib.pyplot as plt


TARGET = datetime(2016, 10, 6, 18, 10, tzinfo=timezone.utc)
TIMES = [TARGET.timestamp() - 10.0, TARGET.timestamp(), TARGET.timestamp() + 10.0]


def synthetic_summary() -> dict:
    return {
        "samples": [{"target_time": TARGET.isoformat()}],
        "context_overview": {
            "static": {
                "times_unix": TIMES,
                "mass_amu": [1.0, 4.0, 16.0, 32.0],
                "mass_eflux": [
                    [1e5, 2e4, 3e3, 2e3],
                    [2e5, 3e4, 5e3, 4e3],
                    [1.5e5, 2.5e4, 4e3, 3e3],
                ],
                "mass_amu_by_time": [
                    [1.0, 4.0, 16.0, 32.0],
                    [1.0, 4.0, 16.0, 32.0],
                    [1.0, 4.0, 16.0, 32.0],
                ],
                "mass_eflux_by_time": [
                    [1e5, 2e4, 3e3, 2e3],
                    [2e5, 3e4, 5e3, 4e3],
                    [1.5e5, 2.5e4, 4e3, 3e3],
                ],
            },
            "mag": {
                "times_unix": TIMES,
                "bx_nT": [5.0, 6.0, 7.0],
                "by_nT": [-2.0, -1.0, 0.0],
                "bz_nT": [1.0, 1.5, 2.0],
                "bmag_nT": [5.5, 6.3, 7.3],
                "x_mso_rm": [1.2, 1.1, 1.0],
                "y_mso_rm": [0.1, 0.1, 0.1],
                "z_mso_rm": [0.2, 0.2, 0.2],
                "latitude_deg": [10.0, 11.0, 12.0],
                "longitude_deg": [100.0, 101.0, 102.0],
                "altitude_km": [800.0, 700.0, 600.0],
            },
            "swe": {
                "times_unix": TIMES,
                "energy_eV": [20.0, 40.0],
                "omni_eflux": [[1e5, 2e5], [2e5, 3e5], [3e5, 4e5]],
                "pad_bands": [
                    {
                        "energy_band_eV": [20.0, 80.0],
                        "pitch_deg": [0.0, 90.0, 180.0],
                        "eflux": [
                            [1e5, 2e5, 1e5],
                            [2e5, 3e5, 2e5],
                            [1e5, 2e5, 1e5],
                        ],
                    }
                ],
            },
            "region_id": {
                "times_unix": TIMES,
                "region_id": [2, 3, 0],
            },
            "magnetic_topology": {
                "times_unix": TIMES,
                "topology_id": [6, 7, 0],
                "topology": ["O-N", "DP", "unknown"],
            },
            "shape_parameter": {
                "times_unix": TIMES,
                "toward_shape_parameter": [0.4, 0.8, 1.2],
                "away_shape_parameter": [1.4, 1.0, 0.6],
            },
            "pad_score": {
                "times_unix": TIMES,
                "toward_pad_score": [-2.5, -0.5, 1.0],
                "away_pad_score": [1.5, 0.2, -3.0],
            },
        },
    }


class PanelSelectionTests(unittest.TestCase):
    def test_trace_command_line_flag(self) -> None:
        args = build_argument_parser().parse_args(
            ["--time", "2016-10-06T18:10:00Z", "--trace"]
        )
        self.assertTrue(args.trace)

    def test_vertical_line_specs_use_defaults_and_style_aliases(self) -> None:
        lines = parse_vertical_line_specs(
            [
                ["2016-10-06T18:09:00"],
                ["2016-10-06T18:10:00", "虚线", "red"],
                ["2016-10-06T18:11:00", "dotted", "#336699"],
            ]
        )
        self.assertEqual(lines[0][1:], ("-", "black"))
        self.assertEqual(lines[1][1:], ("--", "red"))
        self.assertEqual(lines[2][1:], (":", "#336699"))

    def test_vertical_line_specs_reject_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_vertical_line_specs(
                [["2016-10-06T18:10:00", "dashed", "red", "extra"]]
            )
        with self.assertRaises(ValueError):
            parse_vertical_line_specs(
                [["2016-10-06T18:10:00", "not-a-style"]]
            )
        with self.assertRaises(ValueError):
            parse_vertical_line_specs(
                [["2016-10-06T18:10:00", "solid", "not-a-color"]]
            )

    def test_panel_ids_follow_input_order_with_coordinate_footer_last(self) -> None:
        self.assertEqual(validate_panel_ids(["9", "5,1"]), (5, 1, 9))
        self.assertEqual(validate_panel_ids(["9", "10", "2"]), (10, 2, 9))
        self.assertEqual(validate_panel_ids(["11", "9", "2"]), (11, 2, 9))
        self.assertEqual(
            validate_panel_ids(["12", "13,14,10", "9", "2"]),
            (12, 13, 14, 10, 2, 9),
        )

    def test_panel_ids_reject_unknown_and_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            validate_panel_ids(["15"])
        with self.assertRaises(ValueError):
            validate_panel_ids(["5", "5"])

    def test_selected_mag_and_coordinate_panels_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mag_panels.png"
            result = plot_data_panels(
                summary=synthetic_summary(),
                target_time=TARGET,
                output_path=output,
                panel_ids=(5, 6, 9),
            )
            self.assertEqual(result["panel_ids"], [5, 6, 9])
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_custom_vertical_lines_render_and_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vertical_lines.png"
            lines = parse_vertical_line_specs(
                [
                    ["2016-10-06T18:09:30"],
                    ["2016-10-06T18:10:30", "dashdot", "blue"],
                ]
            )
            result = plot_data_panels(
                summary=synthetic_summary(),
                target_time=TARGET,
                output_path=output,
                panel_ids=(5, 6, 9),
                vertical_lines=lines,
            )
            self.assertEqual(
                result["vertical_lines"],
                [
                    {
                        "time": "2016-10-06T18:09:30+00:00",
                        "linestyle": "-",
                        "color": "black",
                    },
                    {
                        "time": "2016-10-06T18:10:30+00:00",
                        "linestyle": "-.",
                        "color": "blue",
                    },
                ],
            )
            self.assertTrue(output.exists())

    def test_default_selection_keeps_all_catalog_panels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "all_panels.png"
            result = plot_data_panels(
                summary=synthetic_summary(),
                target_time=TARGET,
                output_path=output,
            )
            self.assertEqual(result["panel_ids"], list(range(1, 10)))
            self.assertTrue(output.exists())

    def test_one_pad_panel_requires_only_one_energy_band(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pad_panel.png"
            result = plot_data_panels(
                summary=synthetic_summary(),
                target_time=TARGET,
                output_path=output,
                panel_ids=(7,),
                pad_energy_bands_eV=((20.0, 80.0),),
            )
            self.assertEqual(result["pad_energy_bands_by_panel"], {"7": [20.0, 80.0]})
            self.assertTrue(output.exists())

    def test_region_id_panel_precedes_coordinate_footer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "region_id_panel.png"
            result = plot_data_panels(
                summary=synthetic_summary(),
                target_time=TARGET,
                output_path=output,
                panel_ids=(10, 9),
            )
            self.assertEqual(result["panel_ids"], [10, 9])
            self.assertTrue(output.exists())

    def test_topology_id_panel_precedes_coordinate_footer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "topology_id_panel.png"
            result = plot_data_panels(
                summary=synthetic_summary(),
                target_time=TARGET,
                output_path=output,
                panel_ids=(11, 9),
            )
            self.assertEqual(result["panel_ids"], [11, 9])
            self.assertTrue(output.exists())

    def test_region_id_panel_draws_points_without_connecting_line(self) -> None:
        figure, axis = plt.subplots()
        try:
            plot_region_id_panel(axis, TIMES, [2, 3, 0])
            self.assertEqual(len(axis.lines), 0)
            self.assertGreater(len(axis.collections), 0)
        finally:
            plt.close(figure)

    def test_topology_id_panel_draws_points_without_connecting_line(self) -> None:
        figure, axis = plt.subplots()
        try:
            plot_topology_id_panel(axis, TIMES, [6, 7, 0])
            self.assertEqual(len(axis.lines), 0)
            self.assertGreater(len(axis.collections), 0)
        finally:
            plt.close(figure)

    def test_full_mass_panel_uses_log_mass_spectrogram(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "full_mass_panel.png"
            result = plot_data_panels(
                summary=synthetic_summary(),
                target_time=TARGET,
                output_path=output,
                panel_ids=(14, 9),
            )
            self.assertEqual(result["panel_ids"], [14, 9])
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_pad_score_panel_renders_toward_and_away_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pad_score_panel.png"
            result = plot_data_panels(
                summary=synthetic_summary(),
                target_time=TARGET,
                output_path=output,
                panel_ids=(13, 9),
            )
            self.assertEqual(result["panel_ids"], [13, 9])
            self.assertEqual(
                result["panel_names"],
                ["toward/away PAD score", "UTC and spacecraft coordinates"],
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_shape_parameter_panel_renders_toward_and_away_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "shape_parameter_panel.png"
            result = plot_data_panels(
                summary=synthetic_summary(),
                target_time=TARGET,
                output_path=output,
                panel_ids=(12, 9),
            )
            self.assertEqual(result["panel_ids"], [12, 9])
            self.assertEqual(
                result["panel_names"],
                ["toward/away shape parameter", "UTC and spacecraft coordinates"],
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

if __name__ == "__main__":
    unittest.main()
