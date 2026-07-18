from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from plot_maven_data_panels import plot_data_panels, validate_panel_ids


TARGET = datetime(2016, 10, 6, 18, 10, tzinfo=timezone.utc)
TIMES = [TARGET.timestamp() - 10.0, TARGET.timestamp(), TARGET.timestamp() + 10.0]


def synthetic_summary() -> dict:
    return {
        "samples": [{"target_time": TARGET.isoformat()}],
        "context_overview": {
            "static": {},
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
        },
    }


class PanelSelectionTests(unittest.TestCase):
    def test_panel_ids_accept_spaces_and_commas_in_catalog_order(self) -> None:
        self.assertEqual(validate_panel_ids(["9", "5,1"]), (1, 5, 9))
        self.assertEqual(validate_panel_ids(["9", "10", "2"]), (2, 10, 9))

    def test_panel_ids_reject_unknown_and_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            validate_panel_ids(["11"])
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


if __name__ == "__main__":
    unittest.main()
