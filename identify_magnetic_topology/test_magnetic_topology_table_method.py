from __future__ import annotations

import pandas as pd

from identify_magnetic_topology.magnetic_topology_table_method import (
    build_topology_dataframe,
)


def shape_row(time_unix: float, shape_value: float = float("nan")) -> dict:
    return {
        "time_unix": time_unix,
        "time_utc": pd.Timestamp(time_unix, unit="s", tz="UTC").isoformat(),
        "away_shape_parameter": shape_value,
        "toward_shape_parameter": shape_value,
    }


def region_row(time_unix: float, region_id: int, region_name: str) -> dict:
    return {
        "time_unix": time_unix,
        "time_utc": pd.Timestamp(time_unix, unit="s", tz="UTC").isoformat(),
        "region_id": region_id,
        "region_name": region_name,
        "confidence": 0.8,
        "reason": "test_region",
        "geometry_only": region_id in {1, 2},
    }


def build(shape_rows: list[dict], region_rows: list[dict], pad_df=None):
    return build_topology_dataframe(
        shape_rows=shape_rows,
        pad_df=pd.DataFrame() if pad_df is None else pad_df,
        ratio_by_time={},
        photoelectron_shape_threshold=1.0,
        max_pad_delta_seconds=6.0,
        loss_cone_pad_score_threshold=-3.0,
        region_df=pd.DataFrame(region_rows),
        max_region_delta_seconds=2.0,
    )


def test_region_id_zero_keeps_xu_table_result():
    df = build(
        [shape_row(100.0)],
        [region_row(100.0, 0, "Unknown")],
    )
    row = df.iloc[0]
    assert row["topology"] == "unknown"
    assert row["topology_id"] == 0
    assert row["topology_source"] == "xu2019_shape_pad_table"
    assert row["table_topology"] == "unknown"
    assert row["region_id"] == 0


def test_region_id_one_overrides_as_draped_dp():
    df = build(
        [shape_row(200.0)],
        [region_row(200.0, 1, "Solar wind")],
    )
    row = df.iloc[0]
    assert row["topology"] == "DP"
    assert row["topology_id"] == 7
    assert row["topology_subcase"] == "7b"
    assert row["topology_source"] == "region_id_1_2_override"
    assert row["region_id_geometry_only"]
    assert row["region_id"] == 1


def test_region_id_two_overrides_as_draped_dp():
    df = build(
        [shape_row(250.0)],
        [region_row(250.0, 2, "Magnetosheath")],
    )
    row = df.iloc[0]
    assert row["topology"] == "DP"
    assert row["topology_id"] == 7
    assert row["topology_subcase"] == "7b"
    assert row["topology_source"] == "region_id_1_2_override"
    assert row["region_id_geometry_only"]
    assert row["region_id"] == 2


def test_region_id_three_keeps_xu_table_result():
    time_unix = 300.0
    pad_df = pd.DataFrame(
        [
            {
                "time_unix": time_unix,
                "time": pd.Timestamp(time_unix, unit="s", tz="UTC").isoformat(),
                "away_pad_score": 0.0,
                "toward_pad_score": 0.0,
                "valid": True,
                "reason": "test_pad",
            }
        ]
    )
    df = build(
        [shape_row(time_unix, shape_value=2.0)],
        [region_row(time_unix, 3, "Ionosphere")],
        pad_df=pad_df,
    )
    row = df.iloc[0]
    assert row["topology"] == "DP"
    assert row["topology_id"] == 7
    assert row["topology_subcase"] == "7a"
    assert row["topology_source"] == "xu2019_shape_pad_table"
    assert row["region_id"] == 3
