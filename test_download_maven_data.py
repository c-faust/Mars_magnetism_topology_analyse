from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import requests

import download_maven_data
from download_maven_data import ProductSpec, download_products_for_timestamp


def test_product_failure_does_not_stop_later_downloads(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    specs = (
        ProductSpec("swe", "first", ("first",)),
        ProductSpec("lpw", "missing", ("missing",)),
        ProductSpec("mag", "ss1s", ("ss1s",), format_preference=("sts",)),
    )
    calls: list[str] = []

    monkeypatch.setattr(download_maven_data, "build_session", object)

    def fake_download(session, spec, day, data_root):
        del session, day, data_root
        calls.append(spec.datatype)
        if spec.datatype == "missing":
            raise FileNotFoundError("remote product is absent")
        return tmp_path / f"{spec.datatype}.dat"

    monkeypatch.setattr(
        download_maven_data,
        "download_product_for_day",
        fake_download,
    )

    downloaded = download_products_for_timestamp(
        datetime(2020, 7, 23, tzinfo=timezone.utc),
        specs=specs,
        data_root=tmp_path,
    )

    assert calls == ["first", "missing", "ss1s"]
    assert set(downloaded) == {"swe_first", "mag_ss1s"}
    assert "skip lpw_missing for 2020-07-23" in capsys.readouterr().out


def test_network_failure_is_skipped_and_returns_only_successes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    specs = (
        ProductSpec("swi", "failed", ("failed",)),
        ProductSpec("mag", "ss1s", ("ss1s",), format_preference=("sts",)),
    )

    monkeypatch.setattr(download_maven_data, "build_session", object)

    def fake_download(session, spec, day, data_root):
        del session, day, data_root
        if spec.datatype == "failed":
            raise requests.Timeout("temporary timeout")
        return tmp_path / "mag.sts"

    monkeypatch.setattr(
        download_maven_data,
        "download_product_for_day",
        fake_download,
    )

    downloaded = download_products_for_timestamp(
        datetime(2020, 7, 23, tzinfo=timezone.utc),
        specs=specs,
        data_root=tmp_path,
    )

    assert downloaded == {"mag_ss1s": tmp_path / "mag.sts"}
