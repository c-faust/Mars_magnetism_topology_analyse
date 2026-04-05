from __future__ import annotations
"""
Download helper for the MAVEN workflow.

This file is the data-ingest入口:
1. Query the LASP public file API for a given day
2. Pick the "best" file for each requested product
3. Download the file into the local `data/maven/...` tree

When you read the rest of the project, you can treat this module as the only
place that knows:
- how to talk to the remote data service
- how to map instrument/product names to LASP filenames
- where downloaded files should be stored locally
"""

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = BASE_DIR / "data" / "maven"

FILE_NAMES_URL = (
    "https://lasp.colorado.edu/maven/sdc/public/files/api/v1/search/science/fn_metadata/file_names"
)
DOWNLOAD_URL = (
    "https://lasp.colorado.edu/maven/sdc/public/files/api/v1/search/science/fn_metadata/download"
)

FILENAME_PATTERN = re.compile(
    r"^mvn_"
    r"(?P<instrument>[a-zA-Z0-9]+)_"
    r"(?P<level>l[a-zA-Z0-9]+)"
    r"(?P<description>|_[a-zA-Z0-9\-]+)_"
    r"(?P<year>[0-9]{4})"
    r"(?P<month>[0-9]{2})"
    r"(?P<day>[0-9]{2})"
    r"(?P<time>|T[0-9]{6}|t[0-9]{6})_"
    r"v(?P<version>[0-9]+)_"
    r"r(?P<revision>[0-9]+)\."
    r"(?P<extension>cdf|xml|sts|md5|tab)"
    r"(?P<gz>\.gz)?$"
)


@dataclass(frozen=True)
class ProductSpec:
    """Describe one downloadable science product.

    `datatype` is the human-facing product name used by our pipeline, while
    `aliases` contains the strings that may appear in LASP filenames.
    """
    instrument: str
    datatype: str
    aliases: tuple[str, ...]
    level: str = "l2"
    format_preference: tuple[str, ...] = ("cdf", "sts", "tab")


PIPELINE_PRODUCTS = (
    ProductSpec("swe", "svypad", ("svypad",)),
    ProductSpec("sta", "c6-32e64m", ("c6-32e64m", "32e64m"), format_preference=("cdf",)),
    ProductSpec("mag", "sunstate-1sec", ("sunstate-1sec", "ss1s"), format_preference=("sts", "tab")),
    ProductSpec("mag", "planetocentric-1sec", ("planetocentric-1sec", "pc1s"), format_preference=("sts", "tab")),
)
#32e64m means the data shape c6 indicates how the data is compressed

def build_session() -> requests.Session:
    """Create one HTTP session with retry logic.

    All downloads share the same session so we reuse connections and get a
    small amount of resilience against temporary server/network errors.
    """
    session = requests.Session()
    retry = Retry(   #if the server doesnt answer,this function define the retry strategy
        total=3,  #retry three times
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504], 
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry) 
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_filename(filename: str) -> dict | None:
    """Parse a MAVEN filename into structured metadata.

    The LASP API mostly gives us file names, so this parser is how we recover
    information such as instrument, product description, date, version, and
    revision in a machine-friendly way.
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    info = match.groupdict()
    info["description"] = info["description"].lstrip("_")
    info["version"] = int(info["version"])
    info["revision"] = int(info["revision"])
    return info


def daterange(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def fetch_filenames(
    session: requests.Session,
    instrument: str,
    start_date: date,
    end_date: date,
    level: str = "l2",
) -> list[str]:
    """Ask LASP for all filenames that match one instrument and date range."""
    params = {
        "instrument": instrument,
        "level": level,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    response = session.get(FILE_NAMES_URL, params=params, timeout=(10, 30))
    response.raise_for_status()
    payload = response.text.strip()
    if not payload:
        return []
    return [item.strip() for item in payload.split(",") if item.strip()]


def choose_best_filename(filenames: list[str], spec: ProductSpec, day: date) -> tuple[str, dict] | tuple[None, None]:
    """Pick the preferred file for one product on one day.

    Selection priority is:
    1. preferred file extension
    2. highest version number
    3. highest revision number

    This keeps the rest of the pipeline simple: downstream code can assume it
    receives the best local candidate rather than handling many alternatives.
    """
    day_code = day.strftime("%Y%m%d")
    aliases = [alias.lower() for alias in spec.aliases]
    extension_rank = {suffix: index for index, suffix in enumerate(spec.format_preference)}
    candidates: list[tuple[tuple[int, int, int, str], str, dict]] = []

    for filename in filenames:
        parsed = parse_filename(filename)
        if not parsed:
            continue
        if parsed["instrument"] != spec.instrument or parsed["level"] != spec.level:
            continue
        if f"{parsed['year']}{parsed['month']}{parsed['day']}" != day_code:
            continue

        description = parsed["description"].lower()
        if not any(
            description == alias or description.startswith(alias) or alias in description
            for alias in aliases
        ):
            continue

        ext = parsed["extension"]
        if ext not in extension_rank:
            continue

        sort_key = (
            extension_rank[ext],
            -parsed["version"],
            -parsed["revision"],
            filename,
        )
        candidates.append((sort_key, filename, parsed))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: item[0])
    _, filename, parsed = candidates[0]
    return filename, parsed


def build_local_path(data_root: Path, filename: str, parsed: dict) -> Path:
    """Map one remote MAVEN filename into our local folder convention."""
    year = parsed["year"]
    month = parsed["month"]
    return data_root / parsed["instrument"] / parsed["level"] / parsed["description"] / year / month / filename


def download_file(session: requests.Session, filename: str, local_path: Path) -> Path:
    """Download a single file unless it already exists locally."""
    ensure_parent(local_path)
    if local_path.exists():
        return local_path

    with session.get(DOWNLOAD_URL, params={"file": filename}, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        with local_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return local_path


def download_product_for_day(
    session: requests.Session,
    spec: ProductSpec,
    day: date,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Resolve and download one product for one UTC day."""
    filenames = fetch_filenames(session, instrument=spec.instrument, start_date=day, end_date=day, level=spec.level)
    filename, parsed = choose_best_filename(filenames, spec=spec, day=day)
    if not filename or not parsed:
        raise FileNotFoundError(f"No {spec.instrument}/{spec.datatype} file was found for {day.isoformat()}.")

    local_path = build_local_path(data_root, filename, parsed)
    return download_file(session, filename, local_path)


def download_products_for_timestamp(
    target_time: datetime,
    specs: Iterable[ProductSpec] = PIPELINE_PRODUCTS,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Path]:
    """Download the default pipeline products for a target timestamp.

    Even though the input is a timestamp, the public MAVEN files are daily
    products, so internally we download by `target_time.date()`.
    """
    session = build_session()
    day = target_time.date()
    downloaded: dict[str, Path] = {}

    for spec in specs:
        key = f"{spec.instrument}_{spec.datatype}"
        downloaded[key] = download_product_for_day(session, spec=spec, day=day, data_root=data_root)

    return downloaded


def parse_iso_timestamp(value: str) -> datetime:
    """Parse flexible ISO-like input and normalize it to UTC."""
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

#define what options a user can type when running the script from the terminal 
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download MAVEN files for a target timestamp.")
    parser.add_argument(
        "--time",
        required=True,
        help="Target timestamp, for example 2024-11-07T12:00:00.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Directory used to store downloaded MAVEN files.",
    )
    return parser
# for example we can type "python download.py --time 2024-11-07T12:00:00 --data-root G:/MAVEN/data" in the terminal 

def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    target_time = parse_iso_timestamp(args.time)
    data_root = Path(args.data_root).expanduser().resolve()
    downloaded = download_products_for_timestamp(target_time=target_time, data_root=data_root)

    for key, path in downloaded.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
