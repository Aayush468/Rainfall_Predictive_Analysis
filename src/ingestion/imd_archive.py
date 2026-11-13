from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import ROOT

logger = logging.getLogger(__name__)
ARCHIVE_URL = "https://imdpune.gov.in/cmpg/Griddata/RF25.php"
NETCDF_SIGNATURES = (b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n")


def _is_netcdf(path: Path) -> bool:
    with path.open("rb") as stream:
        signature = stream.read(8)
    return any(signature.startswith(candidate) for candidate in NETCDF_SIGNATURES)


def download_year(year: int, output_dir: Path | None = None, session: requests.Session | None = None) -> Path:
    if not 1901 <= year <= 2024:
        raise ValueError("The IMD archive currently publishes yearly files for 1901 through 2024")
    output_dir = output_dir or ROOT / "data" / "raw" / "imd"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"RF25_ind{year}_rfp25.nc"
    if destination.exists() and _is_netcdf(destination):
        logger.info("Already have valid archive file %s", destination.name)
        return destination

    client = session or requests.Session()
    if session is None:
        retry = Retry(total=4, connect=4, read=4, backoff_factor=1,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"POST"}))
        client.mount("https://", HTTPAdapter(max_retries=retry))
    response = client.post(
        ARCHIVE_URL, data={"RF25": str(year)},
        headers={"Referer": "https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html"},
        stream=True, timeout=(20, 120),
    )
    response.raise_for_status()
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with temp_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
        with temp_path.open("rb") as downloaded:
            signature = downloaded.read(8)
            if not any(signature.startswith(candidate) for candidate in NETCDF_SIGNATURES):
                raise ValueError(f"IMD did not return a NetCDF file for {year}; response was not saved")
        os.replace(temp_path, destination)
    finally:
        response.close()
        if temp_path.exists():
            temp_path.unlink()
    logger.info("Downloaded %s (%d bytes)", destination.name, destination.stat().st_size)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download yearly official IMD 0.25-degree rainfall NetCDF files")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw" / "imd")
    parser.add_argument("--import-to-db", action="store_true",
                        help="Import each downloaded year to PostgreSQL after download")
    parser.add_argument("--max-cells", type=int, default=200,
                        help="Consistent spatial sample size used with --import-to-db")
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("--end-year must be greater than or equal to --start-year")
    if args.import_to_db and args.max_cells < 1:
        parser.error("--max-cells must be at least 1")
    for year in range(args.start_year, args.end_year + 1):
        try:
            path = download_year(year, args.out_dir)
            print(path)
            if args.import_to_db:
                from .netcdf_import import import_netcdf
                count = import_netcdf(path, max_cells=args.max_cells)
                print(f"Imported {count:,} observations for {year}")
        except Exception as exc:
            logger.exception("Failed to download IMD year %s", year)
            print(f"Failed to download {year}: {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
