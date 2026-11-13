from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import ROOT, settings

logger = logging.getLogger(__name__)


class IMDAuthError(PermissionError):
    status_code = 401


# Names and paths are taken from the official IMD API reference.
ENDPOINTS = {
    # Fetch the geography-bearing forecast first so its district/state IDs are
    # available when district observations and warning rows are loaded.
    "district_forecast": "state_district_rainfall_forecast",
    "state_rainfall": "staterainfall",
    "district_rainfall": "districtrainfall",
    "current_weather": "current_wx",
    "aws_arg": "aws_data",
    "subdivision_forecast": "subdivision_rainfall_forecast",
    "district_warnings": "districtwarning",
}


class IMDClient:
    def __init__(self, base_url: str | None = None, raw_dir: Path | None = None):
        self.base_url = (base_url or settings.imd_api_base).rstrip("/")
        self.raw_dir = raw_dir or ROOT / "data" / "raw" / "imd"
        self.session = requests.Session()
        retry = Retry(total=3, connect=3, read=3, backoff_factor=1,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"GET"}), raise_on_status=False)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        # IMD's reference does not document an auth scheme: only use a header
        # configured explicitly from account instructions supplied by IMD.
        if settings.imd_auth_header and settings.imd_auth_value:
            self.session.headers[settings.imd_auth_header] = settings.imd_auth_value
        self.session.headers["Accept"] = "application/json"

    def fetch(self, name: str, params: dict | None = None) -> tuple[object, int, Path]:
        if name not in ENDPOINTS:
            raise ValueError(f"Unknown documented IMD endpoint key: {name}")
        url = f"{self.base_url}/{ENDPOINTS[name]}"
        response = self.session.get(url, params=params, timeout=(10, 45))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        raw_path = self.raw_dir / f"{timestamp}_{safe_name}.json"
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError:
            payload = {"non_json_response": response.text[:2000]}
        raw_path.write_text(json.dumps({
            "source": "India Meteorological Department API",
            "endpoint": urlparse(response.url).path,
            "params": params or {},
            "fetched_at_utc": timestamp,
            "http_status": response.status_code,
            "payload": payload,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if response.status_code == 401:
            raise IMDAuthError(
                "IMD returned 401 Unauthorized. Register at https://api.imd.gov.in/public/ "
                "and configure only the authentication method specified by your IMD account."
            )
        response.raise_for_status()
        logger.info("IMD endpoint %s returned HTTP %s; saved raw response %s", name, response.status_code, raw_path)
        return payload, response.status_code, raw_path
