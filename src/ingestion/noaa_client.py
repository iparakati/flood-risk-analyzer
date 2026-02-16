"""Client for NOAA Climate Data Online (CDO) API.

Fetches precipitation, temperature, and storm event data
to enrich flood risk analysis with climate context.
"""

import time

import httpx
import pandas as pd
from loguru import logger

from src.config import NOAA_API_TOKEN, RAW_DATA_DIR

NOAA_CDO_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"


class NOAAClient:
    """Fetches climate data from NOAA's Climate Data Online API."""

    def __init__(self, token: str = NOAA_API_TOKEN):
        if not token:
            logger.warning(
                "No NOAA API token set. Get one free at https://www.ncdc.noaa.gov/cdo-web/token"
            )
        self.token = token
        self.headers = {"token": self.token}

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{NOAA_CDO_BASE}/{endpoint}"
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=self.headers, params=params or {})
            resp.raise_for_status()
            return resp.json()

    def fetch_precipitation(
        self,
        state_fips: str,
        start_date: str,
        end_date: str,
        dataset_id: str = "GHCND",
        datatype_id: str = "PRCP",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch daily precipitation data for a state.

        Args:
            state_fips: Two-digit FIPS code (e.g., '12' for Florida).
            start_date: ISO date string 'YYYY-MM-DD'.
            end_date: ISO date string 'YYYY-MM-DD'.
            dataset_id: NOAA dataset identifier.
            datatype_id: Data type (PRCP = precipitation).
            limit: Max records per request.

        Returns:
            DataFrame with precipitation records.
        """
        params = {
            "datasetid": dataset_id,
            "datatypeid": datatype_id,
            "locationid": f"FIPS:{state_fips}",
            "startdate": start_date,
            "enddate": end_date,
            "limit": limit,
            "units": "standard",
        }

        all_records = []
        offset = 1

        while True:
            params["offset"] = offset
            try:
                data = self._get("data", params)
            except httpx.HTTPStatusError as e:
                logger.error(f"NOAA API error: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            all_records.extend(results)
            offset += limit

            count = data.get("metadata", {}).get("resultset", {}).get("count", 0)
            if offset > count:
                break

            time.sleep(0.3)  # rate limit: 5 req/sec

        df = pd.DataFrame(all_records)
        logger.info(f"Fetched {len(df)} precipitation records for FIPS {state_fips}")
        return df

    def fetch_storm_events(
        self,
        state_fips: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch storm event data from NOAA.

        Uses the GHCND dataset to get extreme precipitation events
        that could correlate with flood claims.
        """
        return self.fetch_precipitation(
            state_fips=state_fips,
            start_date=start_date,
            end_date=end_date,
            datatype_id="PRCP",
        )

    def get_stations(self, state_fips: str, limit: int = 1000) -> pd.DataFrame:
        """Get weather stations in a state."""
        params = {
            "locationid": f"FIPS:{state_fips}",
            "datasetid": "GHCND",
            "limit": limit,
        }
        data = self._get("stations", params)
        return pd.DataFrame(data.get("results", []))
