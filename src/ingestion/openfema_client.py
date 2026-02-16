"""Client for the OpenFEMA API.

Fetches NFIP claims, policies, and disaster declarations data with
pagination support and local caching.
"""

import time
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.config import OPENFEMA_BASE_URL, RAW_DATA_DIR

# OpenFEMA dataset endpoints
DATASETS = {
    "claims": "/FimaNfipClaims",
    "policies": "/FimaNfipPolicies",
    "disasters": "/DisasterDeclarationsSummaries",
}

# Default fields to fetch per dataset (keeps downloads manageable)
DEFAULT_FIELDS = {
    "claims": [
        "dateOfLoss",
        "yearOfLoss",
        "monthOfLoss",
        "state",
        "countyCode",
        "censusTract",
        "floodZone",
        "occupancyType",
        "amountPaidOnBuildingClaim",
        "amountPaidOnContentsClaim",
        "totalBuildingInsuranceCoverage",
        "totalContentsInsuranceCoverage",
        "reportedZipCode",
        "latitude",
        "longitude",
    ],
    "policies": [
        "policyEffectiveDate",
        "policyTerminationDate",
        "policyCount",
        "propertyState",
        "countyCode",
        "censusTract",
        "floodZone",
        "occupancyType",
        "totalBuildingInsuranceCoverage",
        "totalContentsInsuranceCoverage",
        "reportedZipCode",
    ],
    "disasters": [
        "disasterNumber",
        "declarationDate",
        "state",
        "fipsStateCode",
        "fipsCountyCode",
        "designatedArea",
        "declarationType",
        "incidentType",
        "title",
        "incidentBeginDate",
        "incidentEndDate",
    ],
}

PAGE_SIZE = 1000  # OpenFEMA max per request


class OpenFEMAClient:
    """Handles paginated data retrieval from the OpenFEMA API."""

    def __init__(self, base_url: str = OPENFEMA_BASE_URL, timeout: float = 60.0):
        self.base_url = base_url
        self.timeout = timeout

    def _build_url(self, dataset: str, skip: int, top: int, filters: dict | None = None) -> str:
        endpoint = DATASETS[dataset]
        url = f"{self.base_url}{endpoint}?$skip={skip}&$top={top}&$format=json"

        # Select only the fields we need
        if dataset in DEFAULT_FIELDS:
            fields = ",".join(DEFAULT_FIELDS[dataset])
            url += f"&$select={fields}"

        if filters:
            filter_parts = []
            for key, value in filters.items():
                if isinstance(value, str):
                    filter_parts.append(f"{key} eq '{value}'")
                else:
                    filter_parts.append(f"{key} eq {value}")
            url += "&$filter=" + " and ".join(filter_parts)

        return url

    def fetch_dataset(
        self,
        dataset: str,
        max_records: int | None = None,
        filters: dict | None = None,
        cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch a full dataset from OpenFEMA with pagination.

        Args:
            dataset: One of 'claims', 'policies', or 'disasters'.
            max_records: Cap on total records to fetch. None = fetch all.
            filters: OData filter expressions as {field: value}.
            cache: If True, save to and load from local parquet cache.

        Returns:
            DataFrame of all fetched records.
        """
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASETS.keys())}")

        cache_path = RAW_DATA_DIR / f"{dataset}.parquet"
        if cache and cache_path.exists():
            logger.info(f"Loading cached {dataset} from {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Fetching {dataset} from OpenFEMA API...")
        all_records: list[dict] = []
        skip = 0

        # Determine the JSON key OpenFEMA uses for this dataset
        endpoint_key_map = {
            "claims": "FimaNfipClaims",
            "policies": "FimaNfipPolicies",
            "disasters": "DisasterDeclarationsSummaries",
        }
        json_key = endpoint_key_map[dataset]

        max_retries = 3

        with httpx.Client(timeout=self.timeout) as client:
            pbar = tqdm(desc=f"Fetching {dataset}", unit=" records")
            retries = 0
            while True:
                url = self._build_url(dataset, skip, PAGE_SIZE, filters)
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    retries = 0  # reset on success
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error fetching {dataset}: {e}")
                    break
                except httpx.RequestError as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries exceeded fetching {dataset}: {e}")
                        break
                    logger.warning(f"Request error (attempt {retries}/{max_retries}), retrying in 5s: {e}")
                    time.sleep(5)
                    continue

                data = resp.json()
                records = data.get(json_key, [])

                if not records:
                    break

                all_records.extend(records)
                pbar.update(len(records))
                skip += PAGE_SIZE

                if max_records and len(all_records) >= max_records:
                    all_records = all_records[:max_records]
                    break

                # Respect rate limits
                time.sleep(0.25)

            pbar.close()

        logger.info(f"Fetched {len(all_records):,} {dataset} records")
        df = pd.DataFrame(all_records)

        if cache and not df.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached {dataset} to {cache_path}")

        return df

    def fetch_claims(self, max_records: int | None = None, **kwargs) -> pd.DataFrame:
        """Convenience method to fetch NFIP claims."""
        return self.fetch_dataset("claims", max_records=max_records, **kwargs)

    def fetch_policies(self, max_records: int | None = None, **kwargs) -> pd.DataFrame:
        """Convenience method to fetch NFIP policies."""
        return self.fetch_dataset("policies", max_records=max_records, **kwargs)

    def fetch_disasters(self, max_records: int | None = None, **kwargs) -> pd.DataFrame:
        """Convenience method to fetch disaster declarations."""
        return self.fetch_dataset("disasters", max_records=max_records, **kwargs)


def ingest_all(max_records: int | None = 10000) -> dict[str, pd.DataFrame]:
    """Run full ingestion pipeline for all datasets.

    Args:
        max_records: Limit per dataset (None for full download).

    Returns:
        Dict mapping dataset names to DataFrames.
    """
    client = OpenFEMAClient()
    results = {}
    for dataset in DATASETS:
        df = client.fetch_dataset(dataset, max_records=max_records)
        results[dataset] = df
        logger.info(f"{dataset}: {len(df):,} rows, {len(df.columns)} columns")
    return results


if __name__ == "__main__":
    # Quick test: fetch a small sample of each dataset
    data = ingest_all(max_records=1000)
    for name, df in data.items():
        print(f"\n{name}: {df.shape}")
        print(df.head())
