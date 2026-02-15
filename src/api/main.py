"""FastAPI application for the Flood Risk Analyzer.

Provides REST endpoints for claims data, coverage gap analysis,
and forecasting results.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.analysis.coverage_gap import compute_coverage_metrics, find_top_gaps
from src.forecasting.timeseries import run_forecast
from src.ingestion.openfema_client import OpenFEMAClient

app = FastAPI(
    title="Flood Risk Analyzer API",
    description="API for NFIP flood risk analysis — coverage gaps, forecasting, and claims data.",
    version="0.1.0",
)

client = OpenFEMAClient()


# --- Response models ---


class HealthResponse(BaseModel):
    status: str
    version: str


class DataSummary(BaseModel):
    dataset: str
    rows: int
    columns: int
    column_names: list[str]


class CoverageGapRecord(BaseModel):
    geography: str
    claim_count: float
    policy_count: float
    claims_per_policy: float
    avg_claim: float
    total_paid: float
    gap_score: float
    risk_cluster: int


class ForecastPoint(BaseModel):
    date: str
    forecast: float
    lower_ci: float
    upper_ci: float


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="0.1.0")


@app.get("/data/{dataset}", response_model=DataSummary)
def get_data_summary(
    dataset: str,
    max_records: int = Query(1000, ge=100, le=50000),
):
    """Fetch a dataset sample and return a summary."""
    if dataset not in ("claims", "policies", "disasters"):
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset}")

    df = client.fetch_dataset(dataset, max_records=max_records)
    return DataSummary(
        dataset=dataset,
        rows=len(df),
        columns=len(df.columns),
        column_names=list(df.columns),
    )


@app.get("/coverage-gaps", response_model=list[CoverageGapRecord])
def get_coverage_gaps(
    geo_col: str = Query("countyCode", description="Geography grouping column"),
    max_records: int = Query(5000, ge=100, le=50000),
    top_n: int = Query(20, ge=1, le=100),
):
    """Compute and return the top coverage gap regions."""
    claims = client.fetch_claims(max_records=max_records)
    policies = client.fetch_policies(max_records=max_records)

    if claims.empty or policies.empty:
        raise HTTPException(status_code=503, detail="Unable to fetch data from OpenFEMA")

    gaps = find_top_gaps(claims, policies, geo_col=geo_col, top_n=top_n)

    records = []
    for _, row in gaps.iterrows():
        records.append(CoverageGapRecord(
            geography=str(row.get(geo_col, "")),
            claim_count=float(row.get("claim_count", 0)),
            policy_count=float(row.get("policy_count", 0)),
            claims_per_policy=float(row.get("claims_per_policy", 0)),
            avg_claim=float(row.get("avg_claim", 0)),
            total_paid=float(row.get("total_paid", 0)),
            gap_score=float(row.get("gap_score", 0)),
            risk_cluster=int(row.get("risk_cluster", 0)),
        ))
    return records


@app.get("/forecast", response_model=list[ForecastPoint])
def get_forecast(
    target: str = Query("claim_count", description="Forecast target column"),
    periods: int = Query(24, ge=6, le=60),
    backend: str = Query("statsmodels", description="Model backend"),
    state: str | None = Query(None, description="Optional state filter"),
    max_records: int = Query(10000, ge=1000, le=50000),
):
    """Run a time-series forecast and return predicted values."""
    claims = client.fetch_claims(max_records=max_records)
    if claims.empty:
        raise HTTPException(status_code=503, detail="Unable to fetch claims data")

    geo_col = "state" if state else None
    try:
        _, forecast_df = run_forecast(
            claims,
            target_col=target,
            periods=periods,
            backend=backend,
            geo_col=geo_col,
            geo_value=state,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")

    if forecast_df.empty:
        raise HTTPException(status_code=422, detail="Not enough data for forecasting")

    return [
        ForecastPoint(
            date=str(row["date"]),
            forecast=round(float(row["forecast"]), 2),
            lower_ci=round(float(row["lower_ci"]), 2),
            upper_ci=round(float(row["upper_ci"]), 2),
        )
        for _, row in forecast_df.iterrows()
    ]
