"""Tests for the time-series forecasting module."""

import pandas as pd
import pytest

from src.forecasting.timeseries import forecast_statsmodels, prepare_claims_timeseries


@pytest.fixture
def sample_claims():
    """Generate synthetic monthly claims data spanning multiple years."""
    import numpy as np

    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", "2023-12-31", freq="D")
    # Sample ~2000 claims spread across the date range
    n = 2000
    sampled_dates = rng.choice(dates, size=n, replace=True)
    return pd.DataFrame({
        "dateOfLoss": sampled_dates,
        "state": rng.choice(["FL", "TX", "LA"], size=n),
        "countyCode": rng.choice(["12001", "48201", "22071"], size=n),
        "amountPaidOnBuildingClaim": rng.uniform(1000, 200000, size=n),
        "amountPaidOnContentsClaim": rng.uniform(0, 50000, size=n),
    })


def test_prepare_claims_timeseries(sample_claims):
    ts = prepare_claims_timeseries(sample_claims)
    assert isinstance(ts, pd.DataFrame)
    assert "claim_count" in ts.columns
    assert "total_paid" in ts.columns
    assert len(ts) > 12  # Should have multiple months


def test_prepare_claims_timeseries_filtered(sample_claims):
    ts = prepare_claims_timeseries(sample_claims, geo_col="state", geo_value="FL")
    assert len(ts) > 0
    # Should have fewer records than unfiltered
    ts_all = prepare_claims_timeseries(sample_claims)
    assert ts["claim_count"].sum() < ts_all["claim_count"].sum()


def test_forecast_statsmodels(sample_claims):
    ts = prepare_claims_timeseries(sample_claims)
    forecast = forecast_statsmodels(ts, target_col="claim_count", periods=12)
    assert len(forecast) == 12
    assert "forecast" in forecast.columns
    assert "lower_ci" in forecast.columns
    assert "upper_ci" in forecast.columns


def test_forecast_short_series():
    """Forecasting with minimal data should still work (no seasonal)."""
    ts = pd.DataFrame({
        "claim_count": [10, 15, 12, 18, 20, 25, 22, 30],
        "total_paid": [1e5, 1.5e5, 1.2e5, 1.8e5, 2e5, 2.5e5, 2.2e5, 3e5],
        "avg_severity": [1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4],
    }, index=pd.date_range("2023-01-01", periods=8, freq="MS"))

    forecast = forecast_statsmodels(ts, target_col="claim_count", periods=6)
    assert len(forecast) == 6
