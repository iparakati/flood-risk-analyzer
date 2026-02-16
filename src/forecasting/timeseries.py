"""Time-series forecasting for flood claim frequency and severity.

Uses statsmodels (ARIMA/SARIMAX) for claim frequency and severity
forecasting by region. Prophet is available as an alternative backend.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning)


def prepare_claims_timeseries(
    claims: pd.DataFrame,
    geo_col: str | None = None,
    geo_value: str | None = None,
    freq: str = "MS",
) -> pd.DataFrame:
    """Convert raw claims data into a time-series DataFrame.

    Args:
        claims: Raw claims DataFrame from OpenFEMA.
        geo_col: Optional geography column to filter on.
        geo_value: Value to filter geo_col by.
        freq: Pandas frequency string ('MS' = month start, 'QS' = quarter).

    Returns:
        DataFrame indexed by date with 'claim_count' and 'total_paid' columns.
    """
    df = claims.copy()

    if geo_col and geo_value:
        df = df[df[geo_col] == geo_value]

    # Parse date
    if "dateOfLoss" in df.columns:
        df["date"] = pd.to_datetime(df["dateOfLoss"], errors="coerce")
    elif "yearOfLoss" in df.columns and "monthOfLoss" in df.columns:
        df["date"] = pd.to_datetime(
            df["yearOfLoss"].astype(str) + "-" + df["monthOfLoss"].astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
    else:
        raise ValueError("Claims data must have 'dateOfLoss' or 'yearOfLoss'+'monthOfLoss' columns")

    df = df.dropna(subset=["date"])
    df["totalPaid"] = (
        df.get("amountPaidOnBuildingClaim", pd.Series(0, index=df.index)).fillna(0)
        + df.get("amountPaidOnContentsClaim", pd.Series(0, index=df.index)).fillna(0)
    )

    ts = df.set_index("date").resample(freq).agg(
        claim_count=("totalPaid", "count"),
        total_paid=("totalPaid", "sum"),
        avg_severity=("totalPaid", "mean"),
    )
    ts = ts.fillna(0)
    return ts


def forecast_statsmodels(
    ts: pd.DataFrame,
    target_col: str = "claim_count",
    periods: int = 24,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
) -> pd.DataFrame:
    """Forecast using SARIMAX from statsmodels.

    Args:
        ts: Time-series DataFrame (output from prepare_claims_timeseries).
        target_col: Column to forecast.
        periods: Number of future periods to predict.
        order: ARIMA (p, d, q) order.
        seasonal_order: Seasonal (P, D, Q, s) order.

    Returns:
        DataFrame with 'date', 'forecast', 'lower_ci', 'upper_ci' columns.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = ts[target_col].astype(float)

    if len(y) < 24:
        logger.warning(f"Only {len(y)} observations — using simple ARIMA (no seasonal component)")
        seasonal_order = (0, 0, 0, 0)

    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)

    try:
        results = model.fit(disp=False, maxiter=200)
    except Exception as e:
        logger.warning(f"SARIMAX fit failed ({e}), falling back to simple ARIMA")
        model = SARIMAX(y, order=order, seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False, maxiter=200)

    forecast = results.get_forecast(steps=periods)
    pred = forecast.predicted_mean
    ci = forecast.conf_int()

    forecast_df = pd.DataFrame({
        "date": pred.index,
        "forecast": pred.values,
        "lower_ci": ci.iloc[:, 0].values,
        "upper_ci": ci.iloc[:, 1].values,
    })

    logger.info(
        f"Forecast {periods} periods for '{target_col}' "
        f"(AIC={results.aic:.1f})"
    )
    return forecast_df


def forecast_prophet(
    ts: pd.DataFrame,
    target_col: str = "claim_count",
    periods: int = 24,
    freq: str = "MS",
) -> pd.DataFrame:
    """Forecast using Facebook Prophet.

    Args:
        ts: Time-series DataFrame (output from prepare_claims_timeseries).
        target_col: Column to forecast.
        periods: Number of future periods.
        freq: Frequency string.

    Returns:
        DataFrame with 'date', 'forecast', 'lower_ci', 'upper_ci' columns.
    """
    from prophet import Prophet

    prophet_df = ts[[target_col]].reset_index()
    prophet_df.columns = ["ds", "y"]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    pred = model.predict(future)

    # Return only the forecast period
    forecast_only = pred.tail(periods)
    forecast_df = pd.DataFrame({
        "date": forecast_only["ds"].values,
        "forecast": forecast_only["yhat"].values,
        "lower_ci": forecast_only["yhat_lower"].values,
        "upper_ci": forecast_only["yhat_upper"].values,
    })

    logger.info(f"Prophet forecast {periods} periods for '{target_col}'")
    return forecast_df


def run_forecast(
    claims: pd.DataFrame,
    target_col: str = "claim_count",
    periods: int = 24,
    backend: str = "statsmodels",
    geo_col: str | None = None,
    geo_value: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end forecasting pipeline.

    Args:
        claims: Raw claims DataFrame.
        target_col: What to forecast ('claim_count', 'total_paid', 'avg_severity').
        periods: Future periods to predict.
        backend: 'statsmodels' or 'prophet'.
        geo_col: Optional geography filter column.
        geo_value: Optional geography filter value.

    Returns:
        Tuple of (historical_ts, forecast_df).
    """
    ts = prepare_claims_timeseries(claims, geo_col=geo_col, geo_value=geo_value)

    if len(ts) < 6:
        logger.error(f"Not enough data points ({len(ts)}) for forecasting")
        return ts, pd.DataFrame()

    if backend == "prophet":
        forecast_df = forecast_prophet(ts, target_col=target_col, periods=periods)
    else:
        forecast_df = forecast_statsmodels(ts, target_col=target_col, periods=periods)

    return ts, forecast_df
