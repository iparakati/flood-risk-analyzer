"""Streamlit dashboard for flood risk analysis.

Displays interactive maps, coverage gap analysis, and time-series forecasts
using data from the OpenFEMA ingestion pipeline.
"""

import sys
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.coverage_gap import cluster_risk_regions, compute_coverage_metrics
from src.forecasting.timeseries import run_forecast
from src.ingestion.openfema_client import OpenFEMAClient

st.set_page_config(
    page_title="Flood Risk Analyzer",
    page_icon="🌊",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_data(max_records: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data from OpenFEMA with caching."""
    client = OpenFEMAClient()
    claims = client.fetch_claims(max_records=max_records)
    policies = client.fetch_policies(max_records=max_records)
    disasters = client.fetch_disasters(max_records=5000)
    return claims, policies, disasters


def render_sidebar():
    """Render the sidebar controls."""
    st.sidebar.title("Flood Risk Analyzer")
    st.sidebar.markdown("Mapping NFIP coverage gaps across the U.S.")

    max_records = st.sidebar.slider(
        "Sample size (records per dataset)",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
    )

    geo_level = st.sidebar.selectbox(
        "Geographic grouping",
        ["countyCode", "reportedZipCode", "state"],
        index=0,
    )

    page = st.sidebar.radio(
        "View",
        ["Overview", "Coverage Gaps", "Forecasting", "Map"],
    )

    return max_records, geo_level, page


def render_overview(claims: pd.DataFrame, policies: pd.DataFrame, disasters: pd.DataFrame):
    """Show high-level summary stats."""
    st.header("Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Claims Loaded", f"{len(claims):,}")
    col2.metric("Policy Records", f"{len(policies):,}")
    col3.metric("Disaster Declarations", f"{len(disasters):,}")

    # Claims over time
    if "yearOfLoss" in claims.columns:
        claims_copy = claims.copy()
        claims_copy["totalPaid"] = (
            claims_copy.get("amountPaidOnBuildingClaim", pd.Series(0, index=claims_copy.index)).fillna(0)
            + claims_copy.get("amountPaidOnContentsClaim", pd.Series(0, index=claims_copy.index)).fillna(0)
        )
        yearly = claims_copy.groupby("yearOfLoss").agg(
            claim_count=("totalPaid", "count"),
            total_paid=("totalPaid", "sum"),
        ).reset_index()

        col_left, col_right = st.columns(2)
        with col_left:
            fig = px.bar(
                yearly, x="yearOfLoss", y="claim_count",
                title="Claims by Year",
                labels={"yearOfLoss": "Year", "claim_count": "Claims"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            fig = px.bar(
                yearly, x="yearOfLoss", y="total_paid",
                title="Total Payouts by Year",
                labels={"yearOfLoss": "Year", "total_paid": "Total Paid ($)"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # Top states
    if "state" in claims.columns:
        state_claims = claims.groupby("state").size().reset_index(name="count")
        state_claims = state_claims.sort_values("count", ascending=False).head(15)
        fig = px.bar(
            state_claims, x="state", y="count",
            title="Top 15 States by Claim Count",
            labels={"state": "State", "count": "Claims"},
        )
        st.plotly_chart(fig, use_container_width=True)


def render_coverage_gaps(claims: pd.DataFrame, policies: pd.DataFrame, geo_col: str):
    """Show coverage gap analysis."""
    st.header("Coverage Gap Analysis")

    st.markdown(
        "Regions where **claim frequency and severity outpace policy coverage**, "
        "indicating potential underinsurance."
    )

    try:
        metrics = compute_coverage_metrics(claims, policies, geo_col=geo_col)
    except ValueError as e:
        st.error(f"Cannot compute coverage gaps: {e}")
        return

    if metrics.empty:
        st.warning("No data available for coverage gap analysis.")
        return

    # Cluster into risk tiers
    n_clusters = st.slider("Number of risk tiers", 2, 8, 5)
    clustered = cluster_risk_regions(metrics, n_clusters=n_clusters)

    # Summary by tier
    tier_summary = clustered.groupby("risk_cluster").agg(
        count=("risk_cluster", "count"),
        avg_gap_score=("gap_score", "mean"),
        avg_claims_per_policy=("claims_per_policy", "mean"),
        total_paid=("total_paid", "sum"),
    ).reset_index()

    fig = px.bar(
        tier_summary, x="risk_cluster", y="count",
        color="avg_gap_score", color_continuous_scale="Reds",
        title="Regions by Risk Tier",
        labels={"risk_cluster": "Risk Tier", "count": "Number of Regions"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top gaps table
    st.subheader("Top Coverage Gaps")
    display_cols = [
        geo_col, "claim_count", "policy_count",
        "claims_per_policy", "avg_claim", "total_paid",
        "gap_score", "risk_cluster",
    ]
    available_cols = [c for c in display_cols if c in clustered.columns]
    st.dataframe(
        clustered[available_cols].head(25),
        use_container_width=True,
    )


def render_forecasting(claims: pd.DataFrame):
    """Show time-series forecast controls and chart."""
    st.header("Claim Forecasting")

    col1, col2, col3 = st.columns(3)
    with col1:
        target = st.selectbox("Forecast target", ["claim_count", "total_paid", "avg_severity"])
    with col2:
        periods = st.slider("Forecast periods (months)", 6, 60, 24)
    with col3:
        backend = st.selectbox("Model", ["statsmodels", "prophet"])

    # Optional geographic filter
    filter_geo = st.checkbox("Filter by state")
    geo_col, geo_value = None, None
    if filter_geo and "state" in claims.columns:
        states = sorted(claims["state"].dropna().unique())
        geo_value = st.selectbox("State", states)
        geo_col = "state"

    if st.button("Run Forecast"):
        with st.spinner("Fitting model..."):
            try:
                ts, forecast_df = run_forecast(
                    claims,
                    target_col=target,
                    periods=periods,
                    backend=backend,
                    geo_col=geo_col,
                    geo_value=geo_value,
                )
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                return

        if forecast_df.empty:
            st.warning("Not enough data for forecasting. Try a larger sample or different filter.")
            return

        # Plot historical + forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts[target],
            mode="lines", name="Historical",
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["forecast"],
            mode="lines", name="Forecast",
            line=dict(dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["upper_ci"],
            mode="lines", name="Upper CI",
            line=dict(width=0),
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["lower_ci"],
            mode="lines", name="Lower CI",
            line=dict(width=0),
            fill="tonexty", fillcolor="rgba(68,68,68,0.15)",
        ))
        fig.update_layout(
            title=f"{target} — Historical + Forecast",
            xaxis_title="Date",
            yaxis_title=target,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_map(claims: pd.DataFrame):
    """Render an interactive Folium map of claim locations."""
    st.header("Claim Locations Map")

    if "latitude" not in claims.columns or "longitude" not in claims.columns:
        st.warning("Latitude/longitude data not available in claims dataset.")
        return

    map_df = claims.dropna(subset=["latitude", "longitude"]).copy()
    map_df = map_df[(map_df["latitude"] != 0) & (map_df["longitude"] != 0)]

    if map_df.empty:
        st.warning("No geocoded claims to display.")
        return

    # Limit points for performance
    max_points = st.slider("Max map points", 100, 5000, 1000)
    if len(map_df) > max_points:
        map_df = map_df.sample(n=max_points, random_state=42)

    center_lat = map_df["latitude"].mean()
    center_lon = map_df["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="cartodbpositron")

    # Color by claim amount if available
    map_df["totalPaid"] = (
        map_df.get("amountPaidOnBuildingClaim", pd.Series(0, index=map_df.index)).fillna(0)
        + map_df.get("amountPaidOnContentsClaim", pd.Series(0, index=map_df.index)).fillna(0)
    )

    for _, row in map_df.iterrows():
        paid = row["totalPaid"]
        if paid > 100000:
            color = "red"
        elif paid > 25000:
            color = "orange"
        else:
            color = "blue"

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=(
                f"State: {row.get('state', 'N/A')}<br>"
                f"Zone: {row.get('floodZone', 'N/A')}<br>"
                f"Paid: ${paid:,.0f}"
            ),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:50px; left:50px; z-index:1000;
         background:white; padding:10px; border-radius:5px; border:1px solid grey;">
        <b>Claim Amount</b><br>
        <span style="color:red;">&#9679;</span> > $100K<br>
        <span style="color:orange;">&#9679;</span> $25K–$100K<br>
        <span style="color:blue;">&#9679;</span> < $25K
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=600)


def main():
    max_records, geo_col, page = render_sidebar()

    with st.spinner("Loading data from OpenFEMA..."):
        claims, policies, disasters = load_data(max_records=max_records)

    if page == "Overview":
        render_overview(claims, policies, disasters)
    elif page == "Coverage Gaps":
        render_coverage_gaps(claims, policies, geo_col)
    elif page == "Forecasting":
        render_forecasting(claims)
    elif page == "Map":
        render_map(claims)


if __name__ == "__main__":
    main()
