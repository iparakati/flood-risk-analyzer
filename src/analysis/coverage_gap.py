"""Coverage gap model — identifies areas where claims outpace policy coverage.

Compares NFIP claim frequency/severity against active policy counts at the
county and ZIP code level to find underinsured regions.
"""

import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_coverage_metrics(
    claims: pd.DataFrame,
    policies: pd.DataFrame,
    geo_col: str = "countyCode",
) -> pd.DataFrame:
    """Compute per-geography coverage gap metrics.

    Args:
        claims: DataFrame of NFIP claims (from OpenFEMA).
        policies: DataFrame of NFIP policies.
        geo_col: Column to group by ('countyCode', 'reportedZipCode', 'censusTract').

    Returns:
        DataFrame with one row per geography containing gap metrics.
    """
    if geo_col not in claims.columns or geo_col not in policies.columns:
        raise ValueError(f"Column '{geo_col}' not found in both claims and policies DataFrames")

    # --- Claims aggregation ---
    claims = claims.copy()
    claims["totalPaid"] = (
        claims.get("amountPaidOnBuildingClaim", pd.Series(0, index=claims.index)).fillna(0)
        + claims.get("amountPaidOnContentsClaim", pd.Series(0, index=claims.index)).fillna(0)
    )

    claims_agg = claims.groupby(geo_col).agg(
        claim_count=(geo_col, "count"),
        total_paid=("totalPaid", "sum"),
        avg_claim=("totalPaid", "mean"),
        max_claim=("totalPaid", "max"),
    ).reset_index()

    # --- Policies aggregation ---
    if "policyCount" in policies.columns:
        policy_agg = policies.groupby(geo_col).agg(
            policy_count=("policyCount", "sum"),
        ).reset_index()
    else:
        policy_agg = policies.groupby(geo_col).size().reset_index(name="policy_count")

    # Coverage amounts
    cov_cols = []
    if "totalBuildingInsuranceCoverage" in policies.columns:
        cov_cols.append("totalBuildingInsuranceCoverage")
    if "totalContentsInsuranceCoverage" in policies.columns:
        cov_cols.append("totalContentsInsuranceCoverage")

    if cov_cols:
        cov_agg = policies.groupby(geo_col)[cov_cols].sum().reset_index()
        policy_agg = policy_agg.merge(cov_agg, on=geo_col, how="left")

    # --- Merge ---
    merged = claims_agg.merge(policy_agg, on=geo_col, how="outer").fillna(0)

    # --- Gap metrics ---
    merged["claims_per_policy"] = merged["claim_count"] / merged["policy_count"].replace(0, 1)
    merged["avg_loss_ratio"] = merged["total_paid"] / merged.get(
        "totalBuildingInsuranceCoverage", pd.Series(1, index=merged.index)
    ).replace(0, 1)

    # Gap score: composite of claims-to-policy ratio and average claim severity
    # Higher = more underinsured
    if len(merged) > 0:
        scaler = StandardScaler()
        score_cols = ["claims_per_policy", "avg_claim"]
        valid = merged[score_cols].replace([float("inf"), float("-inf")], 0).fillna(0)
        if len(valid) > 1:
            scaled = scaler.fit_transform(valid)
            merged["gap_score"] = scaled.mean(axis=1)
        else:
            merged["gap_score"] = 0.0

    merged = merged.sort_values("gap_score", ascending=False)
    logger.info(
        f"Computed coverage metrics for {len(merged)} geographies "
        f"(grouping by {geo_col})"
    )
    return merged


def cluster_risk_regions(
    coverage_df: pd.DataFrame,
    n_clusters: int = 5,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Cluster geographies into risk tiers using KMeans.

    Args:
        coverage_df: Output from compute_coverage_metrics().
        n_clusters: Number of risk clusters.
        features: Columns to use for clustering. Defaults to gap-related metrics.

    Returns:
        Input DataFrame with an added 'risk_cluster' column (0=lowest, n-1=highest).
    """
    if features is None:
        features = ["claims_per_policy", "avg_claim", "claim_count"]

    available = [f for f in features if f in coverage_df.columns]
    if not available:
        raise ValueError(f"None of the requested features {features} found in DataFrame")

    df = coverage_df.copy()
    X = df[available].replace([float("inf"), float("-inf")], 0).fillna(0)

    n_clusters = min(n_clusters, len(X))
    if n_clusters < 2:
        df["risk_cluster"] = 0
        return df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["risk_cluster"] = kmeans.fit_predict(X_scaled)

    # Reorder clusters so higher number = higher risk
    cluster_means = df.groupby("risk_cluster")["gap_score"].mean().sort_values()
    rank_map = {cluster: rank for rank, cluster in enumerate(cluster_means.index)}
    df["risk_cluster"] = df["risk_cluster"].map(rank_map)

    logger.info(f"Clustered {len(df)} geographies into {n_clusters} risk tiers")
    return df


def find_top_gaps(
    claims: pd.DataFrame,
    policies: pd.DataFrame,
    geo_col: str = "countyCode",
    top_n: int = 20,
) -> pd.DataFrame:
    """End-to-end: compute metrics, cluster, and return the top coverage gaps.

    Args:
        claims: Raw claims DataFrame.
        policies: Raw policies DataFrame.
        geo_col: Geography grouping column.
        top_n: Number of top gap regions to return.

    Returns:
        Top N underinsured geographies with metrics and risk tier.
    """
    metrics = compute_coverage_metrics(claims, policies, geo_col=geo_col)
    clustered = cluster_risk_regions(metrics)
    return clustered.head(top_n)
