"""Tests for the coverage gap analysis module."""

import pandas as pd
import pytest

from src.analysis.coverage_gap import (
    cluster_risk_regions,
    compute_coverage_metrics,
    find_top_gaps,
)


@pytest.fixture
def sample_claims():
    return pd.DataFrame({
        "countyCode": ["12001", "12001", "12003", "12003", "12003", "48201", "48201"],
        "state": ["FL", "FL", "FL", "FL", "FL", "TX", "TX"],
        "reportedZipCode": ["33101", "33101", "33301", "33301", "33301", "77001", "77001"],
        "amountPaidOnBuildingClaim": [50000, 75000, 30000, 120000, 45000, 200000, 80000],
        "amountPaidOnContentsClaim": [10000, 15000, 5000, 20000, 8000, 40000, 15000],
    })


@pytest.fixture
def sample_policies():
    return pd.DataFrame({
        "countyCode": ["12001", "12001", "12001", "12003", "48201", "48201", "48201", "48201"],
        "propertyState": ["FL", "FL", "FL", "FL", "TX", "TX", "TX", "TX"],
        "reportedZipCode": ["33101", "33101", "33101", "33301", "77001", "77001", "77001", "77001"],
        "policyCount": [1, 1, 1, 1, 1, 1, 1, 1],
        "totalBuildingInsuranceCoverage": [250000, 250000, 250000, 250000, 250000, 250000, 250000, 250000],
        "totalContentsInsuranceCoverage": [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000],
    })


def test_compute_coverage_metrics(sample_claims, sample_policies):
    metrics = compute_coverage_metrics(sample_claims, sample_policies, geo_col="countyCode")
    assert len(metrics) == 3
    assert "claims_per_policy" in metrics.columns
    assert "gap_score" in metrics.columns
    assert "total_paid" in metrics.columns


def test_compute_coverage_metrics_by_zip(sample_claims, sample_policies):
    metrics = compute_coverage_metrics(sample_claims, sample_policies, geo_col="reportedZipCode")
    assert len(metrics) == 3
    assert "claims_per_policy" in metrics.columns


def test_compute_coverage_metrics_invalid_col(sample_claims, sample_policies):
    with pytest.raises(ValueError, match="not found"):
        compute_coverage_metrics(sample_claims, sample_policies, geo_col="nonexistent")


def test_cluster_risk_regions(sample_claims, sample_policies):
    metrics = compute_coverage_metrics(sample_claims, sample_policies)
    clustered = cluster_risk_regions(metrics, n_clusters=2)
    assert "risk_cluster" in clustered.columns
    assert set(clustered["risk_cluster"].unique()).issubset({0, 1})


def test_find_top_gaps(sample_claims, sample_policies):
    top = find_top_gaps(sample_claims, sample_policies, top_n=2)
    assert len(top) == 2
    assert "risk_cluster" in top.columns
    assert "gap_score" in top.columns
