"""Data-quality tests for the shared production transform contract."""

import pandas as pd
import pytest

from databricks_lakehouse_mlops import gold_features, silver_clean
from data.generate_churn_data import generate_dataset

def raw() -> pd.DataFrame:
    """Build test input in memory so missing generated files cannot hide failures."""
    return generate_dataset()


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(generate_dataset(), generate_dataset())
    assert len(generate_dataset(n_rows=100)) == 101
    assert not generate_dataset(seed=42).equals(generate_dataset(seed=43))


def test_generator_rejects_invalid_shift():
    with pytest.raises(ValueError, match="shift_strength"):
        generate_dataset(shift_strength=1.1)


def test_raw_contains_injected_dirt():
    raw_df = raw()
    # The generator must produce the messiness Silver claims to fix
    assert raw_df["customerID"].duplicated().any(), "expected duplicate rows"
    assert raw_df["Contract"].str.isupper().any(), "expected mixed-case contracts"
    assert (raw_df["TotalCharges"].astype(str).str.strip() == "").any(), "expected blank TotalCharges"


def test_silver_gates():
    silver = silver_clean(raw())
    assert len(silver) > 9_000
    assert silver["customerID"].is_unique
    assert silver["TotalCharges"].notna().all()
    assert set(silver["Contract"].unique()) <= {"Month-to-month", "One year", "Two year"}
    assert set(silver["churn_label"].unique()) <= {0, 1}


def test_gold_feature_contract():
    gold = gold_features(silver_clean(raw()))
    assert gold.notna().all().all(), "Gold must contain no nulls"
    binary_cols = ["is_month_to_month", "is_fiber", "is_echeck", "has_tech_support", "is_paperless"]
    for c in binary_cols:
        assert set(gold[c].unique()) <= {0, 1}
    # churn rate sanity window
    assert 0.10 < gold["churn_label"].mean() < 0.60


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Churn", "Maybe", "invalid values"),
        ("Contract", "Forever", "supported domain"),
        ("tenure", -1, "non-negative"),
        ("SeniorCitizen", 3, "must be 0 or 1"),
    ],
)
def test_silver_rejects_domain_and_range_violations(column, value, message):
    frame = raw()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        silver_clean(frame)


def test_silver_rejects_missing_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        silver_clean(raw().drop(columns=["Churn"]))


def test_silver_rejects_conflicting_duplicate_customer_rows():
    frame = raw()
    duplicate_id = frame.loc[frame["customerID"].duplicated(keep=False), "customerID"].iloc[0]
    matches = frame.index[frame["customerID"] == duplicate_id]
    frame.loc[matches[-1], "MonthlyCharges"] = 999
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        silver_clean(frame)
