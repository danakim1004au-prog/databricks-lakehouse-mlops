"""Data-quality tests mirroring the Silver-layer gates in notebook 02."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from train_local import gold_features, silver_clean  # noqa: E402

DATA = ROOT / "data" / "telco_churn.csv"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    if not DATA.exists():
        pytest.skip("run data/generate_churn_data.py first")
    return pd.read_csv(DATA)


def test_raw_contains_injected_dirt(raw):
    # The generator must produce the messiness Silver claims to fix
    assert raw["customerID"].duplicated().any(), "expected duplicate rows"
    assert raw["Contract"].str.isupper().any(), "expected mixed-case contracts"
    assert (raw["TotalCharges"].astype(str).str.strip() == "").any(), "expected blank TotalCharges"


def test_silver_gates(raw):
    silver = silver_clean(raw)
    assert len(silver) > 9_000
    assert silver["customerID"].is_unique
    assert silver["TotalCharges"].notna().all()
    assert set(silver["Contract"].unique()) <= {"Month-to-month", "One year", "Two year"}
    assert set(silver["churn_label"].unique()) <= {0, 1}


def test_gold_feature_contract(raw):
    gold = gold_features(silver_clean(raw))
    assert gold.notna().all().all(), "Gold must contain no nulls"
    binary_cols = ["is_month_to_month", "is_fiber", "is_echeck", "has_tech_support", "is_paperless"]
    for c in binary_cols:
        assert set(gold[c].unique()) <= {0, 1}
    # churn rate sanity window
    assert 0.10 < gold["churn_label"].mean() < 0.60
