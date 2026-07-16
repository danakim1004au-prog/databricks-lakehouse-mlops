"""Data-quality tests mirroring the Silver-layer gates in notebook 02."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_churn_data import generate_dataset  # noqa: E402
from train_local import gold_features, silver_clean  # noqa: E402

def raw() -> pd.DataFrame:
    """Build test input in memory so missing generated files cannot hide failures."""
    return generate_dataset()


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(generate_dataset(), generate_dataset())
    assert len(generate_dataset(n_rows=100)) == 101


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
