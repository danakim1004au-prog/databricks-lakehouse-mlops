"""Exercise the exact Spark transforms called by the Databricks notebooks."""

import pytest

pyspark = pytest.importorskip("pyspark")

from databricks_lakehouse_mlops.spark_transforms import (  # noqa: E402
    gold_features_spark,
    silver_clean_spark,
)


@pytest.fixture(scope="session")
def spark():
    session = (
        pyspark.sql.SparkSession.builder.master("local[2]")
        .appName("churn-transform-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def raw_rows():
    base = {
        "customerID": "C1",
        "SeniorCitizen": "0",
        "tenure": "0",
        "Contract": "MONTH-TO-MONTH",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "PaperlessBilling": "Yes",
        "MonthlyCharges": "80.0",
        "TotalCharges": " ",
        "Churn": "Yes",
        "_batch_id": "batch-1",
        "_job_run_id": "run-1",
        "_source_file": "raw.csv",
        "_ingest_ts": "2026-01-01T00:00:00Z",
    }
    second = dict(base, customerID="C2", tenure="12", TotalCharges="900", Churn="No")
    return [base, second]


def test_spark_silver_and_gold_contract(spark):
    raw = spark.createDataFrame(raw_rows())
    silver = silver_clean_spark(raw)
    gold = gold_features_spark(silver)
    rows = {row.customerID: row for row in gold.collect()}

    assert rows["C1"].TotalCharges == 0.0
    assert rows["C1"].is_month_to_month == 1
    assert rows["C1"].churn_label == 1
    assert rows["C1"]._batch_id == "batch-1"


def test_spark_transform_rejects_invalid_label(spark):
    frame = spark.createDataFrame([dict(raw_rows()[0], Churn="Maybe")])
    with pytest.raises(ValueError, match="Silver contract failed"):
        silver_clean_spark(frame)
