"""Spark transformations used directly by the Databricks notebooks."""

from __future__ import annotations

from .contracts import (
    CONTRACT_MAP,
    FEATURES,
    LINEAGE_COLUMNS,
    REQUIRED_RAW_COLUMNS,
)


def silver_clean_spark(raw):
    """Apply the same strict Silver contract as the local Pandas implementation."""
    from pyspark.sql import functions as F

    missing = sorted(REQUIRED_RAW_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError(f"Raw schema is missing required columns: {', '.join(missing)}")

    contract_map = F.create_map([F.lit(value) for item in CONTRACT_MAP.items() for value in item])
    frame = (
        raw.withColumn("Contract", contract_map[F.upper(F.trim(F.col("Contract")))])
        .withColumn("Churn", F.initcap(F.trim(F.col("Churn"))))
        .withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int"))
        .withColumn("tenure", F.col("tenure").cast("int"))
        .withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double"))
        .withColumn(
            "TotalCharges",
            F.when(F.trim(F.col("TotalCharges")) == "", F.lit("0"))
            .otherwise(F.col("TotalCharges"))
            .cast("double"),
        )
    )

    invalid = frame.filter(
        F.col("customerID").isNull()
        | (F.trim(F.col("customerID")) == "")
        | F.col("Contract").isNull()
        | ~F.col("Churn").isin("Yes", "No")
        | ~F.col("SeniorCitizen").isin(0, 1)
        | F.col("tenure").isNull()
        | F.col("MonthlyCharges").isNull()
        | F.col("TotalCharges").isNull()
        | (F.col("tenure") < 0)
        | (F.col("MonthlyCharges") < 0)
        | (F.col("TotalCharges") < 0)
    )
    if invalid.limit(1).count():
        raise ValueError("Silver contract failed: null, domain, cast, or range violation")

    blank_nonzero = raw.filter((F.trim(F.col("TotalCharges")) == "") & (F.col("tenure").cast("int") != 0))
    if blank_nonzero.limit(1).count():
        raise ValueError("Blank TotalCharges is only valid when tenure is zero")

    payload = F.to_json(F.struct(*[F.col(column) for column in sorted(REQUIRED_RAW_COLUMNS - {"customerID"})]))
    conflicts = (
        frame.withColumn("_payload", payload)
        .groupBy("customerID")
        .agg(F.countDistinct("_payload").alias("payload_count"))
        .filter(F.col("payload_count") > 1)
    )
    if conflicts.limit(1).count():
        raise ValueError("Conflicting duplicate customer records detected")

    return (
        frame.dropDuplicates(["customerID"])
        .withColumn("churn_label", F.when(F.col("Churn") == "Yes", 1).otherwise(0))
        .drop("Churn")
    )


def gold_features_spark(silver):
    """Create the authoritative Spark Gold feature contract."""
    from pyspark.sql import functions as F

    gold = (
        silver.withColumn(
            "tenure_bucket",
            F.when(F.col("tenure") <= 6, "0-6m")
            .when(F.col("tenure") <= 24, "7-24m")
            .when(F.col("tenure") <= 48, "25-48m")
            .otherwise("49m+"),
        )
        .withColumn(
            "avg_monthly_spend",
            F.round(F.col("TotalCharges") / F.greatest(F.col("tenure"), F.lit(1)), 2),
        )
        .withColumn("spend_delta", F.round(F.col("MonthlyCharges") - F.col("avg_monthly_spend"), 2))
        .withColumn("is_month_to_month", (F.col("Contract") == "Month-to-month").cast("int"))
        .withColumn("is_fiber", (F.col("InternetService") == "Fiber optic").cast("int"))
        .withColumn("is_echeck", (F.col("PaymentMethod") == "Electronic check").cast("int"))
        .withColumn("has_tech_support", (F.col("TechSupport") == "Yes").cast("int"))
        .withColumn("is_paperless", (F.col("PaperlessBilling") == "Yes").cast("int"))
    )
    lineage = [column for column in LINEAGE_COLUMNS if column in gold.columns]
    output = gold.select("customerID", "tenure_bucket", *FEATURES, "churn_label", *lineage)
    if output.filter(F.greatest(*[F.col(column).isNull().cast("int") for column in FEATURES]) == 1).limit(1).count():
        raise ValueError("Gold feature contract contains nulls")
    return output
