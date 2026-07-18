"""Pandas implementation of the production data contract for local CI."""

from __future__ import annotations

import pandas as pd

from .contracts import (
    CONTRACT_MAP,
    FEATURES,
    LINEAGE_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    VALID_CHURN_VALUES,
    VALID_CONTRACTS,
)


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_RAW_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Raw schema is missing required columns: {', '.join(missing)}")


def _reject_conflicting_customer_rows(frame: pd.DataFrame) -> None:
    business_columns = sorted(REQUIRED_RAW_COLUMNS - {"customerID"})
    conflicts = (
        frame.groupby("customerID", dropna=False)[business_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        sample = ", ".join(conflicts[conflicts].index.astype(str)[:5])
        raise ValueError(f"Conflicting duplicate customer records: {sample}")


def silver_clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate, deterministically deduplicate, type, and label raw churn rows."""
    _require_columns(raw)
    if raw["customerID"].isna().any() or raw["customerID"].astype(str).str.strip().eq("").any():
        raise ValueError("customerID must be non-null and non-blank")

    frame = raw.copy()
    frame["Contract"] = frame["Contract"].astype(str).str.strip().str.upper().map(CONTRACT_MAP)
    if frame["Contract"].isna().any():
        raise ValueError("Contract contains values outside the supported domain")

    churn = frame["Churn"].astype(str).str.strip().str.title()
    invalid_churn = sorted(set(churn) - VALID_CHURN_VALUES)
    if invalid_churn:
        raise ValueError(f"Churn contains invalid values: {invalid_churn}")
    frame["Churn"] = churn

    frame["tenure"] = pd.to_numeric(frame["tenure"], errors="raise")
    frame["SeniorCitizen"] = pd.to_numeric(frame["SeniorCitizen"], errors="raise")
    frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="raise")
    total = frame["TotalCharges"].astype(str).str.strip()
    blank_total = total.eq("")
    if (blank_total & frame["tenure"].ne(0)).any():
        raise ValueError("Blank TotalCharges is only valid when tenure is zero")
    frame["TotalCharges"] = pd.to_numeric(total.mask(blank_total, "0"), errors="raise")

    if not frame["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError("SeniorCitizen must be 0 or 1")
    if (frame[["tenure", "MonthlyCharges", "TotalCharges"]] < 0).any().any():
        raise ValueError("tenure and charge values must be non-negative")
    if not set(frame["Contract"].unique()) <= VALID_CONTRACTS:
        raise ValueError("Contract normalisation failed")

    _reject_conflicting_customer_rows(frame)
    frame = frame.sort_values("customerID", kind="stable").drop_duplicates("customerID", keep="last")
    frame["churn_label"] = frame["Churn"].eq("Yes").astype(int)
    return frame.drop(columns=["Churn"]).reset_index(drop=True)


def gold_features(silver: pd.DataFrame) -> pd.DataFrame:
    """Create the model feature contract while preserving batch lineage."""
    frame = silver.copy()
    frame["avg_monthly_spend"] = (
        frame["TotalCharges"] / frame["tenure"].clip(lower=1)
    ).round(2)
    frame["spend_delta"] = (frame["MonthlyCharges"] - frame["avg_monthly_spend"]).round(2)
    frame["is_month_to_month"] = frame["Contract"].eq("Month-to-month").astype(int)
    frame["is_fiber"] = frame["InternetService"].eq("Fiber optic").astype(int)
    frame["is_echeck"] = frame["PaymentMethod"].eq("Electronic check").astype(int)
    frame["has_tech_support"] = frame["TechSupport"].eq("Yes").astype(int)
    frame["is_paperless"] = frame["PaperlessBilling"].eq("Yes").astype(int)
    frame["tenure_bucket"] = pd.cut(
        frame["tenure"],
        bins=[-1, 6, 24, 48, float("inf")],
        labels=["0-6m", "7-24m", "25-48m", "49m+"],
    ).astype(str)

    lineage = [column for column in LINEAGE_COLUMNS if column in frame.columns]
    output = frame[["customerID", "tenure_bucket", *FEATURES, "churn_label", *lineage]].copy()
    if output[FEATURES + ["churn_label"]].isna().any().any():
        raise ValueError("Gold feature contract contains nulls")
    return output
