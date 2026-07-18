"""Authoritative feature and data-contract definitions."""

FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "spend_delta",
    "is_month_to_month",
    "is_fiber",
    "is_echeck",
    "has_tech_support",
    "is_paperless",
]

REQUIRED_RAW_COLUMNS = {
    "customerID",
    "SeniorCitizen",
    "tenure",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}

VALID_CONTRACTS = {"Month-to-month", "One year", "Two year"}
CONTRACT_MAP = {value.upper(): value for value in VALID_CONTRACTS}
VALID_CHURN_VALUES = {"Yes", "No"}
LINEAGE_COLUMNS = ["_batch_id", "_job_run_id", "_source_file", "_ingest_ts"]
