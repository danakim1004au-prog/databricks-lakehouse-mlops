"""Local sklearn mirror of notebooks 02-04 (clean -> features -> train).

Runs without Spark or Azure — used as a CI smoke test to keep the
feature/label contract honest before paying for cluster time.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent.parent / "data" / "telco_churn.csv"

FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "spend_delta", "is_month_to_month", "is_fiber",
    "is_echeck", "has_tech_support", "is_paperless",
]

CONTRACT_MAP = {"MONTH-TO-MONTH": "Month-to-month", "ONE YEAR": "One year", "TWO YEAR": "Two year"}


def silver_clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates(subset=["customerID"]).copy()
    df["Contract"] = df["Contract"].replace(CONTRACT_MAP)
    df["TotalCharges"] = (
        df["TotalCharges"].astype(str).str.strip().replace("", "0").astype(float)
    )
    df["churn_label"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["Churn"])


def gold_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["avg_monthly_spend"] = (out["TotalCharges"] / out["tenure"].clip(lower=1)).round(2)
    out["spend_delta"] = (out["MonthlyCharges"] - out["avg_monthly_spend"]).round(2)
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    out["is_fiber"] = (out["InternetService"] == "Fiber optic").astype(int)
    out["is_echeck"] = (out["PaymentMethod"] == "Electronic check").astype(int)
    out["has_tech_support"] = (out["TechSupport"] == "Yes").astype(int)
    out["is_paperless"] = (out["PaperlessBilling"] == "Yes").astype(int)
    return out[FEATURES + ["churn_label"]]


def main() -> None:
    gold = gold_features(silver_clean(pd.read_csv(DATA)))
    X_train, X_test, y_train, y_test = train_test_split(
        gold[FEATURES], gold["churn_label"],
        test_size=0.2, random_state=42, stratify=gold["churn_label"],
    )

    for name, model in [
        ("logreg_baseline", LogisticRegression(C=1.0, max_iter=1000)),
        ("gradient_boosting", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)),
    ]:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, (proba >= 0.5).astype(int))
        print(f"{name:18s} AUC={auc:.4f}  F1={f1:.4f}")
        assert auc > 0.70, f"{name} AUC below sanity floor"

    print("Local smoke test passed.")


if __name__ == "__main__":
    main()
