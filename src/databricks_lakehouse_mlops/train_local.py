"""Spark-free local training smoke test used by CI."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import FEATURES
from .pandas_transforms import gold_features, silver_clean

DATA = Path(__file__).parents[2] / "data" / "telco_churn_train.csv"


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}. Run 'python data/generate_churn_data.py' first.")

    gold = gold_features(silver_clean(pd.read_csv(DATA)))
    development, test = train_test_split(
        gold, test_size=0.2, random_state=42, stratify=gold["churn_label"]
    )
    train, validation = train_test_split(
        development,
        test_size=0.25,
        random_state=42,
        stratify=development["churn_label"],
    )

    candidates = [
        (
            "logreg_baseline",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
                ]
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
            ),
        ),
    ]
    trained = []
    for name, model in candidates:
        model.fit(train[FEATURES], train["churn_label"])
        validation_probability = model.predict_proba(validation[FEATURES])[:, 1]
        trained.append((roc_auc_score(validation["churn_label"], validation_probability), name, model))

    _, name, winner = max(trained, key=lambda item: item[0])
    probability = winner.predict_proba(test[FEATURES])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(test["churn_label"], probability),
        "average_precision": average_precision_score(test["churn_label"], probability),
        "f1": f1_score(test["churn_label"], prediction),
    }
    print(f"winner={name} " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    if metrics["roc_auc"] <= 0.70:
        raise RuntimeError("Winner ROC-AUC is below the smoke-test floor")


if __name__ == "__main__":
    main()
