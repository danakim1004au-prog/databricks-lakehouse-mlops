"""Generate a seeded synthetic telco churn dataset.

Mimics the well-known Telco Customer Churn schema so the medallion pipeline
exercises realistic messiness: duplicate rows, blank TotalCharges for new
customers, and mixed-case categoricals for Silver to clean up.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 10_000
TRAIN_OUT = Path(__file__).parent / "telco_churn_train.csv"
SCORING_OUT = Path(__file__).parent / "telco_churn_scoring.csv"


def generate_dataset(
    n_rows: int = N_ROWS,
    seed: int = SEED,
    *,
    shift_strength: float = 0.0,
) -> pd.DataFrame:
    """Return a deterministic synthetic batch with optional population shift."""
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100 so the duplicate sample is meaningful")
    if not 0.0 <= shift_strength <= 1.0:
        raise ValueError("shift_strength must be between zero and one")

    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, n_rows)
    monthly = np.round(rng.uniform(18.0 + 8.0 * shift_strength, 120.0, n_rows), 2)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n_rows,
        p=[0.55 + 0.08 * shift_strength, 0.21, 0.24 - 0.08 * shift_strength],
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_rows, p=[0.34, 0.44, 0.22])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )
    support = rng.choice(["Yes", "No"], n_rows, p=[0.29, 0.71])
    paperless = rng.choice(["Yes", "No"], n_rows, p=[0.59, 0.41])
    senior = rng.choice([0, 1], n_rows, p=[0.84, 0.16])

    # Churn probability driven by known telco signals
    logit = (
        -1.2
        - 0.035 * tenure
        + 0.012 * monthly
        + np.where(contract == "Month-to-month", 1.1, 0.0)
        + np.where(contract == "Two year", -1.3, 0.0)
        + np.where(internet == "Fiber optic", 0.55, 0.0)
        + np.where(payment == "Electronic check", 0.45, 0.0)
        + np.where(support == "Yes", -0.6, 0.0)
        + 0.35 * senior
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n_rows) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customerID": [f"C{100000 + i}" for i in range(n_rows)],
            "SeniorCitizen": senior,
            "tenure": tenure,
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
            "TechSupport": support,
            "PaperlessBilling": paperless,
            "MonthlyCharges": monthly,
            "TotalCharges": np.round(
                np.maximum(
                    monthly * np.maximum(tenure, 0) + rng.normal(0, 12, n_rows),
                    0,
                ),
                2,
            ),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )

    # Inject realistic dirt for the Silver layer to handle:
    # 1) blank TotalCharges for brand-new customers (classic telco quirk).
    #    pandas 3.x forbids assigning str into a float64 column, so cast first.
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    # 2) ~1% exact duplicate rows
    dupes = df.sample(n=n_rows // 100, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    # 3) mixed-case contract values in ~2% of rows
    idx = df.sample(frac=0.02, random_state=seed).index
    df.loc[idx, "Contract"] = df.loc[idx, "Contract"].str.upper()

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    training = generate_dataset(seed=SEED)
    scoring = generate_dataset(n_rows=2_500, seed=SEED + 1, shift_strength=0.5)
    training.to_csv(TRAIN_OUT, index=False)
    scoring.to_csv(SCORING_OUT, index=False)
    print(
        f"Wrote training batch: {len(training):,} rows "
        f"({training['Churn'].eq('Yes').mean():.1%} churn) -> {TRAIN_OUT}"
    )
    print(
        f"Wrote scoring batch: {len(scoring):,} rows "
        f"({scoring['Churn'].eq('Yes').mean():.1%} churn) -> {SCORING_OUT}"
    )


if __name__ == "__main__":
    main()
