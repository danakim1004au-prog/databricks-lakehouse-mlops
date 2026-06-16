"""Generate a seeded synthetic telco churn dataset (10,000 rows).

Mimics the well-known Telco Customer Churn schema so the medallion pipeline
exercises realistic messiness: duplicate rows, blank TotalCharges for new
customers, and mixed-case categoricals — exactly what Silver must clean up.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 10_000
OUT = Path(__file__).parent / "telco_churn.csv"


def main() -> None:
    rng = np.random.default_rng(SEED)

    tenure = rng.integers(0, 73, N_ROWS)
    monthly = np.round(rng.uniform(18.0, 120.0, N_ROWS), 2)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], N_ROWS, p=[0.55, 0.21, 0.24]
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], N_ROWS, p=[0.34, 0.44, 0.22])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        N_ROWS,
        p=[0.34, 0.23, 0.22, 0.21],
    )
    support = rng.choice(["Yes", "No"], N_ROWS, p=[0.29, 0.71])
    paperless = rng.choice(["Yes", "No"], N_ROWS, p=[0.59, 0.41])
    senior = rng.choice([0, 1], N_ROWS, p=[0.84, 0.16])

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
    churn = (rng.uniform(0, 1, N_ROWS) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customerID": [f"C{100000 + i}" for i in range(N_ROWS)],
            "SeniorCitizen": senior,
            "tenure": tenure,
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
            "TechSupport": support,
            "PaperlessBilling": paperless,
            "MonthlyCharges": monthly,
            "TotalCharges": np.round(monthly * np.maximum(tenure, 0)
                                     + rng.normal(0, 12, N_ROWS), 2),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )

    # Inject realistic dirt for the Silver layer to handle:
    # 1) blank TotalCharges for brand-new customers (classic telco quirk).
    #    pandas 3.x forbids assigning str into a float64 column, so cast first.
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    # 2) ~1% exact duplicate rows
    dupes = df.sample(n=N_ROWS // 100, random_state=SEED)
    df = pd.concat([df, dupes], ignore_index=True)
    # 3) mixed-case contract values in ~2% of rows
    idx = df.sample(frac=0.02, random_state=SEED).index
    df.loc[idx, "Contract"] = df.loc[idx, "Contract"].str.upper()

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows ({df['Churn'].eq('Yes').mean():.1%} churn) -> {OUT}")


if __name__ == "__main__":
    main()
