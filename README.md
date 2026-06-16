# Databricks Lakehouse MLOps — Customer Churn Prediction

End-to-end **Azure Databricks + Azure ML** portfolio project implementing the patterns most
requested in 2026 data engineering / ML engineering job descriptions:

- **Medallion architecture** (Bronze → Silver → Gold) on **Delta Lake**
- **PySpark** ETL with data-quality gates
- **MLflow** experiment tracking, model registry, and batch inference
- **Unity Catalog**-ready storage layout (ADLS Gen2 + Access Connector)
- **Infrastructure as Code** (Bicep) with same-day deploy → capture → teardown workflow
- Cost-conscious single-node cluster design (~AUD 5–10 per lab session)

## Architecture

```
                       Azure Databricks (Premium)
┌──────────┐   ┌────────────────────────────────────────────┐   ┌─────────────┐
│ Synthetic │   │  Bronze        Silver         Gold         │   │   MLflow    │
│ telco CSV ├──▶│  raw ingest ─▶ clean/dedupe ─▶ features ───┼──▶│ train/track │
│ (ADLS g2) │   │  Delta         Delta + DQ     Delta        │   │  registry   │
└──────────┘   └────────────────────────────────────────────┘   └──────┬──────┘
                                                                       │
                                                          batch inference ▶ Gold
```

## Repo layout

| Path | Purpose |
|---|---|
| `infra/main.bicep` | ADLS Gen2 + Databricks workspace (Premium) + UC Access Connector |
| `infra/deploy.sh` | One-command deploy (resource group `rg-dbx-churn-lab-aue`) |
| `infra/teardown.sh` | One-command full teardown |
| `data/generate_churn_data.py` | Synthetic telco churn dataset generator (10,000 rows, seeded) |
| `notebooks/01_bronze_ingest.py` | CSV → Bronze Delta (schema-on-read, ingest metadata columns) |
| `notebooks/02_silver_clean.py` | Dedupe, type casting, null handling, DQ assertions |
| `notebooks/03_gold_features.py` | Feature engineering (tenure buckets, spend ratios, encodings) |
| `notebooks/04_train_mlflow.py` | Gradient boosting + logistic baseline, MLflow tracking + registry |
| `notebooks/05_batch_inference.py` | Load registered model, score Gold table, write predictions |
| `src/train_local.py` | Local sklearn mirror of notebook 04 — CI-friendly smoke test |
| `tests/test_data_quality.py` | pytest data-quality checks on the generated dataset |
| `docs/SCREENSHOT_CHECKLIST.md` | Evidence-capture checklist for the live lab run |
| `docs/COST_ESTIMATE.md` | Per-session cost breakdown |

## Quick start

```bash
# 1. Generate the dataset locally
python3 data/generate_churn_data.py            # writes data/telco_churn.csv

# 2. Run the local training smoke test (no Azure required)
python3 src/train_local.py

# 3. Deploy Azure resources
./infra/deploy.sh

# 4. In the Databricks workspace: upload data/telco_churn.csv to the raw container,
#    import notebooks/, attach a single-node cluster, run 01 → 05 in order.

# 5. Tear down everything
./infra/teardown.sh
```

## Why these choices (interview talking points)

- **Delta over plain Parquet**: ACID merge for dedupe in Silver, time travel for audit,
  `OPTIMIZE`/`VACUUM` story for cost control.
- **MLflow registry over ad-hoc pickle files**: model lineage, stage transitions
  (None → Staging → Production), reproducible runs with logged params/metrics/artifacts.
- **Single-node cluster**: at 10k rows a multi-node cluster is waste; demonstrates
  cost-awareness employers explicitly screen for.
- **Local sklearn mirror (`src/train_local.py`)**: the same feature/label contract runs in CI
  without a Spark cluster — cheap regression safety net.
