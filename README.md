# Databricks Lakehouse MLOps — Customer Churn Prediction

A small, reproducible **Azure Databricks + MLflow** lab for customer-churn prediction. It demonstrates a Delta Lake medallion pipeline, data-quality checks, model comparison, MLflow registration, and batch scoring on a cost-conscious single-node cluster.

This is intentionally a Phase 1 portfolio lab. It uses path-based Delta tables and a Databricks secret scope so the run can be created and torn down in one session. Production hardening is out of scope for this version.

## Architecture

```text
Synthetic telco CSV
  -> ADLS Gen2 raw container
  -> Bronze Delta
       Raw CSV loaded as strings with ingest metadata
  -> Silver Delta
       Deduplicated, typed, null-checked, and quality-gated
  -> Gold Delta
       ML-ready churn features and labels
  -> MLflow training
       Logistic baseline vs gradient boosting, metrics and artefacts logged
  -> Unity Catalog model registry
       Winning model registered as churn_classifier@staging
  -> Batch inference
       Gold features scored and predictions written back to Delta
```

## Repository layout

| Path | Purpose |
|---|---|
| `infra/main.bicep` | ADLS Gen2, Premium Databricks workspace, and Access Connector |
| `infra/deploy.sh` | Deploy the disposable lab resource group |
| `infra/teardown.sh` | Request full resource-group deletion |
| `data/generate_churn_data.py` | Seeded synthetic telco dataset generator |
| `notebooks/01_bronze_ingest.py` | CSV → Bronze Delta with ingest metadata |
| `notebooks/02_silver_clean.py` | Dedupe, type casting, null handling, and DQ gates |
| `notebooks/03_gold_features.py` | Feature engineering for training and scoring |
| `notebooks/04_train_mlflow.py` | Logistic baseline vs gradient boosting with MLflow |
| `notebooks/05_batch_inference.py` | Load the `@staging` model and write predictions |
| `src/train_local.py` | Spark-free local mirror used by CI |
| `tests/test_data_quality.py` | Data-quality and feature-contract tests |
| `docs/images/*.png` | Evidence screenshots from a real Azure Databricks run |
| `docs/COST_ESTIMATE.md` | Transparent session-cost assumptions |
| `pyproject.toml` | Local and CI dependency definitions |
| `.github/workflows/ci.yml` | Python tests, lint, and Bicep compilation |
| `SECURITY.md` | Secret-handling and credential-rotation notes |

## Quick start (local checks)

Python 3.10 or newer is required. The following commands do not use Azure:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python data/generate_churn_data.py
python -m pytest
python src/train_local.py
ruff check .
```

The generator is seeded, so the data-quality tests can also create an in-memory dataset when the CSV has not been generated yet.

## Azure lab run

### 1. Deploy

```bash
az login
./infra/deploy.sh                 # defaults to australiaeast
./infra/deploy.sh eastus          # optional alternative region
```

The script prints the storage account and workspace URL. The default resource group is `rg-dbx-churn-lab-aue`; override it with `RG=...` if required.

### 2. Upload the raw CSV

```bash
az storage blob upload \
  --account-name <storage-account-from-deploy> \
  --container-name raw \
  --file data/telco_churn.csv \
  --name telco_churn.csv \
  --auth-mode login
```

If this returns an authorization error, grant your signed-in identity **Storage Blob Data Contributor** on the storage account (or upload through the Azure Portal). Do not put the account key in the upload command or in shell history.

### 3. Store the storage key in a Databricks secret scope

The notebooks do not accept a plaintext key. Configure the Databricks CLI for the workspace URL printed by deployment, then create a scope or use the workspace UI. If the scope already exists, skip the first command.

```bash
databricks configure --host <workspace-url-from-deploy>
databricks secrets create-scope churn-lab
STORAGE_KEY="$(az storage account keys list \
  --resource-group rg-dbx-churn-lab-aue \
  --account-name <storage-account-from-deploy> \
  --query "[0].value" --output tsv)"
databricks secrets put-secret churn-lab storage-account-key --string-value "$STORAGE_KEY"
unset STORAGE_KEY
```

Do not pipe the key directly into `put-secret`; some CLI versions preserve a trailing newline in the secret value, which makes the ABFS driver reject the key. Do not commit the value or place it in a notebook cell.

### 4. Run the notebooks

In the workspace, import `notebooks/`, attach a **Dedicated single-user Databricks Runtime ML** cluster (for example, 14.3 LTS ML), and run notebooks `01` through `05` in order. In each notebook, set these widgets before running the first command:

| Widget | Value |
|---|---|
| `storage_account` | Storage account name printed by deployment |
| `secret_scope` | `churn-lab` |
| `secret_key` | `storage-account-key` |

The storage-key path is retained for this disposable Phase 1 lab. The Delta data is accessed by path rather than registered as Unity Catalog tables; this avoids requiring an external location during the lab.

### 5. Tear down

```bash
./infra/teardown.sh
az group exists --name rg-dbx-churn-lab-aue
```

The deletion is requested asynchronously. The final command should return `false` after Azure finishes deleting the resource group.

## Runtime notes

- Use **Dedicated (single user)** access mode. Shared and serverless modes do not support the storage-key Spark configuration used by this lab.
- Use a **Databricks Runtime ML** cluster for notebooks 04–05. It provides compatible MLflow and scikit-learn versions without ad-hoc `%pip` dependency conflicts.
- Set a 30-minute auto-termination limit and use a single node. This dataset has 10,000 source rows and does not need a multi-node cluster.

## Design decisions and trade-offs

- Delta is used instead of plain Parquet for ACID writes, schema management, and time-travel evidence.
- Silver performs explicit data-quality gates before Gold is written. The local mirror keeps the feature/label contract testable without a Spark cluster.
- The model registry uses a Unity Catalog alias (`@staging`) rather than the retired model-version stages.
- Bronze is a rebuildable batch copy in this version. Incremental ingestion and production orchestration are deliberately not part of this Phase 1 lab.

## Lab evidence

These screenshots were captured during a same-day Azure Databricks run in `australiaeast`. They show the minimum proof path: cluster setup, Bronze ingestion, Gold features, MLflow training, model registry aliasing, and batch inference.

| Compute | Bronze Delta | Gold features |
|---|---|---|
| ![Databricks Runtime ML single-node cluster](docs/images/01-cluster-running.png) | ![Bronze Delta ingest output](docs/images/02-bronze-delta-output.png) | ![Gold feature table output](docs/images/03-gold-features-output.png) |

| MLflow training | Model registry | Batch inference |
|---|---|---|
| ![MLflow model metrics and registration output](docs/images/04-mlflow-metrics.png) | ![MLflow registered model with staging alias](docs/images/05-model-registry-staging.png) | ![Batch inference prediction output](docs/images/06-batch-inference-output.png) |

## Implementation notes

### pandas 3.0 dtype assignment

The generator casts `TotalCharges` to string before inserting blank values. pandas 3.x rejects assigning a string into a float column, while older versions only warned about the upcast.

### Unity Catalog model aliases

Unity Catalog does not support the old `transition_model_version_stage` flow. The training notebook registers the winner and assigns the `staging` alias; batch inference loads `models:/churn_classifier@staging`.
