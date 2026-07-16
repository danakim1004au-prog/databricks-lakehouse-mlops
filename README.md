# Azure Databricks Lakehouse MLOps

[![CI](https://github.com/danakim1004au-prog/databricks-lakehouse-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/danakim1004au-prog/databricks-lakehouse-mlops/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-blue.svg)](LICENSE)

An end-to-end customer-retention pipeline that turns raw account data into traceable churn-risk predictions. The implementation covers reproducible Azure infrastructure, Delta Lake data contracts, tracked model selection, Unity Catalog registration, scheduled retraining, batch scoring, and model/data guardrails.

The dataset is synthetic and intentionally familiar. The focus is the operational lifecycle around the model: repeatable deployment, failure gates, lineage, cost controls, automation, and observable outputs.

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
  -> Monitoring Delta
       Accuracy and feature-shift guardrails retained as metric history

Databricks Declarative Automation Bundle
  -> Daily scoring + monitoring Lakeflow Job
  -> Weekly full-pipeline retraining Lakeflow Job
  -> Optional scale-to-zero model serving bundle
```

## Repository layout

| Path | Purpose |
|---|---|
| `infra/main.bicep` | ADLS Gen2, Premium Databricks workspace, and Access Connector |
| `infra/deploy.sh` | Deploy the disposable lab resource group |
| `infra/teardown.sh` | Request full resource-group deletion |
| `databricks.yml` | Development and production bundle targets |
| `resources/churn_jobs.yml` | Scheduled retraining and batch-monitoring Lakeflow Jobs |
| `serving/databricks.yml` | Optional scale-to-zero model serving endpoint bundle |
| `data/generate_churn_data.py` | Seeded synthetic telco dataset generator |
| `notebooks/01_bronze_ingest.py` | CSV to Bronze Delta with ingest metadata |
| `notebooks/02_silver_clean.py` | Dedupe, type casting, null handling, and DQ gates |
| `notebooks/03_gold_features.py` | Feature engineering for training and scoring |
| `notebooks/04_train_mlflow.py` | Model comparison, MLflow tracking, and registration |
| `notebooks/05_batch_inference.py` | Load the `@staging` model and write predictions |
| `notebooks/06_monitoring.py` | Persist performance/shift metrics and enforce guardrails |
| `src/train_local.py` | Spark-free local training smoke test used by CI |
| `tests/` | Data, feature-contract, and bundle-structure tests |
| `docs/images/*.png` | Evidence from a real Azure Databricks run |
| `docs/COST_ESTIMATE.md` | Transparent session-cost assumptions |
| `.github/workflows/ci.yml` | Tests, local training, lint, Bicep, and shell checks |
| `SECURITY.md` | Secret-handling and credential-rotation notes |
| `LICENSE` | MIT licence |

## Local quality gate

Python 3.10 or newer is required. These commands do not use Azure:

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

The generator is seeded. Tests create their own in-memory dataset, so a missing generated CSV cannot hide a failure. The same checks run in GitHub Actions on Python 3.10 and 3.12; CI also compiles the Bicep template and checks the shell scripts.

## Azure setup

### 1. Deploy the disposable environment

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
  --auth-mode login \
  --overwrite
```

If this returns an authorisation error, grant the signed-in identity **Storage Blob Data Contributor** on the storage account or upload through the Azure Portal. Do not put the account key in the command or in shell history.

### 3. Create the Databricks secret

Use browser-based OAuth for the deployed workspace. It supplies the workspace API access required by bundles without keeping a long-lived PAT. If the scope already exists, skip its creation.

```bash
databricks auth login \
  --host https://<workspace-url-from-deploy> \
  --profile churn-lab

export DATABRICKS_CONFIG_PROFILE=churn-lab
databricks secrets create-scope churn-lab

STORAGE_KEY="$(az storage account keys list \
  --resource-group rg-dbx-churn-lab-aue \
  --account-name <storage-account-from-deploy> \
  --query "[0].value" --output tsv)"

databricks secrets put-secret \
  churn-lab storage-account-key \
  --string-value "$STORAGE_KEY"

unset STORAGE_KEY
```

Do not pipe the key directly into `put-secret`; some CLI versions preserve a trailing newline, which makes the ABFS driver reject the key. Never commit the value or place it in a notebook cell.

If `bundle validate` later reports `Provided access token does not have required scopes: workspace`, the CLI is still using a restricted PAT. Check `databricks auth describe` and repeat the OAuth login/export commands above.

## Orchestrated MLOps run

The development target deploys both jobs with schedules paused. This is the safe portfolio-lab default.

```bash
databricks bundle validate -t dev \
  --var="storage_account=<storage-account-from-deploy>" \
  --var="model_name=<catalog.default.churn_classifier>"

databricks bundle deploy -t dev \
  --var="storage_account=<storage-account-from-deploy>" \
  --var="model_name=<catalog.default.churn_classifier>"

databricks bundle run -t dev churn_retraining \
  --var="storage_account=<storage-account-from-deploy>" \
  --var="model_name=<catalog.default.churn_classifier>"
```

The retraining DAG runs `01 -> 02 -> 03 -> 04 -> 05 -> 06` on one Dedicated single-node Runtime ML job cluster. Any data-quality or monitoring guardrail failure stops the run. The separate daily job can be tested with:

```bash
databricks bundle run -t dev churn_batch_monitoring \
  --var="storage_account=<storage-account-from-deploy>" \
  --var="model_name=<catalog.default.churn_classifier>"
```

The `prod` target changes both schedules to `UNPAUSED`: daily scoring/monitoring at 02:00 and weekly retraining at 03:00 Sunday, using the `Australia/Sydney` timezone. Deploy it only when recurring Azure spend is intentional:

```bash
databricks bundle deploy -t prod \
  --var="storage_account=<storage-account-from-deploy>" \
  --var="model_name=<catalog.default.churn_classifier>"
```

### Manual notebook path

For a walkthrough, import `notebooks/`, attach a **Dedicated single-user Databricks Runtime ML** cluster, and run notebooks `01` through `06`. Set the common widgets before running each notebook:

| Widget | Value |
|---|---|
| `storage_account` | Storage account name printed by deployment |
| `secret_scope` | `churn-lab` |
| `secret_key` | `storage-account-key` |
| `model_name` | `churn_classifier` in notebooks 04–05 |

The storage-key path and path-based Delta tables keep the environment disposable. A production tenancy should replace this with Unity Catalog external locations and managed identity access.

## Optional real-time serving

`serving/databricks.yml` defines a separate Small, scale-to-zero endpoint. It is isolated from the main bundle because an endpoint should not be created before a registered model version exists.

From the model registry screen, copy the fully qualified model name (`catalog.schema.churn_classifier`) and the version carrying `@staging`, then run:

```bash
cd serving

databricks bundle validate -t dev \
  --var="registered_model_name=<catalog.schema.churn_classifier>" \
  --var="model_version=<version-number>"

databricks bundle deploy -t dev \
  --var="registered_model_name=<catalog.schema.churn_classifier>" \
  --var="model_version=<version-number>"
```

Model Serving is optional and billed separately. After testing, remove the endpoint from the same directory with the same variable values, then return to the repository root:

```bash
databricks bundle destroy -t dev \
  --var="registered_model_name=<catalog.schema.churn_classifier>" \
  --var="model_version=<version-number>"

cd ..
```

## Cost and runtime controls

- Jobs use a single `Standard_D4ds_v5` node and prevent overlapping runs.
- Development schedules are paused; recurring schedules require an explicit `prod` deployment.
- The serving endpoint uses the smallest workload size and scales to zero.
- Interactive evidence runs use a 15-minute auto-termination setting.
- `docs/COST_ESTIMATE.md` records assumptions and teardown checks rather than calling trial credits “free”.

## Design decisions and trade-offs

- Delta provides ACID writes, schema management, and an auditable metric history.
- Silver fails before Gold when row, key, null, or contract-domain checks break.
- The local sklearn mirror keeps core feature/label behaviour testable without Spark.
- Unity Catalog aliases replace retired model-version stages; batch scoring loads `@staging`.
- Weekly retraining is time-based and intentionally simple. In a higher-volume system it would be triggered by new data plus approval thresholds.
- Monitoring uses labelled accuracy and standardised feature-mean shift. It is transparent and cheap, but not a substitute for a full drift platform or delayed-outcome design.
- The current lab uses a storage key so reviewers can reproduce it in one session. Managed identity and UC external locations are the next security boundary for a long-lived environment.

## Verified Azure evidence

These screenshots were captured during same-day `australiaeast` runs. They document both the manual notebook path and the Databricks Bundle job run that orchestrates retraining, registration, batch scoring, and monitoring. Development schedules are intentionally paused, so this evidence proves the deployable job workflow rather than a live recurring production schedule.

| Compute | Bronze Delta | Gold features |
|---|---|---|
| ![Databricks Runtime ML single-node cluster](docs/images/01-cluster-running.png) | ![Bronze Delta ingest output](docs/images/02-bronze-delta-output.png) | ![Gold feature table output](docs/images/03-gold-features-output.png) |

| MLflow training | Model registry | Batch inference |
|---|---|---|
| ![MLflow model metrics and registration output](docs/images/04-mlflow-metrics.png) | ![MLflow registered model with staging alias](docs/images/05-model-registry-staging.png) | ![Batch inference prediction output](docs/images/06-batch-inference-output.png) |

| Databricks Bundle workflow |
|---|
| ![Databricks workflow run succeeded for the churn retraining DAG](docs/images/07-churn-retraining-workflow-success.png) |

## Implementation notes

### pandas 3 dtype assignment

The generator casts `TotalCharges` to string before inserting blank values. pandas 3 rejects assigning a string into a float column, while older versions only warned about the implicit upcast.

### Unity Catalog model aliases

Unity Catalog does not support the old `transition_model_version_stage` flow. The training notebook registers the winner and assigns the `staging` alias; batch inference loads `models:/churn_classifier@staging`.
