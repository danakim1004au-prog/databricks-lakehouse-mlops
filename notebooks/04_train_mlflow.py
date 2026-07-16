# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Train churn models with MLflow tracking + registry
# MAGIC Logistic-regression baseline vs gradient boosting; both runs logged to MLflow,
# MAGIC the winner registered as `churn_classifier` and assigned the `@staging` alias.
# MAGIC
# MAGIC **Requires a Databricks Runtime ML cluster** (e.g. 14.3 LTS ML) — mlflow and
# MAGIC scikit-learn are pre-installed there. On a standard runtime, `%pip install`
# MAGIC tends to break on `typing_extensions`/dependency conflicts, so use ML runtime.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

dbutils.widgets.text("storage_account", "")
dbutils.widgets.text("secret_scope", "churn-lab")
dbutils.widgets.text("secret_key", "storage-account-key")

STORAGE_ACCOUNT = dbutils.widgets.get("storage_account").strip()
SECRET_SCOPE    = dbutils.widgets.get("secret_scope").strip()
SECRET_KEY_NAME = dbutils.widgets.get("secret_key").strip()

if not STORAGE_ACCOUNT:
    raise ValueError("Set the storage_account widget to the account name printed by deploy.sh")
if not SECRET_SCOPE or not SECRET_KEY_NAME:
    raise ValueError("Set secret_scope and secret_key before running the notebook")

STORAGE_KEY = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY_NAME)

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    STORAGE_KEY,
)

GOLD_PATH = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/churn_features"

FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "spend_delta", "is_month_to_month", "is_fiber",
    "is_echeck", "has_tech_support", "is_paperless",
]
LABEL = "churn_label"
MODEL_NAME = "churn_classifier"

# COMMAND ----------

pdf = spark.read.format("delta").load(GOLD_PATH).select(*FEATURES, LABEL).toPandas()
X_train, X_test, y_train, y_test = train_test_split(
    pdf[FEATURES], pdf[LABEL], test_size=0.2, random_state=42, stratify=pdf[LABEL]
)

mlflow.set_experiment("/Shared/churn-lakehouse")

# COMMAND ----------

def train_and_log(name, model, params):
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.set_tag("model_selection_metric", "roc_auc")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(y_test, proba),
            "average_precision": average_precision_score(y_test, proba),
            "f1": f1_score(y_test, preds),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_param("feature_count", len(FEATURES))
        mlflow.log_text("\n".join(FEATURES), "features.txt")
        mlflow.sklearn.log_model(
            model, artifact_path="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(5),
        )
        print(
            f"{name}: AUC={metrics['roc_auc']:.4f} "
            f"AP={metrics['average_precision']:.4f} F1={metrics['f1']:.4f}"
        )
        return run.info.run_id, metrics["roc_auc"]


baseline_params = {"C": 1.0, "max_iter": 1000, "random_state": 42}
gb_params = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1}

baseline_run, baseline_auc = train_and_log(
    "logreg_baseline", LogisticRegression(**baseline_params), baseline_params
)
gb_run, gb_auc = train_and_log(
    "gradient_boosting", GradientBoostingClassifier(**gb_params, random_state=42), gb_params
)

# COMMAND ----------

# MAGIC %md ## Register the winner and assign the `@staging` alias
# MAGIC Unity Catalog dropped model *stages* in favour of *aliases*
# MAGIC (`transition_model_version_stage` raises `MlflowException` on UC). We set a
# MAGIC `@staging` alias instead — load later with `models:/churn_classifier@staging`.

# COMMAND ----------

best_run = gb_run if gb_auc >= baseline_auc else baseline_run
mv = mlflow.register_model(f"runs:/{best_run}/model", MODEL_NAME)

client = MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "staging", mv.version)
print(f"Registered {MODEL_NAME} v{mv.version} with alias @staging (run {best_run})")
