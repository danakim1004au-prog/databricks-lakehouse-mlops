# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Train churn models with MLflow tracking + registry
# MAGIC Logistic-regression baseline vs gradient boosting; both runs logged to MLflow,
# MAGIC the winner registered as `churn_classifier` and promoted to Staging.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

dbutils.widgets.text("storage_account", "CHANGE_ME")
STORAGE_ACCOUNT = dbutils.widgets.get("storage_account")
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
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(y_test, proba),
            "f1": f1_score(y_test, preds),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, artifact_path="model",
            signature=infer_signature(X_train, model.predict(X_train)),
        )
        print(f"{name}: AUC={metrics['roc_auc']:.4f} F1={metrics['f1']:.4f}")
        return run.info.run_id, metrics["roc_auc"]


baseline_params = {"C": 1.0, "max_iter": 1000}
gb_params = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1}

baseline_run, baseline_auc = train_and_log(
    "logreg_baseline", LogisticRegression(**baseline_params), baseline_params
)
gb_run, gb_auc = train_and_log(
    "gradient_boosting", GradientBoostingClassifier(**gb_params, random_state=42), gb_params
)

# COMMAND ----------

# MAGIC %md ## Register the winner and promote to Staging

# COMMAND ----------

best_run = gb_run if gb_auc >= baseline_auc else baseline_run
mv = mlflow.register_model(f"runs:/{best_run}/model", MODEL_NAME)

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(MODEL_NAME, mv.version, stage="Staging")
print(f"Registered {MODEL_NAME} v{mv.version} -> Staging (run {best_run})")
