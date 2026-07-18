# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Train, validate, register candidate, and gate staging promotion
# MAGIC Candidate selection uses validation data; final gates use an untouched test split.
# MAGIC Passing creates/updates `@candidate` and `@staging`, never `@production`.

# COMMAND ----------

import re
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
from delta.tables import DeltaTable
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

src_path = str((Path.cwd().parent / "src").resolve())
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from databricks_lakehouse_mlops.contracts import FEATURES

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("batch_id", "")
dbutils.widgets.text("job_run_id", "")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("min_roc_auc", "0.75")
dbutils.widgets.text("min_average_precision", "0.55")
dbutils.widgets.text("min_f1", "0.55")
dbutils.widgets.text("max_auc_regression", "0.01")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ENVIRONMENT = dbutils.widgets.get("environment").strip()
BATCH_ID = dbutils.widgets.get("batch_id").strip()
JOB_RUN_ID = dbutils.widgets.get("job_run_id").strip() or BATCH_ID
MODEL_NAME = dbutils.widgets.get("model_name").strip()
MIN_ROC_AUC = float(dbutils.widgets.get("min_roc_auc"))
MIN_AVERAGE_PRECISION = float(dbutils.widgets.get("min_average_precision"))
MIN_F1 = float(dbutils.widgets.get("min_f1"))
MAX_AUC_REGRESSION = float(dbutils.widgets.get("max_auc_regression"))

if not CATALOG or not SCHEMA or not BATCH_ID or MODEL_NAME.count(".") != 2:
    raise ValueError("Set catalog, schema, batch_id, and a fully qualified model_name")
if not re.fullmatch(r"[A-Za-z0-9_.-]+", BATCH_ID):
    raise ValueError("Invalid batch_id")

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}"
GOLD_BATCH_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/features/batches/{BATCH_ID}"
PROFILE_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/training_profiles"
MODEL_METRICS_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/model_metrics"

mlflow.set_experiment(f"/Shared/churn-lakehouse/{ENVIRONMENT}")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

pdf = spark.read.format("delta").load(GOLD_BATCH_PATH).select(*FEATURES, "churn_label").toPandas()
development, test = train_test_split(
    pdf, test_size=0.2, random_state=42, stratify=pdf["churn_label"]
)
train, validation = train_test_split(
    development,
    test_size=0.25,
    random_state=42,
    stratify=development["churn_label"],
)


class ChurnProbabilityModel(mlflow.pyfunc.PythonModel):
    """Serve churn probability so thresholds can change without retraining."""

    def __init__(self, estimator, features):
        self.estimator = estimator
        self.features = features

    def predict(self, context, model_input, params=None):
        del context, params
        return self.estimator.predict_proba(model_input[self.features])[:, 1]


def evaluate(model, frame):
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(frame["churn_label"], probability)),
        "average_precision": float(
            average_precision_score(frame["churn_label"], probability)
        ),
        "f1": float(f1_score(frame["churn_label"], prediction)),
    }


def train_and_log(name, model, params):
    with mlflow.start_run(run_name=name) as run:
        model.fit(train[FEATURES], train["churn_label"])
        validation_metrics = evaluate(model, validation)
        mlflow.log_params(params)
        mlflow.log_param("decision_threshold", 0.5)
        mlflow.log_param("feature_count", len(FEATURES))
        mlflow.log_metrics({f"validation_{key}": value for key, value in validation_metrics.items()})
        mlflow.set_tags(
            {
                "environment": ENVIRONMENT,
                "training_batch_id": BATCH_ID,
                "model_selection_metric": "validation_roc_auc",
            }
        )
        probability_model = ChurnProbabilityModel(model, FEATURES)
        example = train[FEATURES].head(5)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=probability_model,
            signature=infer_signature(example, probability_model.predict(None, example)),
            input_example=example,
        )
        return run.info.run_id, model, validation_metrics


candidates = [
    (
        "logreg_baseline",
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ]
        ),
        {"C": 1.0, "max_iter": 1000, "random_state": 42, "scaled": True},
    ),
    (
        "gradient_boosting",
        GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
        ),
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
    ),
]
trained = [train_and_log(*candidate) for candidate in candidates]
best_run, best_model, validation_metrics = max(
    trained, key=lambda item: item[2]["roc_auc"]
)
test_metrics = evaluate(best_model, test)

with mlflow.start_run(run_id=best_run):
    mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
    mlflow.log_text("\n".join(FEATURES), "features.txt")

# COMMAND ----------

client = MlflowClient()
champion_auc = None
try:
    champion = client.get_model_version_by_alias(MODEL_NAME, "production")
    if DeltaTable.isDeltaTable(spark, MODEL_METRICS_PATH):
        champion_row = (
            spark.read.format("delta")
            .load(MODEL_METRICS_PATH)
            .filter(
                (F.col("model_name") == MODEL_NAME)
                & (F.col("model_version") == str(champion.version))
            )
            .orderBy(F.col("registered_ts").desc())
            .first()
        )
        if champion_row:
            champion_auc = float(champion_row.test_roc_auc)
except Exception as exc:
    print(f"No production champion comparison available yet: {exc}")

gate_results = {
    "minimum_roc_auc": test_metrics["roc_auc"] >= MIN_ROC_AUC,
    "minimum_average_precision": test_metrics["average_precision"] >= MIN_AVERAGE_PRECISION,
    "minimum_f1": test_metrics["f1"] >= MIN_F1,
    "champion_non_regression": (
        champion_auc is None
        or test_metrics["roc_auc"] >= champion_auc - MAX_AUC_REGRESSION
    ),
}
gate_passed = all(gate_results.values())

model_version = mlflow.register_model(f"runs:/{best_run}/model", MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "candidate", model_version.version)

profile_rows = [
    (
        MODEL_NAME,
        str(model_version.version),
        best_run,
        BATCH_ID,
        ENVIRONMENT,
        feature,
        float(train[feature].mean()),
        float(train[feature].std(ddof=0)),
        int(len(train)),
    )
    for feature in FEATURES
]
(
    spark.createDataFrame(
        profile_rows,
        "model_name string, model_version string, model_run_id string, "
        "training_batch_id string, environment string, feature string, "
        "training_mean double, training_stddev double, training_rows long",
    )
    .withColumn("profiled_ts", F.current_timestamp())
    .write.format("delta")
    .mode("append")
    .save(PROFILE_PATH)
)

metric_row = [
    (
        MODEL_NAME,
        str(model_version.version),
        best_run,
        BATCH_ID,
        JOB_RUN_ID,
        ENVIRONMENT,
        float(validation_metrics["roc_auc"]),
        float(test_metrics["roc_auc"]),
        float(test_metrics["average_precision"]),
        float(test_metrics["f1"]),
        float(champion_auc) if champion_auc is not None else None,
        gate_passed,
        str(gate_results),
    )
]
(
    spark.createDataFrame(
        metric_row,
        "model_name string, model_version string, model_run_id string, "
        "training_batch_id string, job_run_id string, environment string, "
        "validation_roc_auc double, test_roc_auc double, test_average_precision double, "
        "test_f1 double, champion_roc_auc double, gate_passed boolean, gate_results string",
    )
    .withColumn("registered_ts", F.current_timestamp())
    .write.format("delta")
    .mode("append")
    .save(MODEL_METRICS_PATH)
)

dbutils.jobs.taskValues.set(key="candidate_model_version", value=str(model_version.version))
if not gate_passed:
    failed = [name for name, passed in gate_results.items() if not passed]
    raise RuntimeError(f"Candidate v{model_version.version} failed gates: {', '.join(failed)}")

client.set_registered_model_alias(MODEL_NAME, "staging", model_version.version)
print(
    f"Candidate {MODEL_NAME} v{model_version.version} passed {gate_results}; "
    "@staging updated. @production requires notebook 07 approval."
)
