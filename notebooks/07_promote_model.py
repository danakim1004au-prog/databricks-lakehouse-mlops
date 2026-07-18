# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Explicit production promotion
# MAGIC This notebook is deliberately unscheduled. It records an approver and promotes only
# MAGIC a gate-passing staging version when `confirm=PROMOTE` is supplied.

# COMMAND ----------

from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("environment", "prod")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("candidate_version", "")
dbutils.widgets.text("approved_by", "")
dbutils.widgets.text("confirm", "")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ENVIRONMENT = dbutils.widgets.get("environment").strip()
MODEL_NAME = dbutils.widgets.get("model_name").strip()
CANDIDATE_VERSION = dbutils.widgets.get("candidate_version").strip()
APPROVED_BY = dbutils.widgets.get("approved_by").strip()
CONFIRM = dbutils.widgets.get("confirm").strip()

if not CATALOG or not SCHEMA or MODEL_NAME.count(".") != 2:
    raise ValueError("Set catalog, schema, and a fully qualified model_name")
if not APPROVED_BY or CONFIRM != "PROMOTE":
    raise ValueError("Production promotion requires approved_by and confirm=PROMOTE")

client = MlflowClient(registry_uri="databricks-uc")
staging = client.get_model_version_by_alias(MODEL_NAME, "staging")
version = CANDIDATE_VERSION or str(staging.version)
if version != str(staging.version):
    raise ValueError("candidate_version must match the current @staging version")

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}"
MODEL_METRICS_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/model_metrics"
PROMOTION_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/model_promotions"

metrics = (
    spark.read.format("delta")
    .load(MODEL_METRICS_PATH)
    .filter(
        (F.col("model_name") == MODEL_NAME)
        & (F.col("model_version") == version)
        & F.col("gate_passed")
    )
    .orderBy(F.col("registered_ts").desc())
    .first()
)
if not metrics:
    raise ValueError("No passing gate record exists for this model version")

previous_version = None
try:
    previous_version = str(client.get_model_version_by_alias(MODEL_NAME, "production").version)
except Exception:
    pass

client.set_registered_model_alias(MODEL_NAME, "production", version)
promotion = spark.createDataFrame(
    [
        (
            MODEL_NAME,
            version,
            previous_version,
            APPROVED_BY,
            ENVIRONMENT,
            metrics.model_run_id,
            metrics.training_batch_id,
        )
    ],
    "model_name string, promoted_version string, previous_version string, approved_by string, "
    "environment string, model_run_id string, training_batch_id string",
).withColumn("promoted_ts", F.current_timestamp())
promotion.write.format("delta").mode("append").save(PROMOTION_PATH)
display(promotion)
print(f"Promoted {MODEL_NAME} v{version} to @production; approved by {APPROVED_BY}")
