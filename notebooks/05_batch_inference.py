# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Probability scoring with immutable lineage
# MAGIC Scores only the current Gold batch and records the exact model, MLflow run,
# MAGIC source Delta version, threshold, environment, and job run.

# COMMAND ----------

import re
import sys
from pathlib import Path

import mlflow
from delta.tables import DeltaTable
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

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
dbutils.widgets.text("model_alias", "staging")
dbutils.widgets.text("decision_threshold", "0.5")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ENVIRONMENT = dbutils.widgets.get("environment").strip()
BATCH_ID = dbutils.widgets.get("batch_id").strip()
JOB_RUN_ID = dbutils.widgets.get("job_run_id").strip() or BATCH_ID
MODEL_NAME = dbutils.widgets.get("model_name").strip()
MODEL_ALIAS = dbutils.widgets.get("model_alias").strip()
DECISION_THRESHOLD = float(dbutils.widgets.get("decision_threshold"))

if not CATALOG or not SCHEMA or not BATCH_ID or MODEL_NAME.count(".") != 2:
    raise ValueError("Set catalog, schema, batch_id, and a fully qualified model_name")
if not re.fullmatch(r"[A-Za-z0-9_.-]+", BATCH_ID):
    raise ValueError("Invalid batch_id")
if MODEL_ALIAS not in {"staging", "production"}:
    raise ValueError("model_alias must be staging or production")
if not 0.0 < DECISION_THRESHOLD < 1.0:
    raise ValueError("decision_threshold must be between zero and one")

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}"
GOLD_BATCH_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/features/batches/{BATCH_ID}"
PRED_BATCH_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/predictions/batches/{BATCH_ID}"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
MODEL_VERSION = str(model_version.version)
MODEL_RUN_ID = str(model_version.run_id)
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# COMMAND ----------

gold = spark.read.format("delta").load(GOLD_BATCH_PATH)
gold_version = int(DeltaTable.forPath(spark, GOLD_BATCH_PATH).history(1).first().version)

predict_probability = mlflow.pyfunc.spark_udf(
    spark, model_uri=MODEL_URI, result_type="double"
)
predictions = (
    gold.withColumn("churn_probability", predict_probability(*[F.col(c) for c in FEATURES]))
    .withColumn(
        "churn_prediction",
        (F.col("churn_probability") >= F.lit(DECISION_THRESHOLD)).cast("int"),
    )
    .withColumn("_scored_ts", F.current_timestamp())
    .withColumn("_batch_id", F.lit(BATCH_ID))
    .withColumn("_job_run_id", F.lit(JOB_RUN_ID))
    .withColumn("_environment", F.lit(ENVIRONMENT))
    .withColumn("_model_name", F.lit(MODEL_NAME))
    .withColumn("_model_alias", F.lit(MODEL_ALIAS))
    .withColumn("_model_version", F.lit(MODEL_VERSION))
    .withColumn("_model_run_id", F.lit(MODEL_RUN_ID))
    .withColumn("_gold_delta_version", F.lit(gold_version))
    .withColumn("_decision_threshold", F.lit(DECISION_THRESHOLD))
    .select(
        "customerID",
        "churn_probability",
        "churn_prediction",
        "churn_label",
        "_scored_ts",
        "_batch_id",
        "_job_run_id",
        "_environment",
        "_model_name",
        "_model_alias",
        "_model_version",
        "_model_run_id",
        "_gold_delta_version",
        "_decision_threshold",
    )
)
rows = predictions.count()
if rows == 0:
    raise ValueError("No rows were scored")

(
    predictions.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(PRED_BATCH_PATH)
)

dbutils.jobs.taskValues.set(key="scored_rows", value=rows)
dbutils.jobs.taskValues.set(key="model_version", value=MODEL_VERSION)
display(predictions.orderBy(F.col("churn_probability").desc()).limit(20))
print(
    f"Scored batch {BATCH_ID}: {rows:,} rows with {MODEL_NAME}@{MODEL_ALIAS} "
    f"(v{MODEL_VERSION}, threshold={DECISION_THRESHOLD})"
)
