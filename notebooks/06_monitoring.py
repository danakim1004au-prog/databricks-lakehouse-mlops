# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Labelled performance and feature-shift monitoring
# MAGIC Metrics are append-only and include batch, environment, model, source version, and job lineage.
# MAGIC In production, schedule this step after delayed outcomes have populated `churn_label`.

# COMMAND ----------

import re
import sys
from pathlib import Path

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql import functions as F

src_path = str((Path.cwd().parent / "src").resolve())
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from databricks_lakehouse_mlops.contracts import FEATURES

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("batch_id", "")
dbutils.widgets.text("min_accuracy", "0.70")
dbutils.widgets.text("min_f1", "0.55")
dbutils.widgets.text("min_roc_auc", "0.70")
dbutils.widgets.text("min_average_precision", "0.50")
dbutils.widgets.text("max_feature_shift", "1.0")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ENVIRONMENT = dbutils.widgets.get("environment").strip()
BATCH_ID = dbutils.widgets.get("batch_id").strip()
MIN_ACCURACY = float(dbutils.widgets.get("min_accuracy"))
MIN_F1 = float(dbutils.widgets.get("min_f1"))
MIN_ROC_AUC = float(dbutils.widgets.get("min_roc_auc"))
MIN_AVERAGE_PRECISION = float(dbutils.widgets.get("min_average_precision"))
MAX_FEATURE_SHIFT = float(dbutils.widgets.get("max_feature_shift"))

if not CATALOG or not SCHEMA or not BATCH_ID:
    raise ValueError("Set catalog, schema, and batch_id")
if not re.fullmatch(r"[A-Za-z0-9_.-]+", BATCH_ID):
    raise ValueError("Invalid batch_id")

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}"
GOLD_BATCH_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/features/batches/{BATCH_ID}"
PRED_BATCH_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/predictions/batches/{BATCH_ID}"
PROFILE_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/training_profiles"
MONITORING_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/monitoring_metrics"

# COMMAND ----------

gold = spark.read.format("delta").load(GOLD_BATCH_PATH)
predictions = spark.read.format("delta").load(PRED_BATCH_PATH)
lineage = predictions.select(
    "_job_run_id",
    "_model_name",
    "_model_alias",
    "_model_version",
    "_model_run_id",
    "_gold_delta_version",
    "_decision_threshold",
).distinct().collect()
if len(lineage) != 1:
    raise ValueError("Prediction batch must contain exactly one lineage record")
lineage = lineage[0]

profile = (
    spark.read.format("delta")
    .load(PROFILE_PATH)
    .filter(
        (F.col("model_name") == lineage._model_name)
        & (F.col("model_version") == lineage._model_version)
    )
)
if profile.count() != len(FEATURES):
    raise ValueError("Training profile is missing or incomplete for the scored model version")

scored = predictions.select(
    "customerID", "churn_probability", "churn_prediction"
).join(gold.select("customerID", "churn_label", *FEATURES), on="customerID", how="inner")
if scored.filter(F.col("churn_label").isNull()).limit(1).count():
    raise ValueError("Labelled monitoring cannot run until churn_label is populated")

counts = scored.agg(
    F.count("*").alias("n"),
    F.sum((F.col("churn_prediction") == F.col("churn_label")).cast("int")).alias("correct"),
    F.sum((F.col("churn_prediction") == 1).cast("int")).alias("predicted_positive"),
    F.sum(((F.col("churn_prediction") == 1) & (F.col("churn_label") == 1)).cast("int")).alias("tp"),
    F.sum(((F.col("churn_prediction") == 1) & (F.col("churn_label") == 0)).cast("int")).alias("fp"),
    F.sum(((F.col("churn_prediction") == 0) & (F.col("churn_label") == 1)).cast("int")).alias("fn"),
).first()
if counts.n == 0:
    raise ValueError("No prediction rows matched the labelled Gold batch")

accuracy = counts.correct / counts.n
positive_rate = counts.predicted_positive / counts.n
precision = counts.tp / max(counts.tp + counts.fp, 1)
recall = counts.tp / max(counts.tp + counts.fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-12)

score_and_labels = scored.select("churn_probability", "churn_label").rdd.map(
    lambda row: (float(row.churn_probability), float(row.churn_label))
)
binary_metrics = BinaryClassificationMetrics(score_and_labels)
roc_auc = float(binary_metrics.areaUnderROC)
average_precision = float(binary_metrics.areaUnderPR)

current_means = scored.agg(*[F.avg(feature).alias(feature) for feature in FEATURES]).first()
shift_rows = []
for row in profile.collect():
    denominator = row.training_stddev or 0.0
    shift = (
        0.0
        if denominator == 0.0
        else abs(current_means[row.feature] - row.training_mean) / denominator
    )
    shift_rows.append((row.feature, float(shift)))
max_shift_feature, max_shift = max(shift_rows, key=lambda item: item[1])

metric_rows = [
    ("accuracy", accuracy, MIN_ACCURACY, "minimum", accuracy >= MIN_ACCURACY),
    ("f1", f1, MIN_F1, "minimum", f1 >= MIN_F1),
    ("roc_auc", roc_auc, MIN_ROC_AUC, "minimum", roc_auc >= MIN_ROC_AUC),
    (
        "average_precision",
        average_precision,
        MIN_AVERAGE_PRECISION,
        "minimum",
        average_precision >= MIN_AVERAGE_PRECISION,
    ),
    ("prediction_positive_rate", positive_rate, 0.0, "observed", True),
    (
        f"max_feature_mean_shift:{max_shift_feature}",
        max_shift,
        MAX_FEATURE_SHIFT,
        "maximum",
        max_shift <= MAX_FEATURE_SHIFT,
    ),
]

monitoring = (
    spark.createDataFrame(
        metric_rows,
        "metric string, value double, threshold double, threshold_type string, passed boolean",
    )
    .withColumn("monitored_ts", F.current_timestamp())
    .withColumn("scored_rows", F.lit(counts.n))
    .withColumn("batch_id", F.lit(BATCH_ID))
    .withColumn("job_run_id", F.lit(lineage._job_run_id))
    .withColumn("environment", F.lit(ENVIRONMENT))
    .withColumn("model_name", F.lit(lineage._model_name))
    .withColumn("model_alias", F.lit(lineage._model_alias))
    .withColumn("model_version", F.lit(lineage._model_version))
    .withColumn("model_run_id", F.lit(lineage._model_run_id))
    .withColumn("gold_delta_version", F.lit(lineage._gold_delta_version))
    .withColumn("decision_threshold", F.lit(lineage._decision_threshold))
)
monitoring.write.format("delta").mode("append").save(MONITORING_PATH)
display(monitoring)

failed = [metric for metric, _, _, _, passed in metric_rows if not passed]
if failed:
    raise RuntimeError(f"Monitoring guardrail failed: {', '.join(failed)}")
print(
    f"Monitoring passed for batch {BATCH_ID}: n={counts.n:,}, AUC={roc_auc:.4f}, "
    f"AP={average_precision:.4f}, F1={f1:.4f}, max_shift={max_shift:.4f}"
)
