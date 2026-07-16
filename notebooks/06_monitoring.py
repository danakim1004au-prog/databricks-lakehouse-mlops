# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Model and feature monitoring
# MAGIC Compares the latest labelled batch with the training profile, writes an auditable
# MAGIC Delta metric history, and fails the Lakeflow Job when a guardrail is breached.

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.text("storage_account", "")
dbutils.widgets.text("secret_scope", "churn-lab")
dbutils.widgets.text("secret_key", "storage-account-key")
dbutils.widgets.text("min_accuracy", "0.70")
dbutils.widgets.text("max_feature_shift", "1.0")

STORAGE_ACCOUNT = dbutils.widgets.get("storage_account").strip()
SECRET_SCOPE = dbutils.widgets.get("secret_scope").strip()
SECRET_KEY_NAME = dbutils.widgets.get("secret_key").strip()
MIN_ACCURACY = float(dbutils.widgets.get("min_accuracy"))
MAX_FEATURE_SHIFT = float(dbutils.widgets.get("max_feature_shift"))

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
PRED_PATH = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/churn_predictions"
PROFILE_PATH = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/training_profile"
MONITORING_PATH = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/monitoring_metrics"

FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "spend_delta", "is_month_to_month", "is_fiber",
    "is_echeck", "has_tech_support", "is_paperless",
]

# COMMAND ----------

gold = spark.read.format("delta").load(GOLD_PATH)
predictions = spark.read.format("delta").load(PRED_PATH)
profile = spark.read.format("delta").load(PROFILE_PATH)

scored = predictions.select("customerID", "churn_prediction").join(
    gold.select("customerID", "churn_label", *FEATURES),
    on="customerID",
    how="inner",
)

counts = scored.agg(
    F.count("*").alias("n"),
    F.sum((F.col("churn_prediction") == F.col("churn_label")).cast("int")).alias("correct"),
    F.sum((F.col("churn_prediction") == 1).cast("int")).alias("predicted_positive"),
    F.sum(((F.col("churn_prediction") == 1) & (F.col("churn_label") == 1)).cast("int")).alias("tp"),
    F.sum(((F.col("churn_prediction") == 1) & (F.col("churn_label") == 0)).cast("int")).alias("fp"),
    F.sum(((F.col("churn_prediction") == 0) & (F.col("churn_label") == 1)).cast("int")).alias("fn"),
).first()

if counts.n == 0:
    raise ValueError("No prediction rows matched Gold customer IDs")

accuracy = counts.correct / counts.n
positive_rate = counts.predicted_positive / counts.n
precision = counts.tp / max(counts.tp + counts.fp, 1)
recall = counts.tp / max(counts.tp + counts.fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-12)

# A value of 1.0 means the current feature mean moved by one training standard
# deviation. Constant features use a zero denominator and therefore report zero.
current_means = scored.agg(*[F.avg(feature).alias(feature) for feature in FEATURES]).first()
shift_rows = []
for row in profile.collect():
    denominator = row.training_stddev or 0.0
    shift = 0.0 if denominator == 0.0 else abs(current_means[row.feature] - row.training_mean) / denominator
    shift_rows.append((row.feature, float(shift)))

max_shift_feature, max_shift = max(shift_rows, key=lambda item: item[1])

metric_rows = [
    ("accuracy", float(accuracy), float(MIN_ACCURACY), "minimum", accuracy >= MIN_ACCURACY),
    ("f1", float(f1), 0.0, "observed", True),
    ("prediction_positive_rate", float(positive_rate), 0.0, "observed", True),
    (
        f"max_feature_mean_shift:{max_shift_feature}",
        float(max_shift),
        float(MAX_FEATURE_SHIFT),
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
)

monitoring.write.format("delta").mode("append").save(MONITORING_PATH)
display(monitoring)

failed = [metric for metric, _, _, _, passed in metric_rows if not passed]
if failed:
    raise RuntimeError(f"Monitoring guardrail failed: {', '.join(failed)}")

print(
    f"Monitoring passed for {counts.n:,} rows: accuracy={accuracy:.4f}, "
    f"f1={f1:.4f}, max feature shift={max_shift:.4f} ({max_shift_feature})"
)
