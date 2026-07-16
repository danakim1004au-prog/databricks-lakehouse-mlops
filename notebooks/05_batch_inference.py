# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Batch inference from the model registry
# MAGIC Loads the `@staging` model, scores the Gold feature table, and writes a prediction
# MAGIC Delta table — the standard nightly-scoring pattern.
# MAGIC
# MAGIC **Requires a Databricks Runtime ML cluster** (same as notebook 04).

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

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
PRED_PATH = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/churn_predictions"

FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "spend_delta", "is_month_to_month", "is_fiber",
    "is_echeck", "has_tech_support", "is_paperless",
]

# COMMAND ----------

gold = spark.read.format("delta").load(GOLD_PATH)

# Distribute the sklearn model across the cluster as a Spark UDF.
# Unity Catalog uses an alias (@staging), not a stage (/Staging).
predict_udf = mlflow.pyfunc.spark_udf(
    spark, model_uri="models:/churn_classifier@staging", result_type="double"
)

predictions = (
    gold
    .withColumn("churn_prediction", predict_udf(*[F.col(c) for c in FEATURES]))
    .withColumn("_scored_ts", F.current_timestamp())
    .select("customerID", "churn_prediction", "churn_label", "_scored_ts")
)

predictions.write.format("delta").mode("overwrite").save(PRED_PATH)

# COMMAND ----------

display(
    predictions.groupBy("churn_prediction", "churn_label").count().orderBy("churn_prediction", "churn_label")
)
print(f"Scored {predictions.count():,} customers")
