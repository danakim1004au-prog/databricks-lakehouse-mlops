# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Batch inference from the model registry
# MAGIC Loads the Staging model, scores the Gold feature table, writes a predictions
# MAGIC Delta table — the standard nightly-scoring pattern.

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

dbutils.widgets.text("storage_account", "CHANGE_ME")
STORAGE_ACCOUNT = dbutils.widgets.get("storage_account")
GOLD_PATH = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/churn_features"
PRED_PATH = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/churn_predictions"

FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "spend_delta", "is_month_to_month", "is_fiber",
    "is_echeck", "has_tech_support", "is_paperless",
]

# COMMAND ----------

gold = spark.read.format("delta").load(GOLD_PATH)

# Distribute the sklearn model across the cluster as a Spark UDF
predict_udf = mlflow.pyfunc.spark_udf(
    spark, model_uri="models:/churn_classifier/Staging", result_type="double"
)

predictions = (
    gold
    .withColumn("churn_prediction", predict_udf(*[F.col(c) for c in FEATURES]))
    .withColumn("_scored_ts", F.current_timestamp())
    .select("customerID", "churn_prediction", "churn_label", "_scored_ts")
)

predictions.write.format("delta").mode("overwrite").save(PRED_PATH)
spark.sql(f"CREATE TABLE IF NOT EXISTS gold_churn_predictions USING DELTA LOCATION '{PRED_PATH}'")

# COMMAND ----------

display(
    predictions.groupBy("churn_prediction", "churn_label").count().orderBy("churn_prediction", "churn_label")
)
print(f"Scored {predictions.count():,} customers")
