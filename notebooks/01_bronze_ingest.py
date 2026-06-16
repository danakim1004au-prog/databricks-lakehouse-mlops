# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze: raw CSV → Delta
# MAGIC Schema-on-read ingest with lineage metadata. No transformations — Bronze is an
# MAGIC immutable replayable copy of the source.

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.text("storage_account", "CHANGE_ME")
STORAGE_ACCOUNT = dbutils.widgets.get("storage_account")

RAW_PATH = f"abfss://raw@{STORAGE_ACCOUNT}.dfs.core.windows.net/telco_churn.csv"
BRONZE_PATH = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net/telco_churn"

# COMMAND ----------

raw_df = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "false")   # Bronze keeps everything as string
    .load(RAW_PATH)
)

bronze_df = (
    raw_df
    .withColumn("_ingest_ts", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
)

(
    bronze_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(BRONZE_PATH)
)

# COMMAND ----------

spark.sql(f"CREATE TABLE IF NOT EXISTS bronze_telco_churn USING DELTA LOCATION '{BRONZE_PATH}'")
print(f"Bronze rows: {spark.read.format('delta').load(BRONZE_PATH).count():,}")
