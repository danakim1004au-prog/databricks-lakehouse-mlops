# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze: raw CSV → Delta
# MAGIC Schema-on-read ingest with lineage metadata. No business transformations are applied
# MAGIC in Bronze; this Phase 1 lab rebuilds the batch copy on each run.

# COMMAND ----------

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

RAW_PATH    = f"abfss://raw@{STORAGE_ACCOUNT}.dfs.core.windows.net/telco_churn.csv"
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

# NOTE: On Unity Catalog clusters, registering an external table via
# CREATE TABLE ... LOCATION 'abfss://...' requires a UC External Location.
# This lab reads/writes Delta by path instead, which needs no UC grant.
print(f"Bronze rows: {spark.read.format('delta').load(BRONZE_PATH).count():,}")
