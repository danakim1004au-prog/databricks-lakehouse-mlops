# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Gold: versioned model feature batch
# MAGIC Feature engineering is shared with CI and output remains isolated by environment and batch.

# COMMAND ----------

import re
import sys
from pathlib import Path

from pyspark.sql import functions as F

src_path = str((Path.cwd().parent / "src").resolve())
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from databricks_lakehouse_mlops.spark_transforms import gold_features_spark

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("batch_id", "")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ENVIRONMENT = dbutils.widgets.get("environment").strip()
BATCH_ID = dbutils.widgets.get("batch_id").strip()

if not CATALOG or not SCHEMA or not BATCH_ID:
    raise ValueError("Set catalog, schema, and batch_id")
if not re.fullmatch(r"[A-Za-z0-9_.-]+", BATCH_ID):
    raise ValueError("Invalid batch_id")

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}"
SILVER_BATCH_PATH = f"{VOLUME_ROOT}/silver/{ENVIRONMENT}/batches/{BATCH_ID}"
GOLD_BATCH_PATH = f"{VOLUME_ROOT}/gold/{ENVIRONMENT}/features/batches/{BATCH_ID}"

# COMMAND ----------

silver = spark.read.format("delta").load(SILVER_BATCH_PATH)
gold = gold_features_spark(silver)
rows = gold.count()
if rows == 0:
    raise ValueError("Gold transform returned no rows")

(
    gold.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(GOLD_BATCH_PATH)
)
dbutils.jobs.taskValues.set(key="gold_rows", value=rows)
display(
    gold.groupBy("tenure_bucket").agg(
        F.avg("churn_label").alias("churn_rate"), F.count("*").alias("n")
    )
)
