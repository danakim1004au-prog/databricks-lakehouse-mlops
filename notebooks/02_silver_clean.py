# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver: shared contract validation and deterministic deduplication
# MAGIC The transform is imported from the same package exercised by local and Spark CI tests.

# COMMAND ----------

import re
import sys
from pathlib import Path

src_path = str((Path.cwd().parent / "src").resolve())
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from databricks_lakehouse_mlops.spark_transforms import silver_clean_spark

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
BRONZE_BATCH_PATH = f"{VOLUME_ROOT}/bronze/{ENVIRONMENT}/batches/{BATCH_ID}"
SILVER_BATCH_PATH = f"{VOLUME_ROOT}/silver/{ENVIRONMENT}/batches/{BATCH_ID}"

# COMMAND ----------

bronze = spark.read.format("delta").load(BRONZE_BATCH_PATH)
silver = silver_clean_spark(bronze)
total = silver.count()
if total == 0:
    raise ValueError("Silver transform returned no rows")

(
    silver.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(SILVER_BATCH_PATH)
)
dbutils.jobs.taskValues.set(key="silver_rows", value=total)
print(f"Silver batch {BATCH_ID}: {total:,} rows; all contract gates passed")
