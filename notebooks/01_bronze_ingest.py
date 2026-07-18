# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze: immutable input batch → Delta
# MAGIC Each job run reads one explicit raw-volume input and writes an isolated batch path.
# MAGIC Training and scoring inputs are separate, so monitoring never re-scores the training set.

# COMMAND ----------

import re
from datetime import datetime, timezone

from delta.tables import DeltaTable
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("input_path", "")
dbutils.widgets.text("input_role", "scoring")
dbutils.widgets.text("batch_id", "")
dbutils.widgets.text("job_run_id", "")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
ENVIRONMENT = dbutils.widgets.get("environment").strip()
INPUT_PATH = dbutils.widgets.get("input_path").strip().lstrip("/")
INPUT_ROLE = dbutils.widgets.get("input_role").strip()
BATCH_ID = dbutils.widgets.get("batch_id").strip() or datetime.now(timezone.utc).strftime(
    "manual-%Y%m%dT%H%M%SZ"
)
JOB_RUN_ID = dbutils.widgets.get("job_run_id").strip() or BATCH_ID

if not CATALOG or not SCHEMA or not INPUT_PATH:
    raise ValueError("Set catalog, schema, and input_path")
if not re.fullmatch(r"[A-Za-z0-9_.-]+", BATCH_ID):
    raise ValueError("batch_id may contain only letters, numbers, dot, underscore, and hyphen")
if ENVIRONMENT not in {"dev", "prod"}:
    raise ValueError("environment must be dev or prod")
if INPUT_ROLE not in {"training", "scoring"}:
    raise ValueError("input_role must be training or scoring")

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}"
RAW_PATH = f"{VOLUME_ROOT}/raw/{INPUT_PATH}"
BRONZE_BATCH_PATH = f"{VOLUME_ROOT}/bronze/{ENVIRONMENT}/batches/{BATCH_ID}"
INGESTION_LEDGER_PATH = f"{VOLUME_ROOT}/bronze/{ENVIRONMENT}/ingestion_ledger"

# COMMAND ----------

raw_source = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "false")
    .load(RAW_PATH)
)
if not raw_source.take(1):
    raise ValueError(f"No input rows found at {RAW_PATH}")

source_files = raw_source.select(
    F.col("_metadata.file_path").alias("source_file"),
    F.col("_metadata.file_modification_time").alias("source_modified_ts"),
    F.col("_metadata.file_size").alias("source_size"),
).distinct()
if DeltaTable.isDeltaTable(spark, INGESTION_LEDGER_PATH):
    previous = (
        spark.read.format("delta")
        .load(INGESTION_LEDGER_PATH)
        .filter(
            (F.col("environment") == ENVIRONMENT)
            & (F.col("input_role") == INPUT_ROLE)
            & (F.col("batch_id") != BATCH_ID)
        )
        .select("source_file", "source_modified_ts", "source_size")
        .distinct()
    )
    source_files = source_files.join(
        previous, on=["source_file", "source_modified_ts", "source_size"], how="left_anti"
    )
if not source_files.take(1):
    raise ValueError(f"No unprocessed {INPUT_ROLE} files found at {RAW_PATH}")

raw_df = raw_source.join(
    source_files,
    on=(F.col("_metadata.file_path") == F.col("source_file"))
    & (F.col("_metadata.file_modification_time") == F.col("source_modified_ts"))
    & (F.col("_metadata.file_size") == F.col("source_size")),
    how="inner",
).drop("source_file", "source_modified_ts", "source_size")

bronze_df = (
    raw_df.withColumn("_batch_id", F.lit(BATCH_ID))
    .withColumn("_job_run_id", F.lit(JOB_RUN_ID))
    .withColumn("_ingest_ts", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
)

(
    bronze_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(BRONZE_BATCH_PATH)
)

rows = spark.read.format("delta").load(BRONZE_BATCH_PATH).count()
ledger_rows = (
    source_files.withColumn("environment", F.lit(ENVIRONMENT))
    .withColumn("input_role", F.lit(INPUT_ROLE))
    .withColumn("batch_id", F.lit(BATCH_ID))
    .withColumn("job_run_id", F.lit(JOB_RUN_ID))
    .withColumn("ingested_ts", F.current_timestamp())
)
ledger_rows.write.format("delta").mode("append").save(INGESTION_LEDGER_PATH)
dbutils.jobs.taskValues.set(key="batch_id", value=BATCH_ID)
dbutils.jobs.taskValues.set(key="bronze_rows", value=rows)
print(
    f"Bronze {INPUT_ROLE} batch {BATCH_ID}: {rows:,} rows from "
    f"{source_files.count():,} new file(s) at {RAW_PATH}"
)
