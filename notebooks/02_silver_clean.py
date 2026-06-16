# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver: clean, dedupe, type, validate
# MAGIC Handles the three dirt patterns injected upstream: blank `TotalCharges`,
# MAGIC duplicate rows, mixed-case `Contract`. Ends with hard data-quality gates.

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.text("storage_account", "stdbxchurnamym3m")
dbutils.widgets.text("storage_key", "PASTE_KEY_HERE")

STORAGE_ACCOUNT = dbutils.widgets.get("storage_account")
STORAGE_KEY     = dbutils.widgets.get("storage_key")

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    STORAGE_KEY,
)

BRONZE_PATH = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net/telco_churn"
SILVER_PATH = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net/telco_churn"

# COMMAND ----------

bronze = spark.read.format("delta").load(BRONZE_PATH)

contract_map = {"MONTH-TO-MONTH": "Month-to-month", "ONE YEAR": "One year", "TWO YEAR": "Two year"}

silver = (
    bronze
    .dropDuplicates(["customerID"])
    # normalise mixed-case Contract values
    .withColumn(
        "Contract",
        F.coalesce(
            F.create_map([F.lit(x) for kv in contract_map.items() for x in kv])[F.col("Contract")],
            F.col("Contract"),
        ),
    )
    # blank TotalCharges (tenure==0) -> 0.0
    .withColumn("TotalCharges", F.when(F.trim("TotalCharges") == "", "0").otherwise(F.col("TotalCharges")))
    .withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int"))
    .withColumn("tenure", F.col("tenure").cast("int"))
    .withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double"))
    .withColumn("TotalCharges", F.col("TotalCharges").cast("double"))
    .withColumn("churn_label", F.when(F.col("Churn") == "Yes", 1).otherwise(0))
    .drop("_ingest_ts", "_source_file", "Churn")
)

# COMMAND ----------

# MAGIC %md ## Data-quality gates — fail the job rather than poison Gold

# COMMAND ----------

total = silver.count()
assert total > 9_000, f"Row count collapsed: {total}"
assert silver.filter(F.col("customerID").isNull()).count() == 0, "Null customerIDs"
assert silver.select("customerID").distinct().count() == total, "Duplicates survived"
assert silver.filter(F.col("TotalCharges").isNull()).count() == 0, "TotalCharges cast produced nulls"
bad_contract = silver.filter(~F.col("Contract").isin("Month-to-month", "One year", "Two year")).count()
assert bad_contract == 0, f"{bad_contract} unnormalised Contract values"

silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(SILVER_PATH)
print(f"Silver rows: {total:,} (all DQ gates passed)")
