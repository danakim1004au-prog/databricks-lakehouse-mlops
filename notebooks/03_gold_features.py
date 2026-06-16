# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Gold: ML-ready feature table
# MAGIC Business-level feature engineering. Output is the single source of truth for
# MAGIC training (notebook 04) and batch inference (notebook 05).

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

SILVER_PATH = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net/telco_churn"
GOLD_PATH   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/churn_features"

# COMMAND ----------

silver = spark.read.format("delta").load(SILVER_PATH)

gold = (
    silver
    .withColumn(
        "tenure_bucket",
        F.when(F.col("tenure") <= 6, "0-6m")
         .when(F.col("tenure") <= 24, "7-24m")
         .when(F.col("tenure") <= 48, "25-48m")
         .otherwise("49m+"),
    )
    .withColumn("avg_monthly_spend", F.round(F.col("TotalCharges") / F.greatest(F.col("tenure"), F.lit(1)), 2))
    .withColumn("spend_delta", F.round(F.col("MonthlyCharges") - F.col("avg_monthly_spend"), 2))
    .withColumn("is_month_to_month", (F.col("Contract") == "Month-to-month").cast("int"))
    .withColumn("is_fiber", (F.col("InternetService") == "Fiber optic").cast("int"))
    .withColumn("is_echeck", (F.col("PaymentMethod") == "Electronic check").cast("int"))
    .withColumn("has_tech_support", (F.col("TechSupport") == "Yes").cast("int"))
    .withColumn("is_paperless", (F.col("PaperlessBilling") == "Yes").cast("int"))
    .select(
        "customerID", "SeniorCitizen", "tenure", "tenure_bucket",
        "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "spend_delta",
        "is_month_to_month", "is_fiber", "is_echeck", "has_tech_support",
        "is_paperless", "churn_label",
    )
)

gold.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(GOLD_PATH)

display(gold.groupBy("tenure_bucket").agg(F.avg("churn_label").alias("churn_rate"), F.count("*").alias("n")))
