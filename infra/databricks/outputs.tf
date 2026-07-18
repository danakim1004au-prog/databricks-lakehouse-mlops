output "catalog_name" {
  value = databricks_catalog.this.name
}

output "dev_schema" {
  value = databricks_schema.environment["churn_dev"].name
}

output "prod_schema" {
  value = databricks_schema.environment["churn_prod"].name
}

output "raw_dev_volume" {
  value = "/Volumes/${databricks_catalog.this.name}/churn_dev/raw"
}
