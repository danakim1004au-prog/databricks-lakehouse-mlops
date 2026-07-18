locals {
  containers = toset(["raw", "bronze", "silver", "gold"])
  schemas    = toset(["churn_dev", "churn_prod"])
}

resource "databricks_metastore_assignment" "workspace" {
  provider     = databricks.accounts
  metastore_id = var.metastore_id
  workspace_id = var.workspace_id
}

resource "databricks_storage_credential" "lake" {
  name = "churn-mlops-managed-identity"
  azure_managed_identity {
    access_connector_id = var.access_connector_id
  }
  comment    = "Managed identity credential for the churn MLOps external volumes"
  depends_on = [databricks_metastore_assignment.workspace]
}

resource "databricks_external_location" "container" {
  for_each        = local.containers
  name            = "churn-mlops-${each.key}"
  url             = "abfss://${each.key}@${var.storage_account_name}.dfs.core.windows.net/"
  credential_name = databricks_storage_credential.lake.id
  comment         = "${each.key} medallion container managed by Terraform"
}

resource "databricks_catalog" "this" {
  name          = var.catalog_name
  comment       = "Environment-isolated churn MLOps catalog"
  force_destroy = false
  depends_on    = [databricks_metastore_assignment.workspace]
}

resource "databricks_schema" "environment" {
  for_each     = local.schemas
  catalog_name = databricks_catalog.this.name
  name         = each.key
  comment      = "Churn MLOps ${each.key} schema"
}

resource "databricks_volume" "environment" {
  for_each = {
    for pair in setproduct(local.schemas, local.containers) : "${pair[0]}-${pair[1]}" => {
      schema    = pair[0]
      container = pair[1]
    }
  }
  name             = each.value.container
  catalog_name     = databricks_catalog.this.name
  schema_name      = databricks_schema.environment[each.value.schema].name
  volume_type      = "EXTERNAL"
  storage_location = "${databricks_external_location.container[each.value.container].url}${each.value.schema}/"
  comment          = "${each.value.container} external volume for ${each.value.schema}"
}

resource "databricks_grants" "catalog" {
  catalog = databricks_catalog.this.name
  grant {
    principal  = var.data_principal
    privileges = ["USE_CATALOG"]
  }
}

resource "databricks_grants" "schema" {
  for_each = local.schemas
  schema   = "${databricks_catalog.this.name}.${databricks_schema.environment[each.key].name}"
  grant {
    principal = var.data_principal
    privileges = [
      "USE_SCHEMA",
      "CREATE_MODEL",
      "CREATE_VOLUME",
    ]
  }
}

resource "databricks_grants" "volume" {
  for_each = databricks_volume.environment
  volume   = "${each.value.catalog_name}.${each.value.schema_name}.${each.value.name}"
  grant {
    principal  = var.data_principal
    privileges = ["READ_VOLUME", "WRITE_VOLUME"]
  }
}
