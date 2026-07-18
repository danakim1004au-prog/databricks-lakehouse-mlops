variable "workspace_host" {
  description = "Azure Databricks workspace URL, including https://"
  type        = string
}

variable "workspace_resource_id" {
  description = "Azure resource ID of the Databricks workspace"
  type        = string
}

variable "workspace_id" {
  description = "Numeric Databricks workspace ID used for metastore assignment"
  type        = number
}

variable "account_id" {
  description = "Databricks account ID"
  type        = string
}

variable "metastore_id" {
  description = "Existing regional Unity Catalog metastore ID"
  type        = string
}

variable "access_connector_id" {
  description = "Azure resource ID of the Databricks Access Connector"
  type        = string
}

variable "storage_account_name" {
  description = "ADLS Gen2 account created by the Bicep deployment"
  type        = string
}

variable "catalog_name" {
  description = "Unity Catalog catalog for this project"
  type        = string
  default     = "churn_mlops"
}

variable "data_principal" {
  description = "Databricks group or service principal granted pipeline data access"
  type        = string
}
