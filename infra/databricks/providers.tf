provider "databricks" {
  host                        = var.workspace_host
  azure_workspace_resource_id = var.workspace_resource_id
}

provider "databricks" {
  alias      = "accounts"
  host       = "https://accounts.azuredatabricks.net"
  account_id = var.account_id
}
