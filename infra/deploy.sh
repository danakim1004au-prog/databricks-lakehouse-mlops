#!/usr/bin/env bash
# Deploy the Databricks Lakehouse MLOps lab.
# Usage: ./infra/deploy.sh [location]
set -euo pipefail

LOCATION="${1:-${LOCATION:-australiaeast}}"
RG="${RG:-rg-dbx-churn-lab-aue}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v az >/dev/null 2>&1 || {
  echo "Azure CLI (az) is required. Install it before running this script." >&2
  exit 1
}
az account show --output none >/dev/null 2>&1 || {
  echo "No Azure CLI session is active. Run 'az login' first." >&2
  exit 1
}

echo "==> Validating Bicep template"
az bicep build --file "$SCRIPT_DIR/main.bicep" --stdout >/dev/null

echo "==> Creating resource group $RG in $LOCATION"
az group create --name "$RG" --location "$LOCATION" \
  --tags project=databricks-lakehouse-mlops env=lab delete-after=same-day --output none

DEPLOYMENT_NAME="dbx-churn-$(date -u +%Y%m%d%H%M%S)"
echo "==> Deploying Bicep template (Databricks workspace takes several minutes)"
OUTPUT="$(az deployment group create \
  --resource-group "$RG" \
  --name "$DEPLOYMENT_NAME" \
  --template-file "$SCRIPT_DIR/main.bicep" \
  --query "properties.outputs" --output json)"

echo "$OUTPUT"

echo ""
echo "==> Done. Next steps:"
echo "    1. Configure Unity Catalog volumes with infra/databricks Terraform"
echo "    2. Upload generated training/scoring CSVs to the raw external volume"
echo "    3. Validate and deploy the Databricks bundle"
echo "    4. When finished: RG=$RG ./infra/teardown.sh"
