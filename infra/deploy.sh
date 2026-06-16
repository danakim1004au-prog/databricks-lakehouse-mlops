#!/usr/bin/env bash
# Deploy the Databricks Lakehouse MLOps lab.
# Usage: ./infra/deploy.sh [location]
set -euo pipefail

LOCATION="${1:-australiaeast}"
RG="rg-dbx-churn-lab-aue"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating resource group $RG in $LOCATION"
az group create --name "$RG" --location "$LOCATION" \
  --tags project=databricks-lakehouse-mlops env=lab delete-after=same-day --output none

echo "==> Deploying Bicep template (Databricks workspace takes ~3-5 min)"
az deployment group create \
  --resource-group "$RG" \
  --name "dbx-churn-$(date +%Y%m%d%H%M)" \
  --template-file "$SCRIPT_DIR/main.bicep" \
  --query "properties.outputs" --output json

echo ""
echo "==> Done. Next steps:"
echo "    1. Open the workspace URL above and upload data/telco_churn.csv to the 'raw' container"
echo "       (az storage blob upload --account-name <storage> -c raw -f data/telco_churn.csv -n telco_churn.csv --auth-mode login)"
echo "    2. Import notebooks/ into the workspace, attach a single-node cluster"
echo "    3. When finished: ./infra/teardown.sh"
