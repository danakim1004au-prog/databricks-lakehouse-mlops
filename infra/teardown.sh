#!/usr/bin/env bash
# Tear down the lab. The Databricks managed resource group is deleted automatically
# when the workspace resource is deleted.
set -euo pipefail

RG="${RG:-rg-dbx-churn-lab-aue}"

command -v az >/dev/null 2>&1 || {
  echo "Azure CLI (az) is required. Install it before running this script." >&2
  exit 1
}
az account show --output none >/dev/null 2>&1 || {
  echo "No Azure CLI session is active. Run 'az login' first." >&2
  exit 1
}

if [[ "$(az group exists --name "$RG")" != "true" ]]; then
  echo "==> Resource group $RG does not exist; nothing to delete."
  exit 0
fi

echo "==> Deleting resource group $RG (and the Databricks managed RG with it)"
az group delete --name "$RG" --yes --no-wait

echo "==> Delete requested (runs in background, ~5-10 min)."
echo "    Verify with: az group exists --name $RG"
