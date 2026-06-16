#!/usr/bin/env bash
# Tear down the lab. The Databricks managed resource group is deleted automatically
# when the workspace resource is deleted.
set -euo pipefail

RG="rg-dbx-churn-lab-aue"

echo "==> Deleting resource group $RG (and the Databricks managed RG with it)"
az group delete --name "$RG" --yes --no-wait

echo "==> Delete requested (runs in background, ~5-10 min)."
echo "    Verify with: az group exists --name $RG"
