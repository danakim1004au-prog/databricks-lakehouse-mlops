# Cost estimate — same-day deploy/teardown (australiaeast, indicative July 2026 assumptions)

Assumption: one lab session uses a single-node cluster for roughly three hours and the resource group is deleted the same day. Azure and Databricks prices vary by subscription, exchange rate, region, and discount.

| Item | Unit price (approx) | Usage | Cost |
|---|---|---|---|
| Databricks workspace (Premium) | No separate workspace line item in this estimate | — | AUD 0 |
| Databricks Jobs Compute DBUs | Check current Databricks pricing for the selected runtime/SKU | One single-node job cluster for the measured run duration | subscription-dependent |
| VM (Standard_D4ds_v5, 4 vCPU 16GB) | Check the current Azure calculator | 3h × 1 node | subscription-dependent |
| ADLS Gen2 (Standard LRS) | ~AUD 0.03/GB/month | <1GB, 1 day | ~AUD 0.01 |
| Access Connector | no separate service charge in this estimate | — | AUD 0 |
| Unity Catalog metadata and Terraform state | no separate compute line item in this estimate | small lab metadata | AUD 0 |
| Transactions / egress | negligible | — | <AUD 0.10 |

**Working budget: AUD 5–15 per interactive session.** This is a planning guardrail, not a quote; it leaves room for job-cluster start-up time, regional VM/DBU differences, transactions, and exchange-rate movement.

Credits or a free trial can reduce the billed amount, but they do not make the underlying compute free. Confirm the actual estimate in the Azure and Databricks billing portals before running a long session.

## Cost traps to watch

1. **No cluster auto-terminate** — set 30 minutes when creating the cluster. A cluster left running overnight can cost materially more than this lab estimate.
2. **Multi-node default** — cluster creation defaults to 2–8 worker autoscale. Always pick "Single node".
3. **Orphaned managed RG** — the managed RG is auto-deleted with the workspace, but if `az group delete`
   fails and leaves it behind, a NAT GW / IP inside it can keep billing.
   After teardown, check with `az group list --query "[?contains(name,'dbx-churn')]"`.
4. **Premium SKU DBU rate** — typically higher than Standard. This lab keeps Premium for the workspace configuration and Unity Catalog-compatible runtime; keep sessions short.
5. **Unpaused production schedules** — the bundle's `dev` target pauses both jobs. The `prod` target enables daily scoring/monitoring and weekly retraining, so it creates recurring job-cluster spend.
6. **Model Serving** — the optional serving bundle is not included in the session estimate. It uses a Small scale-to-zero endpoint, but active requests and endpoint build/warm-up time are billed separately.

## To save more

- Spot VMs cut VM cost 60–80% (fine for a lab).
- Trial or promotional credits may cover DBU charges, but VM and other subscription charges can still apply.
