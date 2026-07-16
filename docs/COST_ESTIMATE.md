# Cost estimate — same-day deploy/teardown (australiaeast, indicative June 2026 rates)

Assumption: one lab session uses a single-node cluster for roughly three hours and the resource group is deleted the same day. Azure and Databricks prices vary by subscription, exchange rate, region, and discount.

| Item | Unit price (approx) | Usage | Cost |
|---|---|---|---|
| Databricks workspace (Premium) | No separate workspace line item in this estimate | — | AUD 0 |
| DBU (All-Purpose Compute, Premium) | ~USD 0.55/DBU | DS3_v2 single node = 0.75 DBU/h × 3h | ~USD 1.24 |
| VM (Standard_DS3_v2, 4 vCPU 14GB) | ~AUD 0.42/h | 3h × 1 node | ~AUD 1.26 |
| ADLS Gen2 (Standard LRS) | ~AUD 0.03/GB/month | <1GB, 1 day | ~AUD 0.01 |
| Access Connector | no separate service charge in this estimate | — | AUD 0 |
| Transactions / egress | negligible | — | <AUD 0.10 |

**Estimated total: ~AUD 3.5–5 per session (approximately USD 2.5–3.5 at the exchange rate used for this estimate).**

Credits or a free trial can reduce the billed amount, but they do not make the underlying compute free. Confirm the actual estimate in the Azure and Databricks billing portals before running a long session.

## Cost traps to watch

1. **No cluster auto-terminate** — set 30 minutes when creating the cluster. A cluster left running overnight can cost materially more than this lab estimate.
2. **Multi-node default** — cluster creation defaults to 2–8 worker autoscale. Always pick "Single node".
3. **Orphaned managed RG** — the managed RG is auto-deleted with the workspace, but if `az group delete`
   fails and leaves it behind, a NAT GW / IP inside it can keep billing.
   After teardown, check with `az group list --query "[?contains(name,'dbx-churn')]"`.
4. **Premium SKU DBU rate** — typically higher than Standard. This lab keeps Premium for the workspace configuration and Unity Catalog-compatible runtime; keep sessions short.

## To save more

- Spot VMs cut VM cost 60–80% (fine for a lab).
- Trial or promotional credits may cover DBU charges, but VM and other subscription charges can still apply.
