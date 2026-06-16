# Cost Estimate — same-day deploy/teardown (australiaeast, 2026-06 pricing)

Assumption: one lab session = ~3 hours of cluster runtime, RG deleted the same day.

| Item | Unit price (approx) | Usage | Cost |
|---|---|---|---|
| Databricks workspace (Premium) | Workspace itself is free — billed only on compute | — | $0 |
| DBU (All-Purpose Compute, Premium) | ~USD 0.55/DBU | DS3_v2 single node = 0.75 DBU/h × 3h | ~USD 1.24 |
| VM (Standard_DS3_v2, 4 vCPU 14GB) | ~AUD 0.42/h | 3h × 1 node | ~AUD 1.26 |
| ADLS Gen2 (Standard LRS) | ~AUD 0.03/GB/month | <1GB, 1 day | ~AUD 0.01 |
| Access Connector | free | — | $0 |
| Transactions / egress | negligible | — | <AUD 0.10 |

**Total: ~AUD 3.5–5 (USD 2.5–3.5) per session**

In practice a short same-day run can land at effectively $0 on a free-trial / credit subscription (VM cost only).

## Cost traps to watch

1. **No cluster auto-terminate** — always set 30 min. Left running overnight is ~AUD 25/day.
2. **Multi-node default** — cluster creation defaults to 2–8 worker autoscale. Always pick "Single node".
3. **Orphaned managed RG** — the managed RG is auto-deleted with the workspace, but if `az group delete`
   fails and leaves it behind, a NAT GW / IP inside it can keep billing.
   After teardown, check with `az group list --query "[?contains(name,'dbx-churn')]"`.
4. **Premium SKU DBU rate** — ~50% pricier than Standard. Keep Premium (needed for the UC/RBAC demo) but keep sessions short.

## To save more

- Spot VMs cut VM cost 60–80% (fine for a lab).
- A 14-day Databricks free-trial workspace has $0 DBU charges (VM cost only).
