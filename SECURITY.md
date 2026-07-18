# Security policy and lab boundary

This repository provisions a disposable portfolio lab, but its default data path does not use
storage account keys. Azure Bicep disables shared-key access and creates an Access Connector;
the Databricks Terraform layer turns that identity into a Unity Catalog storage credential,
external locations, and environment-specific external volumes.

## Credential rules

- Use Azure CLI and Databricks browser-based OAuth locally. Do not create long-lived PATs.
- Never commit Azure credentials, Databricks profiles, Terraform state, `tfvars`, or plan files.
- Store CI workspace credentials only in GitHub encrypted secrets. Prefer workload identity/OIDC
  when the workspace supports it.
- Production jobs should run as a dedicated service principal granted only `USE CATALOG`,
  `USE SCHEMA`, model permissions, and the required volume privileges.
- Rotate or revoke a credential immediately if it appears in logs, screenshots, shell history,
  Git history, MLflow artifacts, or notebook output.

## Network boundary

`allowPublicNetwork=true` remains the reproducible lab default. A long-lived environment should
set it to false only after private endpoints, private DNS, VNet injection, and an approved egress
path are configured. Disabling public access without those dependencies makes the lab unreachable.

## CI and supply chain

- GitHub Actions are pinned to commit SHAs.
- Dependabot tracks Python, Actions, and Terraform updates.
- CI runs CodeQL, `pip-audit`, coverage, lint, Bicep compilation, Terraform formatting, shell
  syntax checks, and optional authenticated Databricks Bundle validation.
- Review dependency updates and refresh `requirements-dev.lock` deliberately; never merge an
  automated update solely because CI is green.

## Reporting

Do not open a public issue containing a live credential or exploitable tenant detail. Revoke the
credential first, then use GitHub's private vulnerability-reporting channel if it is enabled.
