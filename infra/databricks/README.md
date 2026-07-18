# Unity Catalog bootstrap

This Terraform layer completes the Azure Bicep deployment by attaching the workspace to an
existing regional Unity Catalog metastore and provisioning:

- an Access Connector-backed storage credential;
- external locations for raw, bronze, silver, and gold containers;
- `churn_dev` and `churn_prod` schemas;
- one external volume per medallion layer and environment;
- least-privilege catalog, schema, and volume grants for the pipeline principal.

Account-admin and workspace-admin permissions are required for the first apply. Copy
`terraform.tfvars.example` to an untracked `terraform.tfvars`, fill the values from the Bicep
outputs and Databricks account console, then run:

```bash
terraform init
terraform fmt -check
terraform plan
terraform apply
```

Terraform state can contain workspace identifiers. Store it in an encrypted remote backend for
shared or long-lived use; never commit local state files.
