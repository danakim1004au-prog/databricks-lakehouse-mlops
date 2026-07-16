# Security notes

This repository is a disposable lab. Never commit Azure storage keys, Databricks tokens, connection strings, or secret-scope values.

- Store the storage account key in a Databricks secret scope as described in the README.
- Prefer short-lived browser-based OAuth for local Databricks CLI access; do not commit PATs or CLI profiles.
- Use `--auth-mode login` for Azure CLI uploads rather than putting a key in shell history.
- If a key or token is exposed, revoke or rotate it in Azure/Databricks before continuing the lab.
- The Bicep deployment creates a dedicated resource group so the full lab can be deleted after the run.

The notebook path reads the storage key only through the Databricks secret API. A long-lived environment should replace this lab shortcut with a Unity Catalog external location backed by the Access Connector's managed identity, then apply least-privilege job and endpoint permissions.
