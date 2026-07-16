# Security notes

This repository is a disposable lab. Never commit Azure storage keys, Databricks tokens, connection strings, or secret-scope values.

- Store the storage account key in a Databricks secret scope as described in the README.
- Use `--auth-mode login` for Azure CLI uploads rather than putting a key in shell history.
- If a key or token is exposed, revoke or rotate it in Azure/Databricks before continuing the lab.
- The Bicep deployment creates a dedicated resource group so the full lab can be deleted after the run.

The Phase 1 notebook path uses a storage key only through the Databricks secret API. Unity Catalog-managed credentials and production access policies are outside this version's scope.
