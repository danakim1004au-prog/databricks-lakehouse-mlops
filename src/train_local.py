"""Compatibility entrypoint; implementation lives in the installed package."""

from databricks_lakehouse_mlops.train_local import main

if __name__ == "__main__":
    main()
