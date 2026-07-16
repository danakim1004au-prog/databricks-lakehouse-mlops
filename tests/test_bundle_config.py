"""Structural checks for the Databricks bundle definitions."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_retraining_job_has_ordered_operational_lifecycle():
    jobs = load_yaml(ROOT / "resources" / "churn_jobs.yml")["resources"]["jobs"]
    tasks = jobs["churn_retraining"]["tasks"]

    assert [task["task_key"] for task in tasks] == [
        "bronze_ingest",
        "silver_clean",
        "gold_features",
        "train_and_register",
        "batch_inference",
        "monitor_model",
    ]

    for previous, current in zip(tasks, tasks[1:]):
        assert current["depends_on"] == [{"task_key": previous["task_key"]}]


def test_schedules_are_safe_by_default_and_explicit_in_prod():
    bundle = load_yaml(ROOT / "databricks.yml")
    jobs = load_yaml(ROOT / "resources" / "churn_jobs.yml")["resources"]["jobs"]

    assert all(job["schedule"]["pause_status"] == "PAUSED" for job in jobs.values())
    prod_jobs = bundle["targets"]["prod"]["resources"]["jobs"]
    assert all(job["schedule"]["pause_status"] == "UNPAUSED" for job in prod_jobs.values())


def test_serving_endpoint_scales_to_zero():
    serving = load_yaml(ROOT / "serving" / "databricks.yml")
    endpoints = serving["resources"]["model_serving_endpoints"]
    entity = endpoints["churn_classifier"]["config"]["served_entities"][0]

    assert entity["workload_size"] == "Small"
    assert entity["scale_to_zero_enabled"] is True
