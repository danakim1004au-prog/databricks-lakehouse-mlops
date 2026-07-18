"""Structural safety checks for the Databricks bundle definitions."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def task_keys(job: dict) -> list[str]:
    return [task["task_key"] for task in job["tasks"]]


def test_training_and_scoring_use_distinct_operational_lifecycles():
    jobs = load_yaml(ROOT / "resources" / "churn_jobs.yml")["resources"]["jobs"]
    assert task_keys(jobs["churn_retraining"]) == [
        "bronze_ingest",
        "silver_clean",
        "gold_features",
        "train_and_register",
    ]
    assert task_keys(jobs["churn_batch_monitoring"]) == [
        "bronze_ingest",
        "silver_clean",
        "gold_features",
        "batch_inference",
        "monitor_model",
    ]
    assert task_keys(jobs["churn_model_promotion"]) == ["promote_model"]


def test_schedules_are_safe_and_production_uses_production_alias():
    bundle = load_yaml(ROOT / "databricks.yml")
    jobs = load_yaml(ROOT / "resources" / "churn_jobs.yml")["resources"]["jobs"]
    scheduled = [jobs["churn_retraining"], jobs["churn_batch_monitoring"]]
    assert all(job["schedule"]["pause_status"] == "PAUSED" for job in scheduled)
    prod_jobs = bundle["targets"]["prod"]["resources"]["jobs"]
    assert all(job["schedule"]["pause_status"] == "UNPAUSED" for job in prod_jobs.values())
    assert bundle["targets"]["dev"]["variables"]["inference_alias"] == "staging"
    assert bundle["targets"]["prod"]["variables"]["inference_alias"] == "production"


def test_jobs_have_lineage_parameters_retries_and_failure_notifications():
    jobs = load_yaml(ROOT / "resources" / "churn_jobs.yml")["resources"]["jobs"]
    for name in ["churn_retraining", "churn_batch_monitoring"]:
        job = jobs[name]
        assert job["email_notifications"]["on_failure"]
        first_task = job["tasks"][0]
        parameters = first_task["notebook_task"]["base_parameters"]
        assert parameters["batch_id"] == "{{job.run_id}}"
        assert parameters["job_run_id"] == "{{job.run_id}}"
        assert parameters["input_role"] in {"training", "scoring"}
        assert first_task["max_retries"] >= 1


def test_storage_keys_are_not_part_of_bundle_contract():
    bundle_text = (ROOT / "databricks.yml").read_text(encoding="utf-8")
    resource_text = (ROOT / "resources" / "churn_jobs.yml").read_text(encoding="utf-8")
    assert "storage-account-key" not in bundle_text + resource_text
    assert "secret_scope" not in bundle_text + resource_text


def test_serving_endpoint_scales_to_zero_and_is_environment_tagged():
    serving = load_yaml(ROOT / "serving" / "databricks.yml")
    endpoint = serving["resources"]["model_serving_endpoints"]["churn_classifier"]
    entity = endpoint["config"]["served_entities"][0]
    tags = {item["key"]: item["value"] for item in endpoint["tags"]}
    assert entity["workload_size"] == "Small"
    assert entity["scale_to_zero_enabled"] is True
    assert tags["environment"] == "${var.environment}"
