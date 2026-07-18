"""Smoke-test the packaged local training entrypoint."""

from pathlib import Path

from data.generate_churn_data import generate_dataset
from databricks_lakehouse_mlops import train_local


def test_local_training_smoke(tmp_path: Path, monkeypatch, capsys):
    data_path = tmp_path / "training.csv"
    generate_dataset(n_rows=1_000).to_csv(data_path, index=False)
    monkeypatch.setattr(train_local, "DATA", data_path)

    train_local.main()

    output = capsys.readouterr().out
    assert "winner=" in output
    assert "roc_auc=" in output
