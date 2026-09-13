from pathlib import Path

import pytest

import qgan_v2.experiments as experiments


def test_single_experiment_loads_an_existing_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "training_data.pth"
    checkpoint.touch()
    monkeypatch.setattr(
        experiments,
        "load_run_config",
        lambda path: {"run": {"id": "example"}},
    )
    monkeypatch.setattr(
        experiments,
        "get_training_data_filename",
        lambda config: checkpoint,
    )
    monkeypatch.setattr(
        experiments,
        "load_training_data_file",
        lambda path: {"loaded": path},
    )

    experiment = experiments.SingleExperiment.select(tmp_path / "config.yaml")

    assert experiment.run_or_load() == {"loaded": checkpoint}
    assert experiment.summary()["run_id"] == "example"


def test_single_experiment_reports_a_missing_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "missing.pth"
    monkeypatch.setattr(
        experiments,
        "load_run_config",
        lambda path: {"run": {"id": "missing"}},
    )
    monkeypatch.setattr(
        experiments,
        "get_training_data_filename",
        lambda config: checkpoint,
    )
    experiment = experiments.SingleExperiment.select(tmp_path / "config.yaml")

    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        experiment.run_or_load()


def test_battery_rows_are_limited(monkeypatch, tmp_path):
    paths = [tmp_path / name / "config.yaml" for name in ("one", "two")]
    battery = experiments.BatteryExperiments(tmp_path / "battery.yaml")
    battery.valid_config_files = paths
    battery.states = {"one": object()}
    monkeypatch.setattr(
        experiments,
        "load_run_config",
        lambda path: {"run": {"id": path.parent.name}},
    )
    monkeypatch.setattr(
        experiments,
        "get_training_data_filename",
        lambda config: Path("does-not-exist"),
    )

    assert [row["run_id"] for row in battery.config_rows(limit=1)] == ["one"]
