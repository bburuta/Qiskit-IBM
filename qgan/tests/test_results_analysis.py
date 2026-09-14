from pathlib import Path

import pytest

from qgan_v2.analysis.results import RunResult, is_completed
from qgan_v2.analysis.workflow import (
    ResultsAnalysis,
    ResultsData,
    expected_battery_results,
    incomplete_limitation,
    scientific_config_key,
)


def make_config(*, device="CPU", randomness=0):
    return {
        "run": {"id": "run", "seed": 2},
        "experiment": {
            "implementation": "ang",
            "execution_type": "noisy",
            "gradient_method": "PSR",
            "n_qubits": 4,
        },
        "implementation": {
            "name": "qml_torch",
            "discriminator_packing": "separate",
        },
        "encoding": {
            "random_circuit": 7,
            "randomness": randomness,
            "batch_size": 4,
            "eval_batch_size": 8,
            "eval_method": "gradient",
        },
        "training": {
            "learning_rate": 0.01,
            "max_iterations": 1000,
            "gen_iterations": 1,
            "disc_iterations": 1,
        },
        "backend": {
            "precision": 0.01,
            "simulator": {
                "device": device,
                "noiseless_method": "statevector",
                "noisy_method": "density_matrix",
                "noisy_backend_mapping": "hardware",
            },
        },
    }


def test_scientific_config_key_ignores_device_and_unused_random_circuit():
    cpu = make_config(device="CPU", randomness=0)
    gpu = make_config(device="GPU", randomness=0)
    gpu["encoding"]["random_circuit"] = 99

    assert scientific_config_key(cpu) == scientific_config_key(gpu)


def test_expected_battery_results_creates_a_missing_result(monkeypatch, tmp_path):
    config = make_config()
    run_path = tmp_path / "run"
    monkeypatch.setattr(
        "qgan_v2.analysis.workflow.get_config_filename",
        lambda value: run_path / "config.yaml",
    )

    result = expected_battery_results([config], [], "convergence")[0]

    assert result.path == run_path
    assert result.status == "missing_training_data"
    assert result.metadata["analysis_source"] == "convergence"


def test_expected_result_uses_battery_budget_without_hardware_special_case(
    monkeypatch,
    tmp_path,
):
    config = make_config()
    config["experiment"]["execution_type"] = "real"
    run_path = tmp_path / "run"
    observed = RunResult(
        path=run_path,
        config=config,
        metadata={"execution_type": "real", "max_iterations": 1000},
        eval={0: 1.0, 1: 0.5},
        gloss={},
        dloss={},
        times={},
    )
    monkeypatch.setattr(
        "qgan_v2.analysis.workflow.get_config_filename",
        lambda value: run_path / "config.yaml",
    )

    result = expected_battery_results([config], [observed], "convergence")[0]

    assert result.metadata["max_iterations"] == 1000
    assert not is_completed(result)


def test_expected_result_is_complete_at_or_above_battery_budget(monkeypatch, tmp_path):
    config = make_config()
    config["training"]["max_iterations"] = 5
    run_path = tmp_path / "run"
    observed = RunResult(
        path=run_path,
        config=config,
        metadata={"max_iterations": 1000, "simulator_device": "GPU"},
        eval={epoch: float(epoch) for epoch in range(6)},
        gloss={},
        dloss={},
        times={},
    )
    monkeypatch.setattr(
        "qgan_v2.analysis.workflow.get_config_filename",
        lambda value: run_path / "config.yaml",
    )

    result = expected_battery_results([config], [observed], "timing")[0]

    assert result.metadata["max_iterations"] == 5
    assert result.metadata["simulator_device"] == "CPU"
    assert is_completed(result)


def test_default_feasibility_sources_use_one_convergence_and_three_timing_batteries(
    monkeypatch,
    tmp_path,
):
    data = ResultsData(tmp_path, [], [], [], [], [], [])
    analysis = ResultsAnalysis(data)
    expanded = []

    def record_batteries(paths, **kwargs):
        expanded.append([Path(path).name for path in paths])
        return []

    monkeypatch.setattr("qgan_v2.analysis.workflow.battery_configs", record_batteries)

    results, convergence_battery, timing_batteries = analysis._feasibility_results()

    assert results == []
    assert convergence_battery.name == "train_conv_gpu.yaml"
    assert expanded == [["train_conv_gpu.yaml"], [
        "train_times_cpu.yaml",
        "train_times_gpu.yaml",
        "train_times_rh.yaml",
    ]]
    assert [path.name for path in timing_batteries] == expanded[1]


def test_incomplete_limitation_classifies_known_cases():
    assert incomplete_limitation({
        "analysis_source": "timing",
        "execution_type": "real",
    }) == "execution time unavailable"
    assert incomplete_limitation({
        "analysis_source": "convergence",
        "execution_type": "noisy",
        "gradient_method": "SPSA",
        "n_qubits": 16,
        "simulator_device": "CPU",
    }) == "out of memory"
    assert incomplete_limitation({
        "analysis_source": "convergence",
        "execution_type": "noisy",
        "gradient_method": "SPSA",
        "n_qubits": 16,
        "simulator_device": "GPU",
    }) == "time-expensive"


def test_randomness_sweep_filters_candidates_before_grouping(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    runs = [
        RunResult(tmp_path / preset, {}, {"preset": preset}, {}, {}, {}, {})
        for preset in ("amp", "ang", "base")
    ]
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], runs, [], [], []))
    captured = {}

    def capture_groups(results, **kwargs):
        captured["presets"] = [run.metadata["preset"] for run in results]
        return {}

    monkeypatch.setattr(
        "qgan_v2.analysis.workflow.factor_sweep_groups",
        capture_groups,
    )

    selected = analysis.randomness_sweep(filters={"preset": "base"})

    assert selected == []
    assert captured["presets"] == ["base"]


def test_render_randomness_output_uses_one_run_and_best_parameters(
    monkeypatch,
    tmp_path,
):
    metadata = {
        "preset": "base",
        "implementation": "qml_torch",
        "execution_type": "noiseless",
        "gradient_method": "PSR",
        "n_qubits": 4,
        "randomness": 0.5,
        "seed": 2,
    }
    run = RunResult(tmp_path / "selected", {}, metadata, {}, {}, {}, {})
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], [run], [], [], []))
    plotted = {}

    def plot_output(config_file, **kwargs):
        plotted.update(config_file=config_file, **kwargs)
        return object(), {
            "run_id": "selected",
            "random_seed": kwargs["random_seed"],
            "num_outputs": kwargs["num_outputs"],
        }

    monkeypatch.setattr(
        "qgan_v2.visualization.plot_generated_output",
        plot_output,
    )
    monkeypatch.setattr("qgan_v2.analysis.workflow.display_rows", lambda rows: rows)
    monkeypatch.setattr(analysis, "_finish", lambda fig, stem: None)

    record = analysis.render_randomness_output(
        filters={"preset": "base", "randomness": 0.5, "seed": 2},
        random_seed=None,
        output_count=4,
    )

    assert plotted == {
        "config_file": run.path / "config.yaml",
        "parameter_set": "best",
        "random_seed": None,
        "num_outputs": 4,
    }
    assert record["training_seed"] == 2
    assert record["random_input_applied"] is True
    assert record["num_outputs"] == 4
