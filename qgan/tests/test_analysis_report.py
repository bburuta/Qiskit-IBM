from pathlib import Path

from qgan_v2.analysis.results import RunResult
from qgan_v2.analysis.report import (
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
        "qgan_v2.analysis.report.get_config_filename",
        lambda value: run_path / "config.yaml",
    )

    result = expected_battery_results([config], [], "convergence")[0]

    assert result.path == run_path
    assert result.status == "missing_training_data"
    assert result.metadata["analysis_source"] == "convergence"


def test_expected_real_hardware_result_uses_observed_budget(monkeypatch, tmp_path):
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
        "qgan_v2.analysis.report.get_config_filename",
        lambda value: run_path / "config.yaml",
    )

    result = expected_battery_results([config], [observed], "convergence")[0]

    assert result.metadata["max_iterations"] == 2


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
    }) == "out of memory"
