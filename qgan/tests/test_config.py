import pytest

from qgan_v2.config.loader import prepare_run_config
from qgan_v2.config.validation import ConfigValidationError


def base_config():
    return {
        "implementation": {
            "name": "qml_torch",
            "discriminator_packing": "separate",
        },
        "run": {
            "id": None,
            "label": None,
            "data_path": "qgan/data/test",
            "seed": 0,
            "device": "CPU",
        },
        "experiment": {
            "implementation": "base",
            "execution_type": "noiseless",
            "n_qubits": 3,
            "gradient_method": "PSR",
        },
        "training": {
            "max_iterations": 1,
            "gen_iterations": 1,
            "disc_iterations": 1,
            "init_scale": 0.1,
            "learning_rate": 0.005,
            "print_every": 1,
            "checkpoint_every": 1,
        },
        "dataset": {
            "id": None,
            "reset": False,
            "type": "quantum",
            "source": "specific_distribution",
            "parameters": {},
        },
        "circuits": {
            "reset": False,
            "generator": "real_amplitudes",
            "discriminator": "efficient_su2",
        },
        "backend": {
            "reset": False,
            "save_backend_file": False,
            "precision": 0.5,
            "transpilation": {
                "optimization_level": 1,
                "layout_method": "trivial",
                "routing_method": "basic",
            },
            "simulator": {
                "device": "CPU",
                "data_type": "double",
                "noiseless_method": "statevector",
                "noisy_method": "density_matrix",
                "noisy_backend_mapping": "hardware",
                "max_parallel_threads": 0,
                "max_parallel_experiments": 1,
                "max_parallel_shots": 0,
                "gpu": {
                    "cuStateVec_enable": True,
                    "batched_shots_gpu": False,
                    "blocking_enable": False,
                    "runtime_parameter_bind_enable": False,
                },
            },
            "real": {
                "id": None,
                "name": "ibm_basquecountry",
                "reset_info": False,
                "confirm_runtime_execution": False,
                "estimator": {
                    "resilience_level": 1,
                    "dynamical_decoupling": {
                        "enable": True,
                    },
                },
            },
        },
        "encoding": {
            "type": "direct_circuit",
            "contrast": 1,
            "random_circuit": 1,
            "randomness": 0.1,
            "batch_size": 1,
            "eval_batch_size": 1,
            "max_parallel_threads": 1,
            "eval_method": "kl",
        },
    }


def test_prepare_run_config_accepts_current_backend_options():
    config = prepare_run_config(base_config())

    assert config["backend"]["real"]["confirm_runtime_execution"] is False
    assert config["backend"]["save_backend_file"] is False


def test_prepare_run_config_keeps_random_circuit_separate_from_randomness():
    raw_config = base_config()
    raw_config["encoding"]["random_circuit"] = 0
    raw_config["encoding"]["randomness"] = 0.1

    config = prepare_run_config(raw_config)

    assert config["encoding"]["random_circuit"] == 0
    assert config["encoding"]["randomness"] == 0.1
    assert "rand0.1" in config["run"]["id"]
    assert "rc0" not in config["run"]["id"]


@pytest.mark.parametrize("random_circuit", [0, 1, 2, 3])
def test_prepare_run_config_accepts_random_circuit_types(random_circuit):
    raw_config = base_config()
    raw_config["encoding"]["random_circuit"] = random_circuit

    config = prepare_run_config(raw_config)

    assert config["encoding"]["random_circuit"] == random_circuit


@pytest.mark.parametrize("random_circuit", [False, True, 4])
def test_prepare_run_config_rejects_invalid_random_circuit_types(random_circuit):
    raw_config = base_config()
    raw_config["encoding"]["random_circuit"] = random_circuit

    with pytest.raises(ConfigValidationError):
        prepare_run_config(raw_config)
