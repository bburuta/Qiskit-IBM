import pytest

from qgan_v2.config.battery import (
    build_config_combinations,
    build_config_from_options,
    validate_unique_config_filenames,
)
from qgan_v2.config.loader import prepare_run_config
from qgan_v2.config.validation import (
    ConfigValidationError,
    RuntimeValidationError,
    validate_backend_capacity,
    validate_config,
    validate_loaded_dataset,
    warn_simulation_memory,
)


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
                "info_storage": "shared",
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


def test_battery_config_preserves_explicit_real_backend_id():
    raw_config = base_config()
    raw_config["backend"]["real"]["id"] = "ibm_basquecountry_new"

    config = build_config_from_options(raw_config, {"experiment.execution_type": "noisy"})

    assert config["backend"]["real"]["id"] == "ibm_basquecountry_new"


def test_battery_config_generates_missing_real_backend_id():
    config = build_config_from_options(base_config(), {"experiment.execution_type": "noisy"})

    assert config["backend"]["real"]["id"] == "ibm_basquecountry"


def test_battery_config_disables_reset_info_for_shared_real_backend_info():
    raw_config = base_config()
    raw_config["backend"]["real"]["info_storage"] = "shared"
    raw_config["backend"]["real"]["reset_info"] = True

    config = build_config_from_options(raw_config, {"experiment.execution_type": "noisy"})

    assert config["backend"]["real"]["reset_info"] is False


def test_battery_config_keeps_reset_info_for_run_real_backend_info():
    raw_config = base_config()
    raw_config["backend"]["real"]["info_storage"] = "run"
    raw_config["backend"]["real"]["reset_info"] = True

    config = build_config_from_options(raw_config, {"experiment.execution_type": "noisy"})

    assert config["backend"]["real"]["reset_info"] is True


def test_prepare_run_config_keeps_random_circuit_separate_from_randomness():
    raw_config = base_config()
    raw_config["encoding"]["random_circuit"] = 0
    raw_config["encoding"]["randomness"] = 0.1

    config = prepare_run_config(raw_config)

    assert config["encoding"]["random_circuit"] == 0
    assert config["encoding"]["randomness"] == 0.1
    assert "rand0.1" in config["run"]["id"]
    assert "rc0" not in config["run"]["id"]


@pytest.mark.parametrize("random_circuit", [0, 1, 2])
def test_prepare_run_config_accepts_random_circuit_types(random_circuit):
    raw_config = base_config()
    raw_config["encoding"]["random_circuit"] = random_circuit

    config = prepare_run_config(raw_config)

    assert config["encoding"]["random_circuit"] == random_circuit


@pytest.mark.parametrize("random_circuit", [False, True, 3, 4])
def test_prepare_run_config_rejects_invalid_random_circuit_types(random_circuit):
    raw_config = base_config()
    raw_config["encoding"]["random_circuit"] = random_circuit

    with pytest.raises(ConfigValidationError):
        prepare_run_config(raw_config)


def test_prepare_run_config_rejects_randomness_above_one():
    raw_config = base_config()
    raw_config["encoding"]["randomness"] = 1.1

    with pytest.raises(ConfigValidationError, match="encoding.randomness must be <= 1"):
        prepare_run_config(raw_config)


def test_prepare_run_config_rejects_invalid_real_backend_info_storage():
    raw_config = base_config()
    raw_config["backend"]["real"]["info_storage"] = "per_run"

    with pytest.raises(ConfigValidationError, match="backend.real.info_storage"):
        prepare_run_config(raw_config)


def test_prepare_run_config_requires_real_backend_info_storage():
    raw_config = base_config()
    del raw_config["backend"]["real"]["info_storage"]

    with pytest.raises(ConfigValidationError, match="backend.real.info_storage"):
        prepare_run_config(raw_config)


def test_prepare_run_config_rejects_non_finite_numbers():
    raw_config = base_config()
    raw_config["training"]["learning_rate"] = float("nan")

    with pytest.raises(ConfigValidationError, match="training.learning_rate must be finite"):
        prepare_run_config(raw_config)


def test_prepare_run_config_rejects_invalid_simulator_precision():
    raw_config = base_config()
    raw_config["backend"]["simulator"]["data_type"] = "float"

    with pytest.raises(ConfigValidationError, match="backend.simulator.data_type"):
        prepare_run_config(raw_config)


def test_prepare_run_config_rejects_unimplemented_manual_estimator():
    raw_config = base_config()
    raw_config["implementation"]["name"] = "manual_estimator"

    with pytest.raises(ConfigValidationError, match="implementation.name"):
        prepare_run_config(raw_config)


def test_base_preset_requires_two_qubits():
    raw_config = base_config()
    raw_config["experiment"]["n_qubits"] = 1

    with pytest.raises(ConfigValidationError, match="base preset requires"):
        prepare_run_config(raw_config)


def test_reg_requires_local_noiseless_qml_torch():
    raw_config = base_config()
    raw_config["experiment"]["execution_type"] = "noisy"
    raw_config["experiment"]["gradient_method"] = "REG"

    with pytest.raises(ConfigValidationError, match="REG requires"):
        prepare_run_config(raw_config)


def test_amplitude_preset_requires_state_dimension():
    raw_config = base_config()
    raw_config["experiment"]["implementation"] = "amp"
    config = prepare_run_config(raw_config)
    config["dataset"]["parameters"]["total_pixels"] = 7

    with pytest.raises(ConfigValidationError, match="requires dataset.parameters.total_pixels=8"):
        validate_config(config)


def test_loaded_dataset_must_match_encoding_dimension():
    np = pytest.importorskip("numpy")
    raw_config = base_config()
    raw_config["experiment"]["implementation"] = "amp"
    config = prepare_run_config(raw_config)

    with pytest.raises(RuntimeValidationError, match="requires 8 values per sample"):
        validate_loaded_dataset(config, np.ones((2, 7)))


def test_backend_must_have_enough_qubits():
    config = prepare_run_config(base_config())
    backend = type("SmallBackend", (), {"num_qubits": 2})()

    with pytest.raises(RuntimeValidationError, match="requires 3"):
        validate_backend_capacity(config, backend)


def test_large_density_matrix_emits_memory_warning():
    raw_config = base_config()
    raw_config["experiment"]["execution_type"] = "noisy"
    raw_config["experiment"]["gradient_method"] = "SPSA"
    raw_config["experiment"]["n_qubits"] = 16
    config = prepare_run_config(raw_config)
    with pytest.warns(RuntimeWarning, match="density_matrix simulation of 16 qubits"):
        estimated_bytes = warn_simulation_memory(config)

    assert estimated_bytes == 64 * 1024**3


def test_transpiler_plugin_names_are_not_restricted():
    raw_config = base_config()
    raw_config["backend"]["transpilation"]["layout_method"] = "custom_layout_plugin"
    raw_config["backend"]["transpilation"]["routing_method"] = "custom_routing_plugin"

    prepare_run_config(raw_config)


def test_parallelism_combination_is_left_to_aer():
    raw_config = base_config()
    raw_config["backend"]["simulator"]["max_parallel_experiments"] = 2
    raw_config["backend"]["simulator"]["max_parallel_shots"] = 2

    prepare_run_config(raw_config)


def test_battery_option_values_must_be_non_empty_lists():
    with pytest.raises(ConfigValidationError, match="non-empty list"):
        build_config_combinations(base_config(), {"run.seed": []})


def test_battery_rejects_duplicate_output_files():
    config = prepare_run_config(base_config())

    with pytest.raises(ConfigValidationError, match="same config file"):
        validate_unique_config_filenames([("first", [config]), ("second", [config])])
