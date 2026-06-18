from copy import deepcopy


#- Implementation presets -#

# Dataset and encoding defaults for each experiment preset
IMPLEMENTATION_PRESETS = {
    "base": {
        "dataset": {
            "type": "quantum",
            "source": "specific_distribution",
            "parameters": {},
        },
        "encoding": {
            "type": "direct_circuit",
        },
    },
    "ang": {
        "dataset": {
            "type": "classical",
            "source": "generated_gradients",
            "parameters": lambda n_qubits: {"total_pixels": n_qubits},
        },
        "encoding": {
            "type": "angle",
        },
    },
    "amp": {
        "dataset": {
            "type": "classical",
            "source": "generated_gradients",
            "parameters": lambda n_qubits: {"total_pixels": 2 ** n_qubits},
        },
        "encoding": {
            "type": "amplitude",
        },
    },
}


# Get preset value for current number of qubits
def _preset_value(value, n_qubits):
    if callable(value):
        return value(n_qubits)
    return deepcopy(value)


# Apply preset values to a config section
def apply_preset_section(config_section, preset_section, n_qubits):
    for key, value in preset_section.items():
        config_section[key] = _preset_value(value, n_qubits)


# Apply dataset and encoding defaults for the selected qGAN preset
def apply_implementation_preset(config):
    experiment = config["experiment"]
    implementation = experiment["implementation"]
    n_qubits = experiment["n_qubits"]

    if implementation not in IMPLEMENTATION_PRESETS:
        raise ValueError(f"Unknown implementation preset: {implementation}")

    preset = IMPLEMENTATION_PRESETS[implementation]
    apply_preset_section(config["dataset"], preset["dataset"], n_qubits)
    apply_preset_section(config["encoding"], preset["encoding"], n_qubits)


#- Config normalization -#

# Normalize a raw config dict into the expected format
def normalize_config(config):
    apply_implementation_preset(config)
    create_config_ids(config)
    return config


# Generate run id from experiment and run options
def generate_run_id(config):
    experiment = config["experiment"]
    run = config["run"]
    impl_name = config["implementation"]["name"]

    return (
        f"{experiment['implementation']}-"
        f"{impl_name}-"
        f"q{experiment['n_qubits']}-"
        f"{experiment['execution_type']}-"
        f"{experiment['gradient_method']}-"
        f"{run['device']}-"
        f"seed{run['seed']}"
    )


# Generate dataset id from dataset source and parameters
def generate_dataset_id(config):
    dataset = config["dataset"]
    dataset_id = dataset["source"]
    parameters = dataset["parameters"]

    parameter_id = "-".join(
        f"{key}{value}"
        for key, value in sorted(parameters.items())
        if value is not None
    )
    if parameter_id:
        dataset_id = f"{dataset_id}-{parameter_id}"

    return dataset_id


# Create missing run, dataset, and real backend ids
def create_config_ids(config):
    config["run"]["id"] = config["run"]["id"] or generate_run_id(config)
    config["dataset"]["id"] = config["dataset"]["id"] or generate_dataset_id(config)

    real_backend = config["backend"]["real"]
    if real_backend["id"] is None:
        if config["experiment"]["execution_type"] == "fake_real":
            real_backend["id"] = "fake_sherbrooke"
        else:
            real_backend["id"] = real_backend["name"]
