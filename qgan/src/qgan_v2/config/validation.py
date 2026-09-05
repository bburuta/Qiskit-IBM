import math
import warnings


#- Supported config values -#

# Valid values
VALID_IMPLEMENTATIONS = {"qml_torch", "runtime_packed"}
VALID_PRESETS = {"base", "ang", "amp"}
VALID_EXECUTION_TYPES = {"noiseless", "noisy", "fake_real", "real"}
VALID_GRADIENT_METHODS = {"PSR", "SPSA", "REG"}
VALID_DEVICES = {"CPU", "GPU"}
VALID_ENCODINGS = {"direct_circuit", "angle", "amplitude"}
VALID_DATASET_TYPES = {"quantum", "classical"}
VALID_DATASET_SOURCES = {"specific_distribution", "generated_gradients"}
VALID_CIRCUITS = {"real_amplitudes", "efficient_su2"}
VALID_SIMULATOR_METHODS = {"statevector", "density_matrix"}
VALID_SIMULATOR_PRECISIONS = {"single", "double"}
VALID_SIMULATOR_MAPPINGS = {"hardware", "noise_model"}
VALID_DISCRIMINATOR_PACKING = {"separate", "joined"}
VALID_RANDOM_CIRCUITS = {0, 1, 2}
VALID_REAL_BACKEND_INFO_STORAGE = {"shared", "run"}
VALID_EVALUATION_METHODS = {"gradient", "kl"}

CHOICE_RULES = [
    ("implementation.name", VALID_IMPLEMENTATIONS),
    ("implementation.discriminator_packing", VALID_DISCRIMINATOR_PACKING),
    ("experiment.implementation", VALID_PRESETS),
    ("experiment.execution_type", VALID_EXECUTION_TYPES),
    ("experiment.gradient_method", VALID_GRADIENT_METHODS),
    ("run.device", VALID_DEVICES),
    ("backend.simulator.device", VALID_DEVICES),
    ("encoding.type", VALID_ENCODINGS),
    ("dataset.type", VALID_DATASET_TYPES),
    ("dataset.source", VALID_DATASET_SOURCES),
    ("circuits.generator", VALID_CIRCUITS),
    ("circuits.discriminator", VALID_CIRCUITS),
    ("backend.simulator.noiseless_method", VALID_SIMULATOR_METHODS),
    ("backend.simulator.noisy_method", VALID_SIMULATOR_METHODS),
    ("backend.simulator.data_type", VALID_SIMULATOR_PRECISIONS),
    ("backend.simulator.noisy_backend_mapping", VALID_SIMULATOR_MAPPINGS),
    ("backend.real.info_storage", VALID_REAL_BACKEND_INFO_STORAGE),
    ("encoding.eval_method", VALID_EVALUATION_METHODS),
]


# Numeric values
NUMBER_RULES = [
    ("run.seed", 0, 2**32 - 1, True),
    ("experiment.n_qubits", 1, None, True),
    ("training.max_iterations", 0, None, True),
    ("training.gen_iterations", 0, None, True),
    ("training.disc_iterations", 0, None, True),
    ("training.init_scale", 0, None, False),
    ("training.learning_rate", 0, None, False),
    ("training.print_every", 0, None, True),
    ("training.checkpoint_every", 0, None, True),
    ("backend.precision", 0, 1, False),
    ("backend.transpilation.optimization_level", 0, 3, True),
    ("backend.real.estimator.resilience_level", 0, 2, True),
    ("backend.simulator.max_parallel_threads", 0, None, True),
    ("backend.simulator.max_parallel_experiments", 0, None, True),
    ("backend.simulator.max_parallel_shots", 0, None, True),
    ("encoding.contrast", 0, None, False),
    ("encoding.randomness", 0, 1, False),
    ("encoding.batch_size", 1, None, True),
    ("encoding.eval_batch_size", 1, None, True),
    ("encoding.max_parallel_threads", 1, None, True),
]


# Boolean values
BOOLEAN_RULES = [
    "backend.reset",
    "backend.save_backend_file",
    "backend.real.reset_info",
    "backend.real.confirm_runtime_execution",
    "backend.real.estimator.dynamical_decoupling.enable",
    "backend.simulator.gpu.cuStateVec_enable",
    "backend.simulator.gpu.batched_shots_gpu",
    "backend.simulator.gpu.blocking_enable",
    "backend.simulator.gpu.runtime_parameter_bind_enable",
    "circuits.reset",
    "dataset.reset",
]


# Preset constraints
PRESET_ENCODINGS = {
    "base": "direct_circuit",
    "ang": "angle",
    "amp": "amplitude",
}

PRESET_DATASETS = {
    "base": ("quantum", "specific_distribution"),
    "ang": ("classical", "generated_gradients"),
    "amp": ("classical", "generated_gradients"),
}


# Options required before config normalization can run
PRE_NORMALIZATION_REQUIRED_OPTIONS = [
    "implementation.name",
    "implementation.discriminator_packing",
    "run.id",
    "run.seed",
    "experiment.implementation",
    "experiment.execution_type",
    "experiment.n_qubits",
    "experiment.gradient_method",
    "dataset",
    "dataset.id",
    "backend.simulator.device",
    "backend.real.id",
    "backend.real.name",
    "backend.transpilation.layout_method",
    "backend.transpilation.routing_method",
    "backend.real.info_storage",
    "encoding",
    "encoding.random_circuit",
    "encoding.randomness",
]


# Config objects that must be mappings before normalization accesses them
REQUIRED_MAPPINGS = [
    "implementation",
    "run",
    "experiment",
    "training",
    "dataset",
    "circuits",
    "backend",
    "backend.transpilation",
    "backend.simulator",
    "backend.simulator.gpu",
    "backend.real",
    "backend.real.estimator",
    "backend.real.estimator.dynamical_decoupling",
    "encoding",
]


# Options that must be non-empty
NON_EMPTY_STRING_OPTIONS = [
    "run.data_path",
    "run.id",
    "dataset.id",
    "backend.real.id",
    "backend.real.name",
]


#- Validation errors -#

class ConfigValidationError(ValueError):
    """Raised when a qGAN config is structurally invalid."""


class RuntimeValidationError(ValueError):
    """Raised when runtime data or resources cannot satisfy a valid config."""


#- Config path helpers -#

# Require a dotted config path
def require_path(config, path):
    cursor = config
    for key in path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            raise ConfigValidationError(f"Missing config option: {path}")
        cursor = cursor[key]
    return cursor


# Require a config path value to be a mapping
def require_mapping(config, path):
    value = require_path(config, path)
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{path} must be a mapping. Got: {value!r}")
    return value


# Require a config path value to be one of the allowed choices
def require_choice(config, path, choices):
    value = require_path(config, path)
    try:
        valid = value in choices
    except TypeError:
        valid = False
    if not valid:
        allowed = ", ".join(sorted(choices))
        raise ConfigValidationError(f"Invalid value for {path}: {value!r}. Allowed values: {allowed}")
    return value


# Require an integer config path value to be one of the allowed choices
def require_integer_choice(config, path, choices):
    value = require_path(config, path)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigValidationError(f"{path} must be an integer. Got: {value!r}")

    if value not in choices:
        allowed = ", ".join(str(choice) for choice in sorted(choices))
        raise ConfigValidationError(f"Invalid value for {path}: {value!r}. Allowed values: {allowed}")

    return value


# Require a numeric config path value
def require_number(config, path, minimum=None, maximum=None, integer=False):
    value = require_path(config, path)
    if integer:
        valid_type = isinstance(value, int) and not isinstance(value, bool)
    else:
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)

    if not valid_type:
        expected = "integer" if integer else "number"
        raise ConfigValidationError(f"{path} must be a {expected}. Got: {value!r}")

    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigValidationError(f"{path} must be finite. Got: {value!r}")

    if minimum is not None and value < minimum:
        raise ConfigValidationError(f"{path} must be >= {minimum}. Got: {value!r}")

    if maximum is not None and value > maximum:
        raise ConfigValidationError(f"{path} must be <= {maximum}. Got: {value!r}")

    return value


# Require a boolean config path value
def require_boolean(config, path):
    value = require_path(config, path)
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{path} must be a boolean. Got: {value!r}")
    return value


# Require a non-empty string config path
def require_non_empty_string(config, path):
    value = require_path(config, path)
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{path} must be a non-empty string. Got: {value!r}")
    return value


#- Layer 1: option value validation -#

# Validate fields required before normalization can run
def validate_raw_config(config):
    if not isinstance(config, dict):
        raise ConfigValidationError("Config root must be a mapping.")

    for path in REQUIRED_MAPPINGS:
        require_mapping(config, path)

    for path in PRE_NORMALIZATION_REQUIRED_OPTIONS:
        require_path(config, path)

    require_choice(config, "experiment.implementation", VALID_PRESETS)


# Validate normalized option types, choices, and ranges
def validate_option_values(config):
    for path, choices in CHOICE_RULES:
        require_choice(config, path, choices)

    require_integer_choice(config, "encoding.random_circuit", VALID_RANDOM_CIRCUITS)

    for path, minimum, maximum, integer in NUMBER_RULES:
        require_number(config, path, minimum=minimum, maximum=maximum, integer=integer)

    for path in BOOLEAN_RULES:
        require_boolean(config, path)

    require_mapping(config, "dataset.parameters")

    for path in NON_EMPTY_STRING_OPTIONS:
        require_non_empty_string(config, path)


#- Layer 2: option combination validation -#


# Validate the dataset and encoding contract applied by each preset
def validate_preset_combination(config):
    preset = config["experiment"]["implementation"]
    encoding = config["encoding"]["type"]
    expected_encoding = PRESET_ENCODINGS[preset]

    # Circuit/data builders only implement the encoding assigned to each preset.
    if encoding != expected_encoding:
        raise ConfigValidationError(
            f"experiment.implementation={preset!r} requires encoding.type={expected_encoding!r}. "
            f"Got: {encoding!r}"
        )

    dataset_type = config["dataset"]["type"]
    dataset_source = config["dataset"]["source"]
    expected_type, expected_source = PRESET_DATASETS[preset]
    # Each preset selects the only dataset representation supported by its encoder.
    if (dataset_type, dataset_source) != (expected_type, expected_source):
        raise ConfigValidationError(
            f"experiment.implementation={preset!r} requires "
            f"dataset.type={expected_type!r} and dataset.source={expected_source!r}."
        )

    n_qubits = config["experiment"]["n_qubits"]

    # The base dataset applies a controlled gate to its final two qubits.
    if preset == "base" and n_qubits < 2:
        raise ConfigValidationError("The base preset requires experiment.n_qubits >= 2.")

    if preset in {"ang", "amp"}:
        parameters = config["dataset"]["parameters"]
        total_pixels = parameters.get("total_pixels")
        if not isinstance(total_pixels, int) or isinstance(total_pixels, bool) or total_pixels < 1:
            raise ConfigValidationError(
                "dataset.parameters.total_pixels must be a positive integer."
            )

        # Angle encoding needs one feature per qubit; amplitude encoding needs
        # one value for every computational basis state.
        expected_pixels = n_qubits if preset == "ang" else 2**n_qubits
        if total_pixels != expected_pixels:
            raise ConfigValidationError(
                f"The {preset} preset requires dataset.parameters.total_pixels="
                f"{expected_pixels}. Got: {total_pixels!r}"
            )


# Validate gradient support across implementations and execution targets
def validate_gradient_combination(config):
    gradient = config["experiment"]["gradient_method"]
    implementation = config["implementation"]["name"]
    execution_type = config["experiment"]["execution_type"]

    # REG computes an exact local reverse gradient and does not model noise
    # or hardware execution in the current implementation.
    if gradient == "REG" and (
        implementation != "qml_torch" or execution_type != "noiseless"
    ):
        raise ConfigValidationError(
            "REG requires implementation.name='qml_torch' and "
            "experiment.execution_type='noiseless'."
        )


# Validate implementation-specific support
def validate_implementation_combination(config):
    if config["implementation"]["name"] != "runtime_packed":
        return

    execution_type = config["experiment"]["execution_type"]
    # Packed models use primitive jobs that are only created for noisy and
    # hardware-style execution paths.
    if execution_type not in {"noisy", "fake_real", "real"}:
        raise ConfigValidationError(
            "runtime_packed supports execution_type: noisy, fake_real, real."
        )

    gradient_method = config["experiment"]["gradient_method"]
    # The packed gradient builder only implements parameter shift and SPSA.
    if gradient_method not in {"PSR", "SPSA"}:
        raise ConfigValidationError(
            "runtime_packed supports gradient_method: PSR, SPSA."
        )

    encoding = config["encoding"]["type"]
    discriminator_packing = config["implementation"]["discriminator_packing"]
    batch_size = config["encoding"]["batch_size"]

    # Joined packing constructs real and fake branches in one circuit, which
    # gives it stricter width and batching constraints than separate packing.
    if discriminator_packing == "joined":
        # The joined discriminator builder has no amplitude-encoding branch.
        if encoding == "amplitude":
            raise ConfigValidationError(
                "runtime_packed joined discriminator packing does not support amplitude encoding."
            )
        # Direct-circuit joining creates exactly one real and one fake branch.
        if encoding == "direct_circuit" and batch_size != 1:
            raise ConfigValidationError(
                "runtime_packed joined direct_circuit requires encoding.batch_size=1."
            )
        # Angle joining splits the batch equally into real and fake branches.
        if encoding == "angle" and batch_size % 2:
            raise ConfigValidationError(
                "runtime_packed joined angle requires an even encoding.batch_size."
            )

    # Packed trainable parameters share memory with NumPy CPU arrays.
    if config["run"]["device"] != "CPU":
        raise ConfigValidationError(
            "runtime_packed requires run.device=CPU for NumPy-backed parameters. "
            "Use backend.simulator.device=GPU for Aer GPU execution."
        )


# Validate Aer options that cannot be enabled together
def validate_simulator_combination(config):
    simulator = config["backend"]["simulator"]
    gpu = simulator["gpu"]

    # Aer documents cuStateVec and batched-shot GPU execution as incompatible.
    if simulator["device"] == "GPU" and gpu["cuStateVec_enable"] and gpu["batched_shots_gpu"]:
        raise ConfigValidationError(
            "backend.simulator.gpu.cuStateVec_enable and batched_shots_gpu "
            "cannot both be enabled."
        )


# Validate all important cross-option contracts
def validate_option_combinations(config):
    validate_preset_combination(config)
    validate_gradient_combination(config)
    validate_implementation_combination(config)
    validate_simulator_combination(config)


# Validate normalized config values
def validate_config(config):
    validate_option_values(config)
    validate_option_combinations(config)
    return config


#- Layer 3: runtime validation -#

# Validate a loaded classical dataset, including cached files
def validate_loaded_dataset(config, dataset):
    import numpy as np

    values = np.asarray(dataset)
    if values.size == 0 or values.ndim < 2:
        raise RuntimeValidationError("The classical dataset must contain at least one sample.")
    if not np.issubdtype(values.dtype, np.number):
        raise RuntimeValidationError("The classical dataset must contain numeric values.")
    if not np.isrealobj(values):
        raise RuntimeValidationError("The classical dataset must contain real values.")
    if not np.isfinite(values).all():
        raise RuntimeValidationError("The classical dataset contains NaN or infinite values.")

    encoding = config["encoding"]["type"]
    n_qubits = config["experiment"]["n_qubits"]
    features = math.prod(values.shape[1:])
    expected_features = n_qubits if encoding == "angle" else 2**n_qubits
    if features != expected_features:
        raise RuntimeValidationError(
            f"{encoding} encoding requires {expected_features} values per sample. "
            f"The loaded dataset has {features}."
        )

    if encoding == "amplitude":
        if np.any(values < 0):
            raise RuntimeValidationError("Amplitude-encoded samples cannot contain negative values.")
        if np.any(values.reshape(len(values), -1).sum(axis=1) == 0):
            raise RuntimeValidationError("Amplitude-encoded samples cannot be all zero.")

    return dataset


# Validate generated or cached circuits before model construction
def validate_circuit_bundle(config, circuit_bundle):
    if not isinstance(circuit_bundle, (list, tuple)) or len(circuit_bundle) != 4:
        raise RuntimeValidationError(
            "Circuit bundle must contain generator, discriminator, randomizer, and real circuits."
        )

    generator, discriminator, randomizer, real_circuits = circuit_bundle
    if not isinstance(real_circuits, (list, tuple)) or not real_circuits:
        raise RuntimeValidationError("Circuit bundle must contain at least one real-data circuit.")

    n_qubits = config["experiment"]["n_qubits"]
    named_circuits = [
        ("generator", generator),
        ("discriminator", discriminator),
        ("randomizer", randomizer),
        *((f"real circuit {index}", circuit) for index, circuit in enumerate(real_circuits)),
    ]
    for name, circuit in named_circuits:
        if getattr(circuit, "num_qubits", None) != n_qubits:
            raise RuntimeValidationError(
                f"The {name} must use {n_qubits} qubits. "
                f"Got: {getattr(circuit, 'num_qubits', None)!r}"
            )

    return circuit_bundle


# Compute the largest training circuit width before packed circuits are built
def get_required_backend_qubits(config):
    n_qubits = config["experiment"]["n_qubits"]
    if config["implementation"]["name"] != "runtime_packed":
        return n_qubits

    encoding = config["encoding"]["type"]
    packing = config["implementation"]["discriminator_packing"]
    if packing == "joined" and encoding == "direct_circuit":
        return 2 * n_qubits
    return n_qubits * config["encoding"]["batch_size"]


# Validate that a constructed backend is wide enough for the selected model
def validate_backend_capacity(config, backend, required_qubits=None):
    required = required_qubits or get_required_backend_qubits(config)
    available = getattr(backend, "num_qubits", None)
    if available is None:
        available = getattr(getattr(backend, "target", None), "num_qubits", None)

    if available is not None and available < required:
        raise RuntimeValidationError(
            f"The selected backend has {available} qubits, but this configuration "
            f"requires {required}."
        )
    return backend


# Estimate the state storage used by an Aer simulation before workspace overhead
def estimate_simulation_memory_bytes(n_qubits, method, data_type):
    complex_bytes = 8 if data_type == "single" else 16
    amplitudes = 2**n_qubits if method == "statevector" else 4**n_qubits
    return amplitudes * complex_bytes


# Warn before Aer is started when the simulated quantum state is large
def warn_simulation_memory(config):
    execution_type = config["experiment"]["execution_type"]
    simulator = config["backend"]["simulator"]
    data_type = simulator["data_type"]
    n_qubits = config["experiment"]["n_qubits"]

    simulations = [("evaluation", n_qubits, simulator["noiseless_method"])]
    if execution_type in {"noiseless", "noisy"}:
        method = (
            simulator["noiseless_method"]
            if execution_type == "noiseless"
            else simulator["noisy_method"]
        )
        simulations.append(("training", get_required_backend_qubits(config), method))

    stage, width, method = max(
        simulations,
        key=lambda item: estimate_simulation_memory_bytes(item[1], item[2], data_type),
    )
    estimated_bytes = estimate_simulation_memory_bytes(width, method, data_type)
    if estimated_bytes >= 8 * 1024**3:
        warnings.warn(
            f"The quantum state for the {stage} {method} simulation of {width} qubits "
            f"uses about {estimated_bytes / 1024**3:.1f} GiB before Aer and process overhead. "
            "Check the available CPU or GPU memory before running.",
            RuntimeWarning,
            stacklevel=2,
        )

    return estimated_bytes
