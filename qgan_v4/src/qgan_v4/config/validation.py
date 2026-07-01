#- Check config values -#

# Valid values
VALID_IMPLEMENTATIONS = {"qml_torch", "runtime_packed", "manual_estimator"}
VALID_PRESETS = {"base", "ang", "amp"}
VALID_EXECUTION_TYPES = {"noiseless", "noisy", "fake_real", "real"}
VALID_GRADIENT_METHODS = {"PSR", "SPSA", "REG"}
VALID_DEVICES = {"CPU", "GPU"}
VALID_ENCODINGS = {"direct_circuit", "angle", "amplitude"}
VALID_SIMULATOR_MAPPINGS = {"hardware", "noise_model"}
VALID_DISCRIMINATOR_PACKING = {"separate", "joined"}

CHOICE_RULES = [
    ("implementation.name", VALID_IMPLEMENTATIONS),
    ("implementation.discriminator_packing", VALID_DISCRIMINATOR_PACKING),
    ("experiment.implementation", VALID_PRESETS),
    ("experiment.execution_type", VALID_EXECUTION_TYPES),
    ("experiment.gradient_method", VALID_GRADIENT_METHODS),
    ("run.device", VALID_DEVICES),
    ("backend.simulator.device", VALID_DEVICES),
    ("encoding.type", VALID_ENCODINGS),
    ("backend.simulator.noisy_backend_mapping", VALID_SIMULATOR_MAPPINGS),
]


# Numeric values
NUMBER_RULES = [
    ("experiment.n_qubits", 1, True),
    ("training.max_iterations", 0, True),
    ("training.gen_iterations", 0, True),
    ("training.disc_iterations", 0, True),
    ("training.init_scale", 0, False),
    ("training.learning_rate", 0, False),
    ("training.print_every", 0, True),
    ("backend.precision", 0, False),
    ("backend.simulator.max_parallel_threads", 0, True),
    ("backend.simulator.max_parallel_experiments", 0, True),
    ("backend.simulator.max_parallel_shots", 0, True),
    ("encoding.contrast", 0, False),
    ("encoding.randomness", 0, False),
    ("encoding.batch_size", 1, True),
    ("encoding.eval_batch_size", 1, True),
    ("encoding.max_parallel_threads", 1, True),
]


# Boolean values
BOOLEAN_RULES = [
    "backend.reset",
    "backend.real.reset_info",
    "backend.real.confirm_execution",
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
    "encoding",
    "encoding.randomness",
]


# Options that must be non-empty
NON_EMPTY_OPTIONS = [
    "run.data_path",
    "run.id",
    "dataset.id",
    "backend.real.id",
]


#- Validation errors -#

class ConfigValidationError(ValueError):
    """Raised when a qGAN config is structurally invalid."""


#- Config path helpers -#

# Require a dotted config path
def require_path(config, path):
    cursor = config
    for key in path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            raise ConfigValidationError(f"Missing config option: {path}")
        cursor = cursor[key]
    return cursor


# Require a config path value to be one of the allowed choices
def require_choice(config, path, choices):
    value = require_path(config, path)
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ConfigValidationError(f"Invalid value for {path}: {value!r}. Allowed values: {allowed}")
    return value


# Require a numeric config path value
def require_number(config, path, minimum=None, integer=False):
    value = require_path(config, path)
    if integer:
        valid_type = isinstance(value, int) and not isinstance(value, bool)
    else:
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)

    if not valid_type:
        expected = "integer" if integer else "number"
        raise ConfigValidationError(f"{path} must be a {expected}. Got: {value!r}")

    if minimum is not None and value < minimum:
        raise ConfigValidationError(f"{path} must be >= {minimum}. Got: {value!r}")

    return value


# Require a boolean config path value
def require_boolean(config, path):
    value = require_path(config, path)
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{path} must be a boolean. Got: {value!r}")
    return value


#- Config validation -#

# Validate fields required before normalization can run
def validate_raw_config(config):
    for path in PRE_NORMALIZATION_REQUIRED_OPTIONS:
        require_path(config, path)

    require_choice(config, "experiment.implementation", VALID_PRESETS)


# Validate preset and encoding compatibility
def validate_preset_encoding(config):
    preset = config["experiment"]["implementation"]
    encoding = config["encoding"]["type"]
    expected_encoding = PRESET_ENCODINGS[preset]

    if encoding != expected_encoding:
        raise ConfigValidationError(
            f"experiment.implementation={preset!r} requires encoding.type={expected_encoding!r}. "
            f"Got: {encoding!r}"
        )


# Validate implementation-specific support
def validate_implementation_compatibility(config):
    if config["implementation"]["name"] != "runtime_packed":
        return

    execution_type = config["experiment"]["execution_type"]
    if execution_type not in {"noisy", "fake_real", "real"}:
        raise ConfigValidationError(
            "runtime_packed supports execution_type: noisy, fake_real, real."
        )

    gradient_method = config["experiment"]["gradient_method"]
    if gradient_method not in {"PSR", "SPSA"}:
        raise ConfigValidationError(
            "runtime_packed supports gradient_method: PSR, SPSA."
        )

    encoding = config["encoding"]["type"]
    if encoding not in {"direct_circuit", "angle", "amplitude"}:
        raise ConfigValidationError(
            "runtime_packed supports encoding.type: direct_circuit, angle, amplitude."
        )

    discriminator_packing = config["implementation"]["discriminator_packing"]
    batch_size = config["encoding"]["batch_size"]

    # Joined packing has encoding-specific circuit constraints
    if discriminator_packing == "joined":
        if encoding == "amplitude":
            raise ConfigValidationError(
                "runtime_packed joined discriminator packing does not support amplitude encoding."
            )
        if encoding == "direct_circuit" and batch_size != 1:
            raise ConfigValidationError(
                "runtime_packed joined direct_circuit requires encoding.batch_size=1."
            )
        if encoding == "angle" and batch_size % 2:
            raise ConfigValidationError(
                "runtime_packed joined angle requires an even encoding.batch_size."
            )

    if config["run"]["device"] != "CPU":
        raise ConfigValidationError(
            "runtime_packed requires run.device=CPU for NumPy-backed parameters. "
            "Use backend.simulator.device=GPU for Aer GPU execution."
        )


# Validate normalized config values
def validate_config(config):
    for path, choices in CHOICE_RULES:
        require_choice(config, path, choices)

    for path, minimum, integer in NUMBER_RULES:
        require_number(config, path, minimum=minimum, integer=integer)

    for path in BOOLEAN_RULES:
        require_boolean(config, path)

    for path in NON_EMPTY_OPTIONS:
        if not require_path(config, path):
            raise ConfigValidationError(f"{path} must not be empty.")

    validate_preset_encoding(config)
    validate_implementation_compatibility(config)
    return config
