import os
import yaml
import copy
import itertools
from pathlib import Path



#- Configuration manager -#

ALLOWED_CONFIG_VALUES_FILE = Path(__file__).resolve().parents[1] / "config_manager" / "allowed_config_values.yaml"
CONFIG_FILENAME = "config.yaml"

# Generate run id
def generate_run_id(run_config, experiment_config):
    run_id = (
        f"{experiment_config['implementation']}-"
        f"q{experiment_config['n_qubits']}-"
        f"{experiment_config['execution_type']}-"
        f"{experiment_config['gradient_method']}-"
        f"{run_config['device']}-"
        f"seed{run_config['seed']}"
    )

    return run_id


# Generate dataset id
def generate_dataset_id(dataset_config):
    dataset_id = dataset_config['source']
    parameters = dataset_config.get('parameters', {})

    if parameters:
        parameter_id = "-".join(
            f"{key}{value}"
            for key, value in sorted(parameters.items())
            if value is not None
        )
        if parameter_id:
            dataset_id = f"{dataset_id}-{parameter_id}"

    return dataset_id


# Generate real backend id
def generate_real_backend_id(real_backend_config):
    return real_backend_config['name']


# Create config ids
def create_config_ids(config):
    if config['run']['id'] is None:
        config['run']['id'] = generate_run_id(config['run'], config['experiment'])

    if config['dataset']['id'] is None:
        config['dataset']['id'] = generate_dataset_id(config['dataset'])

    if config['backend']['real']['id'] is None:
        config['backend']['real']['id'] = generate_real_backend_id(config['backend']['real'])


# Apply implementation-specific config values
def apply_implementation_config(config):
    implementation = config['experiment']['implementation']
    n_qubits = config['experiment']['n_qubits']

    if implementation == 'base':
        config['dataset']['type'] = 'quantum'
        config['dataset']['source'] = 'specific_distribution'
        config['dataset']['parameters'] = {}
        config['encoding']['type'] = 'direct_circuit'

    elif implementation == 'ang':
        config['dataset']['type'] = 'classical'
        config['dataset']['source'] = 'generated_gradients'
        config['dataset']['parameters'] = {'total_pixels': n_qubits}
        config['encoding']['type'] = 'angle'

    elif implementation == 'amp':
        config['dataset']['type'] = 'classical'
        config['dataset']['source'] = 'generated_gradients'
        config['dataset']['parameters'] = {'total_pixels': 2 ** n_qubits}
        config['encoding']['type'] = 'amplitude'

    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    config['run']['id'] = None
    config['dataset']['id'] = None


# Check config values
def check_config(config, allowed_config=None, path="config"):
    if allowed_config is None:
        allowed_config = load_config_file(ALLOWED_CONFIG_VALUES_FILE)

    if isinstance(allowed_config, dict):
        if not isinstance(config, dict):
            raise ValueError(f"{path} must be a dict.")

        if not allowed_config:
            return

        for key in config:
            if key not in allowed_config:
                raise ValueError(f"Unknown config option: {path}.{key}")

        for key in allowed_config:
            if key not in config:
                raise ValueError(f"Missing config option: {path}.{key}")

        for key in allowed_config:
            check_config(config[key], allowed_config[key], f"{path}.{key}")

    elif isinstance(allowed_config, list):
        if allowed_config and config not in allowed_config:
            raise ValueError(f"Invalid config value for {path}: {config}. Allowed values: {allowed_config}")

    else:
        raise ValueError(f"Invalid allowed config structure at {path}.")
    


#- Config save -#

# Get run path
def get_run_path(config):
    return Path(config['run']['data_path']) / config['run']['id']


# Get config file path
def get_config_filename(config):
    return Path(get_run_path(config)) / CONFIG_FILENAME


# Save config file
def create_config_file(config, overwrite=False):
    create_config_ids(config)
    check_config(config)
    filename = get_config_filename(config)

    fileExists = filename.exists()
    if fileExists and not overwrite:
        print("Configuration file already exists. Path: " + str(filename.parent))
    
    else:
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, "w") as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)

        action = "created" if not fileExists else "overwritten"
        print(f"Configuration file {action}. Path: {filename.parent}")


# Create multiple config files from config list
def create_config_files(confs, overwrite=False):
    for conf in confs:
        create_config_file(conf, overwrite=overwrite)


# Load default config file
def load_config_file(filename):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)

    return config


#- Create config files for parameter combinations -#

# Find nested keys in a config dict
def find_key_paths(config, search_key, path=None):
    if path is None:
        path = []

    paths = []
    for key, value in config.items():
        current_path = path + [key]
        if key == search_key:
            paths.append(current_path)
        if isinstance(value, dict):
            paths.extend(find_key_paths(value, search_key, current_path))

    return paths


# Save value nested in config
def set_nested_value(config, path, value):
    nested_config = config
    for key in path[:-1]:
        nested_config = nested_config[key]
    nested_config[path[-1]] = value


# Build config from options
def build_config_from_options(default_config_values, modified_config_values):
    conf = copy.deepcopy(default_config_values)

    for key, value in modified_config_values.items():
        paths = find_key_paths(conf, key)
        if not paths:
            raise KeyError(f"Unknown variable config key: {key}")
        if len(paths) > 1:
            raise KeyError(f"Ambiguous variable config key: {key}")

        set_nested_value(conf, paths[0], value)

    apply_implementation_config(conf)
    create_config_ids(conf)

    return conf


# Build configs for all parameter combinations
def build_config_combinations(default_config_values, variable_config_values):
    confs = []
    keys = variable_config_values.keys()
    for combination in itertools.product(*variable_config_values.values()):
        modified_config_values = dict(zip(keys, combination))
        conf = build_config_from_options(default_config_values, modified_config_values)
        confs.append(conf)
    return confs



#- Config management -#

# Create config files for parameter combinations
def create_configs(config_filename, overwrite=False):
    config_values = load_config_file(config_filename)
    default_config_values = config_values['default_config_values']
    variable_config_values_dict = config_values['variable_config_values_list']
    
    n_confs = 0
    for config_name, variable_config_values in variable_config_values_dict.items():
        print(f"Creating config files for {config_name}:")
        confs = build_config_combinations(default_config_values, variable_config_values)
        create_config_files(confs, overwrite=overwrite)
        print()

        n_confs += len(confs)

    print(f"{n_confs} parameter-combination configuration files created.")



#- Main -#

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    TEST_CONFIG_FILE = script_dir / "test_config.yaml"
    TRAIN_CONFIG_FILE = script_dir / "train_config.yaml"
    create_configs(TEST_CONFIG_FILE, overwrite=True)
    #create_configs(TRAIN_CONFIG_FILE, overwrite=True)
