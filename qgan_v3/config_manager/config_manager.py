import os
import yaml
import copy
import itertools
from pathlib import Path



#- Configuration manager -#

# Generate run id
def generate_run_id(imp_config):
    run_id = (
        f"{imp_config['implementation']}-"
        f"q{imp_config['n_qubits']}-"
        f"{imp_config['execution_type']}-"
        f"{imp_config['device']}-"
        f"{imp_config['gradient_method']}-"
    )
    if imp_config['random_input'] is not None:
        run_id += f"random{imp_config['random_input']}-"
    run_id += f"seed{imp_config['seed']}"

    return run_id


# Get data path
def get_data_path(config):
    # Generate run id if None
    if config['data_management']['run_id'] is None:
        config['data_management']['run_id'] = generate_run_id(config['implementation_options'])
    
    config_path = config['data_management']['data_path'] + config['data_management']['run_id'] + "/"
    return config_path


# Save config file
def create_config_file(config, overwrite=False):
    config_path = get_data_path(config)
    filename = config_path + "config.yaml"

    fileExists = os.path.exists(filename)
    if fileExists and not overwrite:
        print("Configuration file already exists. Path: " + config_path)
    
    else:
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        
        with open(filename, "w") as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)

        action = "created" if not fileExists else "overwritten"
        print(f"Configuration file {action}. Path: {config_path}")


# Load config file
def load_config_file(data_path):
    filename = data_path + "config.yaml"

    with open(filename, "r") as file:
        config = yaml.safe_load(file)

    return config



#- Create config files for parameter combinations -#

# Build config from options
def build_config_from_options(default_config_values, modified_config_values):
    conf = copy.deepcopy(default_config_values)
    conf['implementation_options'].update(modified_config_values)
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


# Load default config file
def load_default_config_file(filename):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)

    return config


# Create multiple config files from config list
def create_config_files(confs, overwrite=False):
    for conf in confs:
        create_config_file(conf, overwrite=overwrite)


# Create config files for parameter combinations
def create_configs(config_filename, overwrite=False):
    config_values = load_default_config_file(config_filename)
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