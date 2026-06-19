import copy
import itertools

from qgan_v4.config.defaults import normalize_config
from qgan_v4.config.loader import load_config_file, load_run_config, save_config_file
from qgan_v4.config.validation import validate_config, validate_raw_config
from qgan_v4.storage.paths import get_config_filename


#- Config path helpers -#

# Find all nested paths whose final key matches search key
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


# Resolve simple or dotted option key to a nested config path
def resolve_option_path(config, option_key):
    if "." in option_key:
        path = option_key.split(".")
        cursor = config
        for key in path[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                raise KeyError(f"Unknown variable config path: {option_key}")
            cursor = cursor[key]
        if not isinstance(cursor, dict) or path[-1] not in cursor:
            raise KeyError(f"Unknown variable config path: {option_key}")
        return path

    paths = find_key_paths(config, option_key)
    if not paths:
        raise KeyError(f"Unknown variable config key: {option_key}")
    if len(paths) > 1:
        matches = ", ".join(".".join(path) for path in paths)
        raise KeyError(f"Ambiguous variable config key '{option_key}'. Use one of: {matches}")
    return paths[0]


# Set nested config value
def set_nested_value(config, path, value):
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


# Reset generated ids unless battery explicitly overrides them
def reset_derived_ids(config, modified_paths):
    modified = {".".join(path) for path in modified_paths}

    if "run.id" not in modified:
        config["run"]["id"] = None
    if "dataset.id" not in modified:
        config["dataset"]["id"] = None

    if "backend.real.id" not in modified:
        config["backend"]["real"]["id"] = None


#- Battery config building -#

# Build one config from default values and option overrides
def build_config_from_options(default_config, option_values):
    config = copy.deepcopy(default_config)
    modified_paths = []

    for option_key, value in option_values.items():
        path = resolve_option_path(config, option_key)
        set_nested_value(config, path, value)
        modified_paths.append(path)

    reset_derived_ids(config, modified_paths)
    validate_raw_config(config)
    config = normalize_config(config)
    return validate_config(config)


# Build all config combinations for one variable group
def build_config_combinations(default_config, variable_config_values):
    keys = list(variable_config_values.keys())
    configs = []

    for combination in itertools.product(*(variable_config_values[key] for key in keys)):
        option_values = dict(zip(keys, combination))
        configs.append(build_config_from_options(default_config, option_values))

    return configs


# Load battery YAML file
def load_battery_file(filename):
    values = load_config_file(filename)
    return values["default_config_values"], values["variable_config_values_list"]


#- Battery file creation -#

# Create one config file
def create_config_file(config, overwrite=False):
    filename = get_config_filename(config)
    existed_before = filename.exists()

    if existed_before and not overwrite:
        try:
            load_run_config(filename)
        except Exception as exc:
            print("Existing configuration file is invalid and will be rewritten. Path:", filename)
            print("Validation error:", exc)
        else:
            print("Configuration file already exists. Path:", filename)
            return filename

    save_config_file(config, filename)
    if existed_before and overwrite:
        print("Configuration file rewritten. Path:", filename)
    else:
        print("Configuration file written. Path:", filename)
    return filename


# Create all config files from a battery file
def create_battery_configs(battery_filename, overwrite=False):
    default_config, variable_groups = load_battery_file(battery_filename)
    filenames = []

    for group_name, variable_config_values in variable_groups.items():
        print(f"Creating config files for {group_name}:")
        configs = build_config_combinations(default_config, variable_config_values)
        for config in configs:
            filenames.append(create_config_file(config, overwrite=overwrite))
        print()

    print(f"{len(filenames)} battery configuration files ready.")
    return filenames
