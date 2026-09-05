import copy
import itertools

from qgan_v2.config.defaults import normalize_config
from qgan_v2.config.loader import load_config_file, load_run_config, save_config_file
from qgan_v2.config.validation import ConfigValidationError, validate_config, validate_raw_config
from qgan_v2.storage.paths import get_config_filename


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


# Shared backend info is reset explicitly with --reset-rb, not by each run.
def disable_shared_real_backend_reset(config):
    real_backend = config["backend"]["real"]
    if real_backend["info_storage"] == "shared":
        real_backend["reset_info"] = False


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
    disable_shared_real_backend_reset(config)
    return validate_config(config)


# Build all config combinations for one variable group
def build_config_combinations(default_config, variable_config_values):
    if not isinstance(variable_config_values, dict):
        raise ConfigValidationError("Each battery variable group must be a mapping.")

    for option_key, values in variable_config_values.items():
        if not isinstance(values, list) or not values:
            raise ConfigValidationError(
                f"Battery option {option_key!r} must contain a non-empty list of values."
            )

    keys = list(variable_config_values.keys())
    configs = []

    for combination in itertools.product(*(variable_config_values[key] for key in keys)):
        option_values = dict(zip(keys, combination))
        configs.append(build_config_from_options(default_config, option_values))

    return configs


# Load battery YAML file
def load_battery_file(filename):
    values = load_config_file(filename)
    if not isinstance(values, dict):
        raise ConfigValidationError("Battery root must be a mapping.")

    default_config = values.get("default_config_values")
    variable_groups = values.get("variable_config_values_list")
    if not isinstance(default_config, dict):
        raise ConfigValidationError("default_config_values must be a mapping.")
    if not isinstance(variable_groups, dict) or not variable_groups:
        raise ConfigValidationError("variable_config_values_list must be a non-empty mapping.")

    for group_name, variable_values in variable_groups.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise ConfigValidationError("Battery group names must be non-empty strings.")
        if not isinstance(variable_values, dict):
            raise ConfigValidationError(f"Battery group {group_name!r} must be a mapping.")

    return default_config, variable_groups


# Prevent two generated configs from writing to the same run directory
def validate_unique_config_filenames(grouped_configs):
    seen = {}
    for group_name, configs in grouped_configs:
        for config in configs:
            filename = str(get_config_filename(config))
            if filename in seen:
                raise ConfigValidationError(
                    f"Battery groups {seen[filename]!r} and {group_name!r} generate "
                    f"the same config file: {filename}. Give the runs different labels or options."
                )
            seen[filename] = group_name


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
    grouped_configs = [
        (group_name, build_config_combinations(default_config, variable_values))
        for group_name, variable_values in variable_groups.items()
    ]
    validate_unique_config_filenames(grouped_configs)

    filenames = []

    for group_name, configs in grouped_configs:
        print(f"Creating config files for {group_name}:")
        for config in configs:
            filenames.append(create_config_file(config, overwrite=overwrite))
        print()

    print(f"{len(filenames)} battery configuration files ready.")
    return filenames
