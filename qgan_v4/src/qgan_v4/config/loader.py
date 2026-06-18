from pathlib import Path

import yaml

from qgan_v4.config.defaults import normalize_config
from qgan_v4.config.validation import validate_config, validate_raw_config


#- Config file IO -#

# Load YAML config file
def load_config_file(filename):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


# Save YAML config file
def save_config_file(config, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)



#- Run config loading -#

# Validate, normalize, and validate again a run config
def prepare_run_config(config):
    validate_raw_config(config)
    config = normalize_config(config)
    return validate_config(config)


# Load and prepare run config
def load_run_config(filename):
    config = load_config_file(filename)
    return prepare_run_config(config)
