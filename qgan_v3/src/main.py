import argparse

from config_manager.config_manager import (
    apply_implementation_config,
    check_config,
    create_config_ids,
    load_config_file,
)
from train import train



#- Main -#

# Load config file and apply derived run values
def load_run_config(config_path):
    config = load_config_file(config_path)
    apply_implementation_config(config)
    create_config_ids(config)
    check_config(config)

    return config


# Train qGAN from config file
def run_train(config_path):
    config = load_run_config(config_path)

    return train(config)


# Parse script arguments
def get_args():
    parser = argparse.ArgumentParser(description="Fully Quantum GAN")
    parser.add_argument("config_path", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run_train(args.config_path)
