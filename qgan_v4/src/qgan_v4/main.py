import argparse
import copy
import traceback

from qgan_v4.config.battery import create_battery_configs
from qgan_v4.config.loader import load_config_file, load_run_config
from qgan_v4.execution.backend import reset_real_backend_info
from qgan_v4.train import train
from qgan_v4.training.interrupter import Interrupter


#- Arguments -#

# Create CLI parser
def build_parser():
    parser = argparse.ArgumentParser(description="qGAN v4")
    parser.add_argument("-p", "--battery_path", required=True, type=str)
    parser.add_argument("--reset_data", action="store_true")
    parser.add_argument("--reset_real_backend_info", "--reset_rb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")

    return parser.parse_args()


#- Run configuration -#

# Apply runtime reset options
def apply_reset_options(config, reset_data=False):
    if reset_data:
        config["training"]["reset_data"] = True

    return config


# Run one training config
def run_train(config_path, reset_data=False, interrupter=None):
    config = load_run_config(config_path)
    config = apply_reset_options(config, reset_data=reset_data)
    return train(config, interrupter=interrupter)


#- Battery execution -#

# Create battery config files
def create_battery(battery_path, overwrite=False):
    return create_battery_configs(battery_path, overwrite=overwrite)


# Get real backend options from battery default config
def get_battery_real_backend_options(battery_path):
    battery_config = load_config_file(battery_path)
    return copy.deepcopy(battery_config["default_config_values"]["backend"]["real"])


# Format exception details without printing a traceback
def format_error(exc):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


# Run battery config files
def run_battery(battery_path, reset_data=False, reset_rb=False, stop_on_error=False, overwrite=False):
    interrupter = Interrupter()

    # Reset real backend info before running the battery
    if reset_rb:
        real_backend_options = get_battery_real_backend_options(battery_path)
        reset_real_backend_info(real_backend_options)

    config_files = create_battery(battery_path, overwrite=overwrite)
    results = []

    for config_file in config_files:
        if interrupter.kill_now:
            print("Interrupter: battery execution stopped.")
            break

        print("\nRunning:", config_file)
        state = None
        error = None
        try:
            state = run_train(
                config_file,
                reset_data=reset_data,
                interrupter=interrupter,
            )
        except Exception as exc:
            if stop_on_error:
                raise

            print("Run failed:", config_file)
            error = format_error(exc)

        result = {
            "config_file": str(config_file),
            "state": state,
            "error": error,
        }
        results.append(result)

    failed_runs = sum(1 for result in results if result["error"] is not None)
    if failed_runs:
        print(f"{failed_runs} battery run(s) failed.")

    return results


#- Main -#

# Run CLI
def main():
    args = build_parser()

    return run_battery(
        args.battery_path,
        reset_data=args.reset_data,
        reset_rb=args.reset_real_backend_info,
        overwrite=args.overwrite,
        stop_on_error=args.stop_on_error,
    )


if __name__ == "__main__":
    main()
