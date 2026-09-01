import argparse
import copy
from pathlib import Path
import traceback

from qgan_v2 import __version__
from qgan_v2.config.battery import create_battery_configs
from qgan_v2.config.loader import load_config_file, load_run_config
from qgan_v2.training.interrupter import Interrupter

#- Arguments -#

# Create CLI parser
def build_parser():
    parser = argparse.ArgumentParser(prog="qgan-v2", description="qGAN")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a battery config.")
    run_parser.add_argument("-p", "--battery-path", dest="battery_path", required=True, type=str)
    run_parser.add_argument("--reset-data", dest="reset_data", action="store_true")
    run_parser.add_argument(
        "--reset-real-backend-info",
        "--reset-rb",
        dest="reset_real_backend_info",
        action="store_true",
    )
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--stop-on-error", dest="stop_on_error", action="store_true")

    subparsers.add_parser("save-account", help="Interactively save IBM Runtime credentials.")

    return parser.parse_args()

#- Run configuration -#

# Apply runtime reset options
def apply_reset_options(
    config,
    reset_data=False,
    reset_real_backend_info=False,
):
    if reset_data:
        config["training"]["reset_data"] = True

    real_backend = config["backend"]["real"]
    if reset_real_backend_info and real_backend["info_storage"] == "run":
        real_backend["reset_info"] = True

    return config


# Run one training config
def run_train(
    config_path,
    reset_data=False,
    reset_real_backend_info=False,
    interrupter=None,
):
    from qgan_v2.train import train

    config = load_run_config(config_path)
    config = apply_reset_options(
        config,
        reset_data=reset_data,
        reset_real_backend_info=reset_real_backend_info,
    )
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

    # Reset shared real backend info once before running the battery.
    if reset_rb:
        from qgan_v2.execution.backend import reset_real_backend_info

        real_backend_options = get_battery_real_backend_options(battery_path)
        if real_backend_options["info_storage"] == "shared":
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
        traceback_file = Path(config_file).with_name("error_traceback.txt")
        if traceback_file.exists():
            traceback_file.unlink()

        try:
            state = run_train(
                config_file,
                reset_data=reset_data,
                reset_real_backend_info=reset_rb,
                interrupter=interrupter,
            )
        except Exception as exc:
            error = format_error(exc)
            traceback_file.write_text(error["traceback"], encoding="utf-8")

            if stop_on_error:
                print("Traceback saved:", traceback_file)
                raise

            print("Run failed:", config_file)
            print(f"   Error: {error['type']}: {error['message']}")
            print(f"   Traceback saved: {traceback_file}")

        result = {
            "config_file": str(config_file),
            "state": state,
            "error": error,
        }
        results.append(result)

    failed_runs = sum(1 for result in results if result["error"] is not None)
    succeeded_runs = len(results) - failed_runs
    print(f"{succeeded_runs} battery run(s) succeeded.")
    if failed_runs:
        print(f"{failed_runs} battery run(s) failed.")

    return results


#- Main -#

# Run CLI
def main():
    args = build_parser()

    if args.command == "save-account":
        from qgan_v2.execution.account import save_runtime_account

        save_runtime_account()
        return 0

    if args.command == "run":
        run_battery(
            args.battery_path,
            reset_data=args.reset_data,
            reset_rb=args.reset_real_backend_info,
            overwrite=args.overwrite,
            stop_on_error=args.stop_on_error,
        )

    return 0


if __name__ == "__main__":
    main()
