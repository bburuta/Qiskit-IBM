import os
import subprocess
import sys
from pathlib import Path

from qgan_v2.main import apply_reset_options


def run_cli(*args):
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    return subprocess.run(
        [sys.executable, "-m", "qgan_v2.main", *args],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_help_does_not_import_training_dependencies():
    result = run_cli("--help")

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--version" in result.stdout
    assert "run" in result.stdout
    assert "save-account" in result.stdout


def test_run_help_shows_battery_options():
    result = run_cli("run", "--help")

    assert result.returncode == 0
    assert "--battery-path" in result.stdout
    assert "--reset-data" in result.stdout
    assert "--reset-real-backend-info" in result.stdout
    assert "--stop-on-error" in result.stdout


def test_cli_version():
    result = run_cli("--version")

    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_reset_real_backend_info_flag_applies_to_run_storage_only():
    run_storage_config = {
        "training": {},
        "backend": {
            "real": {
                "info_storage": "run",
                "reset_info": False,
            },
        },
    }
    shared_storage_config = {
        "training": {},
        "backend": {
            "real": {
                "id": "ibm_basquecountry",
                "info_storage": "shared",
                "reset_info": False,
            },
        },
    }

    apply_reset_options(run_storage_config, reset_real_backend_info=True)
    apply_reset_options(shared_storage_config, reset_real_backend_info=True)

    assert run_storage_config["backend"]["real"]["reset_info"] is True
    assert shared_storage_config["backend"]["real"]["reset_info"] is False
