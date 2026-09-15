"""High-level workflows used by interactive experiment clients.

The command-line entry point and notebooks should not need to duplicate the
details of locating configs, validating them, and loading their checkpoints.
This module keeps those operations together while leaving presentation choices
to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qgan_v2.config.loader import load_config_file, load_run_config
from qgan_v2.main import run_battery, run_train
from qgan_v2.storage.paths import get_training_data_filename, resolve_data_path
from qgan_v2.training.data import load_training_data_file


def _select_config_file(
    config_path: str | Path | None,
    fallback_directory: str | Path | None,
    pattern: str,
) -> Path:
    if config_path is not None:
        return Path(config_path)
    if fallback_directory is None:
        raise ValueError("A config path or fallback directory is required.")

    candidates = sorted(Path(fallback_directory).glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No configuration files matching {pattern!r} in {fallback_directory}."
        )
    return candidates[0]


@dataclass
class SingleExperiment:
    """A selected training config and its current in-memory state."""

    config_path: Path
    config: dict[str, Any]
    checkpoint: Path
    state: Any = None

    @classmethod
    def select(
        cls,
        config_path: str | Path | None = None,
        *,
        fallback_directory: str | Path | None = None,
    ) -> "SingleExperiment":
        selected = _select_config_file(config_path, fallback_directory, "*.yaml")
        config = load_run_config(selected)
        return cls(
            config_path=selected,
            config=config,
            checkpoint=get_training_data_filename(config),
        )

    @property
    def run_id(self) -> str:
        return str(self.config["run"]["id"])

    def run_or_load(
        self,
        *,
        run: bool = False,
        reset_data: bool = False,
        reset_real_backend_info: bool = False,
    ) -> Any:
        """Run the config or load its checkpoint, and return the state."""

        if run:
            self.state = run_train(
                str(self.config_path),
                reset_data=reset_data,
                reset_real_backend_info=reset_real_backend_info,
            )
        else:
            if not self.checkpoint.exists():
                raise FileNotFoundError(f"No checkpoint found: {self.checkpoint}")
            self.state = load_training_data_file(self.checkpoint)
        return self.state

    def summary(self) -> dict[str, Any]:
        return {
            "config": str(self.config_path),
            "run_id": self.run_id,
            "checkpoint": str(self.checkpoint),
            "checkpoint_exists": self.checkpoint.exists(),
        }


@dataclass
class BatteryExperiments:
    """Configs, checkpoints, and execution results for one battery."""

    battery_path: Path
    config_files: list[Path] = field(default_factory=list)
    valid_config_files: list[Path] = field(default_factory=list)
    invalid_config_files: list[tuple[Path, Exception]] = field(default_factory=list)
    missing_checkpoints: list[Path] = field(default_factory=list)
    states: dict[str, Any] = field(default_factory=dict)
    run_results: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def select(
        cls,
        battery_path: str | Path | None = None,
        *,
        fallback_directory: str | Path | None = None,
    ) -> "BatteryExperiments":
        selected = _select_config_file(
            battery_path,
            fallback_directory,
            "**/*.yaml",
        )
        return cls(battery_path=selected)

    def run_or_load(
        self,
        *,
        run: bool = False,
        reset_data: bool = False,
        reset_real_backend_info: bool = False,
        stop_on_error: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Optionally run the battery, then load all available checkpoints."""

        if run:
            self.run_results = run_battery(
                self.battery_path,
                reset_data=reset_data,
                reset_rb=reset_real_backend_info,
                stop_on_error=stop_on_error,
                overwrite=overwrite,
                keep_states=False,
            )
            self.config_files = [
                Path(result["config_file"]) for result in self.run_results
            ]
        else:
            battery_config = load_config_file(self.battery_path)
            data_path = battery_config["default_config_values"]["run"]["data_path"]
            self.config_files = sorted(resolve_data_path(data_path).glob("*/config.yaml"))

        self.valid_config_files = []
        self.invalid_config_files = []
        self.missing_checkpoints = []
        self.states = {}
        for config_path in self.config_files:
            try:
                config = load_run_config(config_path)
            except Exception as exc:  # Keep inspecting the rest of the battery.
                self.invalid_config_files.append((config_path, exc))
                continue

            self.valid_config_files.append(config_path)
            checkpoint = get_training_data_filename(config)
            if checkpoint.exists():
                self.states[config["run"]["id"]] = load_training_data_file(checkpoint)
            else:
                self.missing_checkpoints.append(checkpoint)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "battery": str(self.battery_path),
            "config_files": len(self.config_files),
            "valid_config_files": len(self.valid_config_files),
            "invalid_config_files": len(self.invalid_config_files),
            "loaded_states": len(self.states),
            "missing_checkpoints": len(self.missing_checkpoints),
        }

    def config_rows(self, *, limit: int | None = 20) -> list[dict[str, Any]]:
        paths = self.valid_config_files if limit is None else self.valid_config_files[:limit]
        rows = []
        for path in paths:
            config = load_run_config(path)
            run_id = config["run"]["id"]
            rows.append({
                "run_id": run_id,
                "done": get_training_data_filename(config).exists(),
                "loaded": run_id in self.states,
                "config_file": str(path),
            })
        return rows

    def implementation_rows(
        self,
        *,
        project_directory: str | Path | None = None,
        limit: int | None = 20,
    ) -> list[dict[str, Any]]:
        paths = self.valid_config_files if limit is None else self.valid_config_files[:limit]
        project_directory = (
            Path(project_directory).resolve()
            if project_directory is not None
            else None
        )
        rows = []
        for path in paths:
            config = load_run_config(path)
            displayed_path: Path | str = path
            if project_directory is not None:
                try:
                    displayed_path = path.resolve().relative_to(project_directory)
                except ValueError:
                    pass
            rows.append({
                "config_file": str(displayed_path),
                "implementation": config["implementation"]["name"],
                "preset": config["experiment"]["preset"],
                "gradient": config["experiment"]["gradient_method"],
            })
        return rows


def manage_runtime_account(*, save: bool = False, check: bool = False) -> int | None:
    """Interactively save credentials and/or return the backend count."""

    if not save and not check:
        return None

    from qiskit_ibm_runtime import QiskitRuntimeService

    if save:
        from getpass import getpass

        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=getpass("IBM Quantum API token: "),
            instance=None,
            set_as_default=True,
            overwrite=True,
            verify=True,
        )
    if check:
        return len(QiskitRuntimeService().backends())
    return None


__all__ = [
    "BatteryExperiments",
    "SingleExperiment",
    "manage_runtime_account",
]
