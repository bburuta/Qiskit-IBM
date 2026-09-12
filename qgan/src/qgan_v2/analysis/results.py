from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_GROUP_FIELDS = (
    "preset",
    "implementation",
    "packing",
    "execution_type",
    "gradient_method",
    "n_qubits",
    "random_circuit",
    "randomness",
    "batch_size",
    "eval_batch_size",
    "eval_method",
    "learning_rate",
    "max_iterations",
    "gen_iterations",
    "disc_iterations",
    "precision",
    "run_device",
    "simulator_device",
    "noiseless_method",
    "noisy_method",
    "noisy_backend_mapping",
    "real_backend",
    "resilience_level",
    "dynamical_decoupling",
)


DEFAULT_PAIR_FIELDS = (
    "preset",
    "implementation",
    "gradient_method",
    "n_qubits",
    "random_circuit",
    "randomness",
    "batch_size",
    "eval_batch_size",
    "eval_method",
    "learning_rate",
    "max_iterations",
    "precision",
    "run_device",
    "simulator_device",
    "seed",
)


# Fields that define a scientific comparison while deliberately ignoring the
# machine used to produce an otherwise equivalent simulator checkpoint.
SCIENTIFIC_RUN_FIELDS = (
    "preset",
    "implementation",
    "packing",
    "execution_type",
    "gradient_method",
    "n_qubits",
    "random_circuit",
    "randomness",
    "batch_size",
    "eval_batch_size",
    "eval_method",
    "learning_rate",
    "max_iterations",
    "gen_iterations",
    "disc_iterations",
    "precision",
    "noiseless_method",
    "noisy_method",
    "noisy_backend_mapping",
)


SCIENTIFIC_PAIR_FIELDS = (*SCIENTIFIC_RUN_FIELDS, "seed")


SUMMARY_LABEL_FIELDS = (
    "preset",
    "implementation",
    "packing",
    "execution_type",
    "gradient_method",
    "n_qubits",
    "randomness",
    "device_mode",
    "run_device",
    "simulator_device",
    "eval_method",
)


METADATA_PATHS = {
    "run_id": "run.id",
    "label": "run.label",
    "seed": "run.seed",
    "run_device": "run.device",
    "preset": "experiment.implementation",
    "implementation": "implementation.name",
    "packing": "implementation.discriminator_packing",
    "execution_type": "experiment.execution_type",
    "gradient_method": "experiment.gradient_method",
    "n_qubits": "experiment.n_qubits",
    "random_circuit": "encoding.random_circuit",
    "randomness": "encoding.randomness",
    "batch_size": "encoding.batch_size",
    "eval_batch_size": "encoding.eval_batch_size",
    "eval_method": "encoding.eval_method",
    "encoding_type": "encoding.type",
    "dataset_type": "dataset.type",
    "dataset_source": "dataset.source",
    "learning_rate": "training.learning_rate",
    "max_iterations": "training.max_iterations",
    "gen_iterations": "training.gen_iterations",
    "disc_iterations": "training.disc_iterations",
    "precision": "backend.precision",
    "simulator_device": "backend.simulator.device",
    "simulator_device_name": "backend.simulator.device_name",
    "cpu_threads": "backend.simulator.max_parallel_threads",
    "encoding_threads": "encoding.max_parallel_threads",
    "noiseless_method": "backend.simulator.noiseless_method",
    "noisy_method": "backend.simulator.noisy_method",
    "noisy_backend_mapping": "backend.simulator.noisy_backend_mapping",
    "real_backend": "backend.real.name",
    "real_backend_id": "backend.real.id",
    "resilience_level": "backend.real.estimator.resilience_level",
    "dynamical_decoupling": "backend.real.estimator.dynamical_decoupling.enable",
}


@dataclass
class RunResult:
    path: Path
    config: dict[str, Any]
    metadata: dict[str, Any]
    eval: dict[int, float]
    gloss: dict[int, float]
    dloss: dict[int, float]
    times: dict[int, float]
    status: str = "ok"
    error: str | None = None

    @property
    def run_id(self) -> str:
        return str(self.metadata.get("run_id") or self.path.name)

    @property
    def seed(self) -> Any:
        return self.metadata.get("seed")

    def metric(self, name: str) -> dict[int, float]:
        if name == "eval":
            return self.eval
        if name == "gloss":
            return self.gloss
        if name == "dloss":
            return self.dloss
        if name == "times":
            return self.times
        raise ValueError(f"Unknown metric: {name}")


def get_nested(mapping: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    cursor: Any = mapping
    for key in dotted_path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _metadata_from_config(config: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        name: get_nested(config, path)
        for name, path in METADATA_PATHS.items()
    }
    return _add_device_metadata(metadata)


def _add_device_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    execution_type = metadata.get("execution_type")
    run_device = metadata.get("run_device")
    simulator_device = metadata.get("simulator_device")

    if execution_type == "real":
        compute_device = "RH"
    elif execution_type == "fake_real":
        compute_device = "fake_real"
    else:
        compute_device = simulator_device or run_device

    if compute_device is None:
        device_mode = execution_type
    elif execution_type is None:
        device_mode = compute_device
    else:
        device_mode = f"{compute_device} {execution_type}"

    implementation = metadata.get("implementation")
    packing = metadata.get("packing")
    implementation_packing = implementation
    if implementation == "runtime_packed" and packing:
        implementation_packing = f"{implementation}/{packing}"

    if compute_device == "CPU" and metadata.get("cpu_threads"):
        execution_environment = f"CPU/{metadata['cpu_threads']} threads"
    else:
        execution_environment = compute_device

    metadata.update({
        "compute_device": compute_device,
        "device_mode": device_mode,
        "execution_environment": execution_environment,
        "implementation_packing": implementation_packing,
    })
    return metadata


def _clean_metric(metric: dict[Any, Any] | None) -> dict[int, float]:
    if not metric:
        return {}
    return {
        int(epoch): float(value)
        for epoch, value in sorted(metric.items(), key=lambda item: int(item[0]))
        if value is not None
    }


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return False


_PREFERRED_CATEGORY_ORDER = {
    "base": 0,
    "ang": 1,
    "amp": 2,
    "noiseless": 0,
    "noisy": 1,
    "real": 2,
    "fake_real": 3,
    "PSR": 0,
    "REG": 1,
    "SPSA": 2,
}


def _sort_key(value: Any) -> tuple[int, float, str]:
    """Sort numeric and common thesis categories in their logical order."""

    if _is_finite_number(value):
        return (0, float(value), "")
    if value in _PREFERRED_CATEGORY_ORDER:
        return (1, float(_PREFERRED_CATEGORY_ORDER[value]), "")
    return (2, 0.0, str(value))


def _count_values(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "numel"):
        return int(value.numel())
    if isinstance(value, dict):
        total = 0
        found = False
        for item in value.values():
            count = _count_values(item)
            if count is not None:
                total += count
                found = True
        return total if found else None
    try:
        return int(np.asarray(value).size)
    except Exception:
        return None


def _add_state_metadata(metadata: dict[str, Any], state: Any, config: dict[str, Any]) -> dict[str, Any]:
    generator_count = _count_values(getattr(state, "init_gen_params", None))
    discriminator_count = _count_values(getattr(state, "model_d_state", None))
    total_count = None
    if generator_count is not None and discriminator_count is not None:
        total_count = generator_count + discriminator_count

    n_qubits = get_nested(config, "experiment.n_qubits")
    batch_size = get_nested(config, "encoding.batch_size")
    implementation = get_nested(config, "implementation.name")
    approx_packed_width = n_qubits
    if implementation == "runtime_packed" and n_qubits is not None and batch_size is not None:
        approx_packed_width = n_qubits * batch_size

    metadata.update({
        "generator_parameter_count": generator_count,
        "discriminator_parameter_count": discriminator_count,
        "total_parameter_count": total_count,
        "approx_packed_width": approx_packed_width,
    })

    # These counters are not present in the original checkpoints, but keeping
    # the loader forward-compatible makes circuit/submission cost plots work as
    # soon as newer training states persist them.
    metrics = getattr(state, "metrics", None)
    for name in ("primitive_calls", "circuit_evaluations", "submissions"):
        value = getattr(state, name, None)
        if value is None and metrics is not None:
            value = getattr(metrics, name, None)
        if value is not None:
            metadata[name] = value
    return metadata


def _load_training_state(training_data_file: Path):
    import torch

    return torch.load(training_data_file, weights_only=False, map_location="cpu")


def _load_config(config_file: Path) -> dict[str, Any]:
    try:
        from qgan_v2.config.loader import load_run_config

        return load_run_config(config_file)
    except Exception:
        return {}


def _read_error_file(run_dir: Path) -> str | None:
    for filename in ("error_traceback.txt", "error.txt"):
        error_file = run_dir / filename
        if error_file.exists():
            try:
                return error_file.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                return None
    return None


def load_result(
    training_data_file: str | Path,
    *,
    config: dict[str, Any] | None = None,
) -> RunResult:
    training_data_file = Path(training_data_file)
    state = _load_training_state(training_data_file)
    # The adjacent normalized YAML may contain metadata fields introduced after
    # an older checkpoint was written (for example ``random_circuit``).  Prefer
    # it when the directory loader already validated it; standalone calls still
    # fall back to the exact saved training config.
    config = config or state.config
    metrics = state.metrics
    metadata = _add_state_metadata(_metadata_from_config(config), state, config)
    return RunResult(
        path=training_data_file.parent,
        config=config,
        metadata=metadata,
        eval=_clean_metric(metrics.eval),
        gloss=_clean_metric(metrics.gloss),
        dloss=_clean_metric(metrics.dloss),
        times=_clean_metric(metrics.times),
    )


def load_results(
    data_path: str | Path,
    *,
    include_manual_estimator: bool = False,
    include_failed_configs: bool = True,
) -> list[RunResult]:
    data_path = Path(data_path)
    results: list[RunResult] = []

    for config_file in sorted(data_path.glob("*/config.yaml")):
        run_dir = config_file.parent
        training_data_file = run_dir / "training_data.pth"
        config = _load_config(config_file)
        metadata = _metadata_from_config(config) if config else {"run_id": run_dir.name}

        if training_data_file.exists():
            try:
                result = load_result(training_data_file, config=config or None)
                saved_error = _read_error_file(run_dir)
                if saved_error:
                    result.error = saved_error
            except Exception as exc:
                result = RunResult(
                    path=run_dir,
                    config=config,
                    metadata=metadata,
                    eval={},
                    gloss={},
                    dloss={},
                    times={},
                    status="load_error",
                    error=str(exc),
                )
        elif include_failed_configs:
            saved_error = _read_error_file(run_dir)
            result = RunResult(
                path=run_dir,
                config=config,
                metadata=metadata,
                eval={},
                gloss={},
                dloss={},
                times={},
                status="missing_training_data",
                error=saved_error or "training_data.pth is missing",
            )
        else:
            continue

        if (
            not include_manual_estimator
            and result.metadata.get("implementation") == "manual_estimator"
        ):
            continue
        results.append(result)

    return results


def load_result_sets(*data_paths: str | Path, **kwargs: Any) -> list[RunResult]:
    results: list[RunResult] = []
    for data_path in data_paths:
        results.extend(load_results(data_path, **kwargs))
    return results


def filter_results(results: Iterable[RunResult], **filters: Any) -> list[RunResult]:
    selected = []
    for result in results:
        keep = True
        for field, expected in filters.items():
            value = result.metadata.get(field)
            if isinstance(expected, (set, tuple, list)):
                keep = value in expected
            else:
                keep = value == expected
            if not keep:
                break
        if keep:
            selected.append(result)
    return selected


def unique_values(results: Iterable[RunResult], field: str) -> list[Any]:
    values = {
        result.metadata.get(field)
        for result in results
        if result.metadata.get(field) is not None
    }
    return sorted(values, key=_sort_key)


def group_key(result: RunResult, fields: Iterable[str] = DEFAULT_GROUP_FIELDS) -> tuple[Any, ...]:
    return tuple(result.metadata.get(field) for field in fields)


def comparable_groups(
    results: Iterable[RunResult],
    fields: Iterable[str] = DEFAULT_GROUP_FIELDS,
) -> dict[tuple[Any, ...], list[RunResult]]:
    groups: dict[tuple[Any, ...], list[RunResult]] = defaultdict(list)
    for result in results:
        groups[group_key(result, fields)].append(result)
    return dict(groups)


def completed_epoch_count(result: RunResult, metric: str = "eval") -> int:
    """Return the number of recorded epochs for a metric."""

    return len(result.metric(metric))


def is_completed(
    result: RunResult,
    *,
    expected_epochs: int | None = None,
    metric: str = "eval",
) -> bool:
    """Whether a checkpoint reached its requested epoch budget.

    Both the number and the largest epoch index are checked so a sparse or
    damaged metric dictionary cannot accidentally be classified as complete.
    """

    if result.status != "ok":
        return False
    series = result.metric(metric)
    if expected_epochs is None:
        expected_epochs = result.metadata.get("max_iterations")
    if expected_epochs is None:
        return bool(series)
    expected_epochs = int(expected_epochs)
    return (
        len(series) >= expected_epochs
        and bool(series)
        and max(series) >= expected_epochs - 1
    )


def select_usable_results(
    results: Iterable[RunResult],
    *,
    metric: str = "eval",
    execution_types: Iterable[str] | None = None,
) -> list[RunResult]:
    """Select successfully loaded checkpoints with at least one metric value."""

    allowed = None if execution_types is None else set(execution_types)
    return [
        result
        for result in results
        if result.status == "ok"
        and bool(result.metric(metric))
        and (allowed is None or result.metadata.get("execution_type") in allowed)
    ]


def select_completed_results(
    results: Iterable[RunResult],
    *,
    expected_epochs: int | None = None,
    metric: str = "eval",
    execution_types: Iterable[str] | None = None,
) -> list[RunResult]:
    """Select only runs that reached a common or configured epoch budget."""

    allowed = None if execution_types is None else set(execution_types)
    return [
        result
        for result in results
        if is_completed(result, expected_epochs=expected_epochs, metric=metric)
        and (allowed is None or result.metadata.get("execution_type") in allowed)
    ]


def deduplicate_simulator_runs(
    results: Iterable[RunResult],
    *,
    fields: Iterable[str] = SCIENTIFIC_PAIR_FIELDS,
    preferred_device: str = "CPU",
) -> list[RunResult]:
    """Collapse CPU/GPU copies of the same scientific simulator run.

    Device is an execution detail for convergence-quality analysis.  The most
    complete checkpoint wins; ties prefer ``preferred_device`` and then the
    lexicographically first run id for deterministic notebook output.  Real and
    fake-real runs are never collapsed with simulator runs.
    """

    chosen: dict[tuple[Any, ...], RunResult] = {}
    passthrough: list[RunResult] = []
    fields = tuple(fields)

    def rank(result: RunResult) -> tuple[int, int, int, str]:
        series = result.eval
        return (
            int(result.status == "ok"),
            len(series),
            int(result.metadata.get("simulator_device") == preferred_device),
            # Reverse lexical preference is handled explicitly below.
            result.run_id,
        )

    for result in results:
        if result.metadata.get("execution_type") not in {"noiseless", "noisy"}:
            passthrough.append(result)
            continue
        key_values = []
        for field in fields:
            value = result.metadata.get(field)
            # Training disables the randomizer when randomness is zero, so old
            # configs that omit random_circuit and newer configs that retain its
            # unused value are scientifically identical.
            if field == "random_circuit" and result.metadata.get("randomness") == 0:
                value = None
            key_values.append(value)
        key = tuple(key_values)
        current = chosen.get(key)
        if current is None:
            chosen[key] = result
            continue
        candidate_rank = rank(result)[:3]
        current_rank = rank(current)[:3]
        if candidate_rank > current_rank or (
            candidate_rank == current_rank and result.run_id < current.run_id
        ):
            chosen[key] = result

    return sorted([*chosen.values(), *passthrough], key=lambda result: result.run_id)


def select_main_convergence_results(
    results: Iterable[RunResult],
    *,
    expected_epochs: int = 1000,
    preferred_device: str = "CPU",
) -> list[RunResult]:
    """Apply the thesis rules for the main simulated convergence population."""

    completed = select_completed_results(
        results,
        expected_epochs=expected_epochs,
        execution_types=("noiseless", "noisy"),
    )
    return deduplicate_simulator_runs(completed, preferred_device=preferred_device)


def factor_sweep_groups(
    results: Iterable[RunResult],
    *,
    factor: str,
    required_levels: Iterable[Any],
    required_seeds: Iterable[Any] = (0, 1, 2),
    group_fields: Iterable[str] | None = None,
) -> dict[tuple[Any, ...], list[RunResult]]:
    """Return only configuration groups with a complete factor × seed grid."""

    required_levels = set(required_levels)
    required_seeds = set(required_seeds)
    if group_fields is None:
        group_fields = tuple(
            field
            for field in SCIENTIFIC_RUN_FIELDS
            if field not in {factor, "max_iterations"}
        )
    else:
        group_fields = tuple(group_fields)

    grouped: dict[tuple[Any, ...], list[RunResult]] = defaultdict(list)
    for result in results:
        grouped[tuple(result.metadata.get(field) for field in group_fields)].append(result)

    complete: dict[tuple[Any, ...], list[RunResult]] = {}
    for key, runs in grouped.items():
        available = {
            (run.metadata.get(factor), run.seed)
            for run in runs
        }
        required = {
            (level, seed)
            for level in required_levels
            for seed in required_seeds
        }
        if required <= available:
            complete[key] = [
                run
                for run in runs
                if run.metadata.get(factor) in required_levels and run.seed in required_seeds
            ]
    return complete


def evaluation_metric_groups(results: Iterable[RunResult]) -> dict[Any, list[RunResult]]:
    """Split runs so incompatible evaluation metrics are never pooled."""

    groups: dict[Any, list[RunResult]] = defaultdict(list)
    for result in results:
        groups[result.metadata.get("eval_method")].append(result)
    return dict(groups)


def metric_arrays(result: RunResult, metric: str = "eval") -> tuple[np.ndarray, np.ndarray]:
    series = result.metric(metric)
    epochs = np.asarray(sorted(series), dtype=float)
    values = np.asarray([series[int(epoch)] for epoch in epochs], dtype=float)
    return epochs, values


def elapsed_arrays(result: RunResult, metric: str = "eval") -> tuple[np.ndarray, np.ndarray]:
    epochs, values = metric_arrays(result, metric)
    if len(epochs) == 0:
        return epochs, values

    elapsed = []
    running = 0.0
    for epoch in epochs.astype(int):
        running += float(result.times.get(epoch, np.nan))
        elapsed.append(running)
    return np.asarray(elapsed, dtype=float), values


def truncate_result(result: RunResult, max_epoch: int) -> RunResult:
    def truncate_metric(metric: dict[int, float]) -> dict[int, float]:
        return {
            epoch: value
            for epoch, value in metric.items()
            if epoch <= max_epoch
        }

    return RunResult(
        path=result.path,
        config=result.config,
        metadata=dict(result.metadata),
        eval=truncate_metric(result.eval),
        gloss=truncate_metric(result.gloss),
        dloss=truncate_metric(result.dloss),
        times=truncate_metric(result.times),
        status=result.status,
        error=result.error,
    )


def truncate_results(results: Iterable[RunResult], max_epoch: int) -> list[RunResult]:
    return [truncate_result(result, max_epoch) for result in results]


def aggregate_metric(
    runs: Iterable[RunResult],
    *,
    metric: str = "eval",
    x_axis: str = "epoch",
    center: str = "median",
    spread: str = "iqr",
    elapsed_points: int = 200,
) -> dict[str, np.ndarray]:
    runs = [run for run in runs if run.metric(metric)]
    if not runs:
        return {
            "x": np.asarray([]),
            "center": np.asarray([]),
            "low": np.asarray([]),
            "high": np.asarray([]),
            "count": np.asarray([]),
        }

    if x_axis == "epoch":
        x = np.asarray(
            sorted({epoch for run in runs for epoch in run.metric(metric)}),
            dtype=float,
        )
        data = np.full((len(runs), len(x)), np.nan, dtype=float)
        x_to_index = {int(epoch): index for index, epoch in enumerate(x)}
        for row, run in enumerate(runs):
            for epoch, value in run.metric(metric).items():
                data[row, x_to_index[int(epoch)]] = value
    elif x_axis == "elapsed":
        elapsed_series = []
        for run in runs:
            run_x, run_y = elapsed_arrays(run, metric)
            valid = np.isfinite(run_x) & np.isfinite(run_y)
            if valid.sum() >= 2:
                elapsed_series.append((run_x[valid], run_y[valid]))

        if not elapsed_series:
            return {
                "x": np.asarray([]),
                "center": np.asarray([]),
                "low": np.asarray([]),
                "high": np.asarray([]),
                "count": np.asarray([]),
            }

        max_elapsed = min(float(np.max(run_x)) for run_x, _ in elapsed_series)
        if not np.isfinite(max_elapsed):
            return {
                "x": np.asarray([]),
                "center": np.asarray([]),
                "low": np.asarray([]),
                "high": np.asarray([]),
                "count": np.asarray([]),
            }

        x = np.linspace(0.0, max_elapsed, elapsed_points)
        data = np.full((len(elapsed_series), len(x)), np.nan, dtype=float)
        for row, (run_x, run_y) in enumerate(elapsed_series):
            data[row] = np.interp(x, run_x, run_y, left=np.nan, right=np.nan)
    else:
        raise ValueError("x_axis must be 'epoch' or 'elapsed'")

    if center == "median":
        center_values = np.nanmedian(data, axis=0)
    elif center == "mean":
        center_values = np.nanmean(data, axis=0)
    else:
        raise ValueError("center must be 'median' or 'mean'")

    if spread == "iqr":
        low = np.nanpercentile(data, 25, axis=0)
        high = np.nanpercentile(data, 75, axis=0)
    elif spread == "std":
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        low = mean - std
        high = mean + std
    else:
        raise ValueError("spread must be 'iqr' or 'std'")

    return {
        "x": x,
        "center": center_values,
        "low": low,
        "high": high,
        "count": np.sum(np.isfinite(data), axis=0),
    }


def run_summary(
    result: RunResult,
    *,
    last_window: int = 100,
    last_fraction: float | None = None,
) -> dict[str, Any]:
    """Build a thesis-ready row for one independent run.

    ``last_window`` is fixed at 100 evaluations by default.  ``last_fraction``
    remains available for old notebooks but should not be used to compare runs
    with different training budgets.
    """

    eval_values = np.asarray(list(result.eval.values()), dtype=float)
    eval_epochs = np.asarray(list(result.eval.keys()), dtype=int)
    time_values = np.asarray(list(result.times.values()), dtype=float)

    finite_eval = np.isfinite(eval_values)

    if finite_eval.any():
        finite_indexes = np.flatnonzero(finite_eval)
        best_index = int(finite_indexes[np.argmin(eval_values[finite_indexes])])
        if last_fraction is not None:
            window = max(1, int(np.ceil(len(eval_values) * last_fraction)))
        else:
            window = max(1, min(int(last_window), len(eval_values)))
        best_eval = float(eval_values[best_index])
        best_epoch = int(eval_epochs[best_index])
        final_eval = float(eval_values[finite_indexes[-1]])
        initial_eval = float(eval_values[finite_indexes[0]])
        last_window_median_eval = float(np.nanmedian(eval_values[-window:]))
        improvement = initial_eval - best_eval
        finite_eval_values = eval_values[finite_eval]
        evaluation_step_volatility = (
            float(np.median(np.abs(np.diff(finite_eval_values))))
            if len(finite_eval_values) >= 2
            else np.nan
        )
    else:
        best_eval = np.nan
        best_epoch = None
        final_eval = np.nan
        initial_eval = np.nan
        last_window_median_eval = np.nan
        improvement = np.nan
        evaluation_step_volatility = np.nan

    if len(time_values):
        total_time = float(np.nansum(time_values))
        time_per_epoch = total_time / len(time_values)
        median_time_per_epoch = float(np.nanmedian(time_values))
        time_per_epoch_q25, time_per_epoch_q75 = np.nanpercentile(time_values, [25, 75])
        projected_1000_epoch_time = median_time_per_epoch * 1000
    else:
        total_time = np.nan
        time_per_epoch = np.nan
        median_time_per_epoch = np.nan
        time_per_epoch_q25 = np.nan
        time_per_epoch_q75 = np.nan
        projected_1000_epoch_time = np.nan

    if best_epoch is not None:
        time_to_best_eval = float(
            sum(value for epoch, value in result.times.items() if epoch <= best_epoch)
        )
    else:
        time_to_best_eval = np.nan

    return {
        **result.metadata,
        "path": str(result.path),
        "status": result.status,
        "completed_epochs": len(eval_values),
        "requested_epochs": result.metadata.get("max_iterations"),
        "completed_requested_budget": is_completed(result),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "initial_eval": initial_eval,
        "last_window_median_eval": last_window_median_eval,
        "improvement": improvement,
        "evaluation_step_volatility": evaluation_step_volatility,
        "epoch_of_best_eval": best_epoch,
        "total_time": total_time,
        "time_per_epoch": time_per_epoch,
        "median_time_per_epoch": median_time_per_epoch,
        "time_per_epoch_q25": float(time_per_epoch_q25),
        "time_per_epoch_q75": float(time_per_epoch_q75),
        "projected_1000_epoch_time": projected_1000_epoch_time,
        "time_to_best_eval": time_to_best_eval,
        "error": result.error,
    }


def results_table(
    results: Iterable[RunResult],
    *,
    last_window: int = 100,
) -> list[dict[str, Any]]:
    return [run_summary(result, last_window=last_window) for result in results]


def grouped_performance_table(
    results: Iterable[RunResult],
    *,
    fields: Iterable[str],
    last_window: int = 100,
) -> list[dict[str, Any]]:
    """Aggregate independent run summaries without treating epochs as samples."""

    fields = tuple(fields)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in results_table(results, last_window=last_window):
        grouped[tuple(row.get(field) for field in fields)].append(row)

    output = []
    metric_names = (
        "best_eval",
        "final_eval",
        "last_window_median_eval",
        "epoch_of_best_eval",
        "evaluation_step_volatility",
        "median_time_per_epoch",
    )
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        summary = {
            **dict(zip(fields, key)),
            "completed_runs": len(rows),
            "seed_count": len({row.get("seed") for row in rows}),
            "seeds": tuple(sorted({row.get("seed") for row in rows}, key=str)),
        }
        for metric_name in metric_names:
            values = np.asarray(
                [row.get(metric_name) for row in rows if _is_finite_number(row.get(metric_name))],
                dtype=float,
            )
            if len(values):
                summary[f"median_{metric_name}"] = float(np.median(values))
                q25, q75 = np.percentile(values, [25, 75])
                summary[f"q25_{metric_name}"] = float(q25)
                summary[f"q75_{metric_name}"] = float(q75)
            else:
                summary[f"median_{metric_name}"] = np.nan
                summary[f"q25_{metric_name}"] = np.nan
                summary[f"q75_{metric_name}"] = np.nan
        output.append(summary)
    return output


def representative_runs(
    results: Iterable[RunResult],
    *,
    metric_name: str = "best_eval",
    lower_is_better: bool = True,
) -> dict[str, RunResult]:
    scored = []
    for result in results:
        value = run_summary(result).get(metric_name)
        if _is_finite_number(value):
            scored.append((float(value), result))

    if not scored:
        return {}

    scored.sort(key=lambda item: item[0], reverse=not lower_is_better)
    values = np.asarray([value for value, _ in scored], dtype=float)
    median_value = float(np.median(values))
    median_index = min(
        range(len(scored)),
        key=lambda index: abs(scored[index][0] - median_value),
    )

    return {
        "best": scored[0][1],
        "median": scored[median_index][1],
        "worst": scored[-1][1],
    }


def classify_run_status(
    result: RunResult,
    *,
    expected_epochs: int | None = None,
) -> str:
    """Classify completion and common feasibility failures for reporting."""

    if is_completed(result, expected_epochs=expected_epochs):
        return "complete"
    error = (result.error or "").lower()
    if any(token in error for token in (
        "out of memory", "oom", "cannot allocate memory", "bad_alloc", "memory allocation",
    )):
        return "out_of_memory"
    if any(token in error for token in ("time limit", "timed out", "timeout", "cancelled")):
        return "time_limit"
    if result.status == "ok" and result.eval:
        return "partial"
    if result.status == "ok" and not result.eval:
        return "empty_checkpoint"
    if result.status == "load_error":
        return "load_error"
    if result.status == "missing_training_data" and result.error == "training_data.pth is missing":
        return "not_started_or_missing"
    if result.status == "missing_training_data":
        return "failed"
    return result.status


def feasibility_table(
    results: Iterable[RunResult],
    *,
    fields: Iterable[str] = ("execution_type", "gradient_method", "n_qubits"),
    expected_epochs: int | None = None,
) -> list[dict[str, Any]]:
    """Count complete, partial, missing, and failed runs by experimental cell."""

    fields = tuple(fields)
    grouped: dict[tuple[Any, ...], list[RunResult]] = defaultdict(list)
    for result in results:
        grouped[tuple(result.metadata.get(field) for field in fields)].append(result)

    rows = []
    statuses = (
        "complete",
        "partial",
        "out_of_memory",
        "time_limit",
        "load_error",
        "failed",
        "empty_checkpoint",
        "not_started_or_missing",
    )
    for key, runs in sorted(grouped.items(), key=lambda item: str(item[0])):
        run_statuses = [
            classify_run_status(run, expected_epochs=expected_epochs)
            for run in runs
        ]
        row = {**dict(zip(fields, key)), "total": len(runs)}
        row.update({status: run_statuses.count(status) for status in statuses})
        row["completion_rate"] = row["complete"] / row["total"] if row["total"] else np.nan
        rows.append(row)
    return rows


def save_figure(
    fig,
    output_dir: str | Path,
    stem: str,
    *,
    formats: Iterable[str] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Export a figure in thesis (PDF) and preview (PNG) formats."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for extension in formats:
        extension = str(extension).lstrip(".")
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(path)
    return saved


def _label(result: RunResult, fields: Iterable[str]) -> str:
    parts = []
    for field in fields:
        value = result.metadata.get(field)
        if value is not None:
            parts.append(f"{field}={value}")
    return ", ".join(parts) or result.run_id


def plot_convergence(
    runs: Iterable[RunResult],
    *,
    metric: str = "eval",
    x_axis: str = "epoch",
    center: str = "median",
    spread: str = "iqr",
    label: str | None = None,
    color: str | None = None,
    linestyle: str = "-",
    show_individual: bool = True,
    ax=None,
):
    import matplotlib.pyplot as plt

    runs = [run for run in runs if run.metric(metric)]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if show_individual:
        for run in runs:
            x, y = elapsed_arrays(run, metric) if x_axis == "elapsed" else metric_arrays(run, metric)
            ax.plot(
                x,
                y,
                color=color or "0.35",
                alpha=0.18,
                linewidth=0.8,
                linestyle=linestyle,
            )

    aggregate = aggregate_metric(
        runs,
        metric=metric,
        x_axis=x_axis,
        center=center,
        spread=spread,
    )
    if len(aggregate["x"]):
        ax.fill_between(
            aggregate["x"],
            aggregate["low"],
            aggregate["high"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            aggregate["x"],
            aggregate["center"],
            color=color,
            linewidth=2.2,
            linestyle=linestyle,
            label=label or f"{center} {metric}",
        )
    else:
        ax.text(0.5, 0.5, "No metric data", transform=ax.transAxes, ha="center")

    ax.set_xlabel("Elapsed time (s)" if x_axis == "elapsed" else "Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    return ax


def plot_convergence_comparison(
    results: Iterable[RunResult],
    *,
    compare_by: str,
    filters: dict[str, Any] | None = None,
    metric: str = "eval",
    x_axis: str = "epoch",
    center: str = "median",
    spread: str = "iqr",
    ax=None,
):
    import matplotlib.pyplot as plt

    selected = filter_results(results, **(filters or {}))
    values = unique_values(selected, compare_by)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    if not values:
        ax.text(0.5, 0.5, "No matching runs", transform=ax.transAxes, ha="center")
        return ax

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, value in enumerate(values):
        group = filter_results(selected, **{compare_by: value})
        plot_convergence(
            group,
            metric=metric,
            x_axis=x_axis,
            center=center,
            spread=spread,
            label=f"{compare_by}={value}",
            color=colors[index % len(colors)],
            show_individual=True,
            ax=ax,
        )

    ax.legend()
    return ax


def plot_training_dynamics_comparison(
    results: Iterable[RunResult],
    *,
    compare_by: str,
    filters: dict[str, Any] | None = None,
    center: str = "median",
    spread: str = "iqr",
    x_axis: str = "epoch",
    axes=None,
):
    """Plot adversarial losses above evaluation in the legacy two-panel style.

    The legacy metric colors are blue for generator loss, red for discriminator
    loss, and orange for evaluation. When several comparison groups share a
    panel, shades within each metric's color family identify the groups.
    Individual runs remain faint and each aggregate retains its IQR shadow.
    """

    import matplotlib.pyplot as plt

    selected = filter_results(results, **(filters or {}))
    values = unique_values(selected, compare_by)
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    else:
        axes = np.asarray(axes).reshape(-1)
        if len(axes) != 2:
            raise ValueError("axes must contain exactly two Matplotlib axes")
        fig = axes[0].figure

    if not values:
        for ax in axes:
            ax.text(0.5, 0.5, "No matching runs", transform=ax.transAxes, ha="center")
        return fig, axes

    from matplotlib.colors import to_rgb

    def metric_shades(base_color: str) -> list[tuple[float, float, float]]:
        base = np.asarray(to_rgb(base_color), dtype=float)
        if len(values) == 1:
            return [tuple(base)]
        # Move from a lighter tint to the original color. This keeps groups
        # distinguishable without changing the metric's visual identity.
        white_mix = np.linspace(0.45, 0.0, len(values))
        return [tuple((1.0 - mix) * base + mix) for mix in white_mix]

    generator_colors = metric_shades("#0094f0")
    discriminator_colors = metric_shades("C3")
    evaluation_colors = metric_shades("#ffaf01")
    for index, value in enumerate(values):
        group = filter_results(selected, **{compare_by: value})
        plot_convergence(
            group,
            metric="gloss",
            x_axis=x_axis,
            center=center,
            spread=spread,
            label=f"{value} — generator",
            color=generator_colors[index],
            ax=axes[0],
        )
        plot_convergence(
            group,
            metric="dloss",
            x_axis=x_axis,
            center=center,
            spread=spread,
            label=f"{value} — discriminator",
            color=discriminator_colors[index],
            ax=axes[0],
        )
        plot_convergence(
            group,
            metric="eval",
            x_axis=x_axis,
            center=center,
            spread=spread,
            label=str(value),
            color=evaluation_colors[index],
            ax=axes[1],
        )

    axes[0].set_ylabel("Adversarial loss")
    axes[1].set_ylabel("Evaluation score")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8)
    axes[0].set_xlabel("")
    axes[-1].set_xlabel("Elapsed time (s)" if x_axis == "elapsed" else "Epoch")
    return fig, axes


def _summary_values(
    results: Iterable[RunResult],
    compare_by: str,
    metric_name: str,
) -> tuple[list[Any], list[list[float]]]:
    rows = results_table(results)
    values = sorted(
        {row.get(compare_by) for row in rows if row.get(compare_by) is not None},
        key=_sort_key,
    )
    filtered_values = []
    data = []
    for value in values:
        samples = [
            float(row[metric_name])
            for row in rows
            if row.get(compare_by) == value and _is_finite_number(row.get(metric_name))
        ]
        if samples:
            filtered_values.append(value)
            data.append(samples)
    return filtered_values, data


def plot_final_metric(
    results: Iterable[RunResult],
    *,
    compare_by: str,
    metric_name: str = "best_eval",
    filters: dict[str, Any] | None = None,
    point_color_by: str | None = None,
    ax=None,
):
    import matplotlib.pyplot as plt

    selected = filter_results(results, **(filters or {}))
    summary_rows = results_table(selected)
    values = sorted(
        {
            row.get(compare_by)
            for row in summary_rows
            if row.get(compare_by) is not None
            and _is_finite_number(row.get(metric_name))
        },
        key=_sort_key,
    )
    rows_by_value = [
        [
            row
            for row in summary_rows
            if row.get(compare_by) == value
            and _is_finite_number(row.get(metric_name))
        ]
        for value in values
    ]
    data = [
        [float(row[metric_name]) for row in rows]
        for rows in rows_by_value
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if not values:
        ax.text(0.5, 0.5, "No matching metric values", transform=ax.transAxes, ha="center")
        ax.set_ylabel(metric_name)
        return ax

    positions = np.arange(1, len(values) + 1)
    ax.boxplot(data, positions=positions, widths=0.55, showfliers=False)
    color_map = {}
    if point_color_by is not None:
        color_values = sorted(
            {row.get(point_color_by) for row in summary_rows},
            key=_sort_key,
        )
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_map = {
            value: colors[index % len(colors)]
            for index, value in enumerate(color_values)
        }

    for position, samples, rows in zip(positions, data, rows_by_value):
        if len(samples) <= 1:
            jitter = np.zeros(len(samples))
        else:
            jitter = np.linspace(-0.08, 0.08, len(samples))
        point_colors = "black"
        if point_color_by is not None:
            point_colors = [color_map[row.get(point_color_by)] for row in rows]
        ax.scatter(
            np.full(len(samples), position) + jitter,
            samples,
            color=point_colors,
            s=22,
            alpha=0.75,
            zorder=3,
        )

    if point_color_by is not None:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=color,
                label=f"{point_color_by}={value}",
            )
            for value, color in color_map.items()
        ]
        ax.legend(handles=handles, fontsize=8, title=point_color_by)

    ax.set_xticks(positions, [str(value) for value in values], rotation=30, ha="right")
    ax.set_ylabel(metric_name)
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def plot_runtime(
    results: Iterable[RunResult],
    *,
    compare_by: str,
    metric_name: str = "time_per_epoch",
    filters: dict[str, Any] | None = None,
    log_scale: bool = False,
    ax=None,
):
    ax = plot_final_metric(
        results,
        compare_by=compare_by,
        metric_name=metric_name,
        filters=filters,
        ax=ax,
    )
    ax.set_ylabel(metric_name.replace("_", " "))
    if log_scale:
        ax.set_yscale("log")
    return ax


def _aggregate_samples(samples: list[float], center: str, spread: str) -> tuple[float, float, float]:
    values = np.asarray(samples, dtype=float)
    if center == "median":
        center_value = float(np.nanmedian(values))
    elif center == "mean":
        center_value = float(np.nanmean(values))
    else:
        raise ValueError("center must be 'median' or 'mean'")

    if spread == "iqr":
        low, high = np.nanpercentile(values, [25, 75])
    elif spread == "std":
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        low, high = mean - std, mean + std
    else:
        raise ValueError("spread must be 'iqr' or 'std'")

    return center_value, float(low), float(high)


def plot_metric_by_numeric_field(
    results: Iterable[RunResult],
    *,
    x_field: str,
    metric_name: str = "best_eval",
    line_by: str | None = None,
    filters: dict[str, Any] | None = None,
    center: str = "median",
    spread: str = "iqr",
    show_points: bool = True,
    ax=None,
):
    import matplotlib.pyplot as plt

    rows = results_table(filter_results(results, **(filters or {})))
    rows = [
        row for row in rows
        if row.get(x_field) is not None
        and _is_finite_number(row.get(x_field))
        and _is_finite_number(row.get(metric_name))
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if not rows:
        ax.text(0.5, 0.5, "No matching metric values", transform=ax.transAxes, ha="center")
        ax.set_xlabel(x_field)
        ax.set_ylabel(metric_name.replace("_", " "))
        return ax

    line_values = [None]
    if line_by is not None:
        line_values = sorted({row.get(line_by) for row in rows}, key=_sort_key)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, line_value in enumerate(line_values):
        line_rows = rows
        label = None
        if line_by is not None:
            line_rows = [row for row in rows if row.get(line_by) == line_value]
            label = f"{line_by}={line_value}"

        xs = sorted({float(row[x_field]) for row in line_rows})
        centers = []
        lows = []
        highs = []
        for x in xs:
            samples = [
                float(row[metric_name])
                for row in line_rows
                if float(row[x_field]) == x
            ]
            center_value, low, high = _aggregate_samples(samples, center, spread)
            centers.append(center_value)
            lows.append(low)
            highs.append(high)

            if show_points:
                jitter = np.zeros(len(samples)) if len(samples) <= 1 else np.linspace(-0.015, 0.015, len(samples))
                ax.scatter(
                    np.full(len(samples), x) + jitter,
                    samples,
                    color=colors[index % len(colors)],
                    alpha=0.35,
                    s=22,
                )

        ax.fill_between(
            xs,
            lows,
            highs,
            color=colors[index % len(colors)],
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            xs,
            centers,
            color=colors[index % len(colors)],
            marker="o",
            linewidth=2,
            label=label,
        )

    if line_by is not None:
        ax.legend()
    ax.set_xlabel(x_field)
    ax.set_ylabel(metric_name.replace("_", " "))
    ax.grid(True, alpha=0.25)
    return ax


def plot_metric_by_category_field(
    results: Iterable[RunResult],
    *,
    x_field: str,
    metric_name: str = "best_eval",
    line_by: str | None = None,
    filters: dict[str, Any] | None = None,
    center: str = "median",
    spread: str = "iqr",
    show_points: bool = True,
    ax=None,
):
    import matplotlib.pyplot as plt

    rows = results_table(filter_results(results, **(filters or {})))
    rows = [
        row for row in rows
        if row.get(x_field) is not None
        and _is_finite_number(row.get(metric_name))
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if not rows:
        ax.text(0.5, 0.5, "No matching metric values", transform=ax.transAxes, ha="center")
        ax.set_xlabel(x_field)
        ax.set_ylabel(metric_name.replace("_", " "))
        return ax

    x_values = sorted({row.get(x_field) for row in rows}, key=_sort_key)
    x_positions = np.arange(len(x_values), dtype=float)
    line_values = [None]
    if line_by is not None:
        line_values = sorted({row.get(line_by) for row in rows}, key=_sort_key)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, line_value in enumerate(line_values):
        line_rows = rows
        label = None
        if line_by is not None:
            line_rows = [row for row in rows if row.get(line_by) == line_value]
            label = f"{line_by}={line_value}"

        positions = []
        centers = []
        lower_errors = []
        upper_errors = []
        for x_index, x_value in enumerate(x_values):
            samples = [
                float(row[metric_name])
                for row in line_rows
                if row.get(x_field) == x_value
            ]
            if not samples:
                continue

            center_value, low, high = _aggregate_samples(samples, center, spread)
            # Every line uses the true category coordinate. Horizontal series
            # offsets made timing observations look assigned to adjacent ticks.
            position = x_positions[x_index]
            positions.append(position)
            centers.append(center_value)
            lower_errors.append(max(0.0, center_value - low))
            upper_errors.append(max(0.0, high - center_value))

            if show_points:
                jitter = np.zeros(len(samples)) if len(samples) <= 1 else np.linspace(-0.03, 0.03, len(samples))
                ax.scatter(
                    np.full(len(samples), position) + jitter,
                    samples,
                    color=colors[index % len(colors)],
                    alpha=0.3,
                    s=22,
                )

        if positions:
            ax.errorbar(
                positions,
                centers,
                yerr=[lower_errors, upper_errors],
                color=colors[index % len(colors)],
                marker="o",
                linewidth=2,
                capsize=3,
                label=label,
            )

    if line_by is not None:
        ax.legend()
    ax.set_xticks(x_positions, [str(value) for value in x_values], rotation=30, ha="right")
    ax.set_xlabel(x_field)
    ax.set_ylabel(metric_name.replace("_", " "))
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def plot_seed_sensitivity(
    results: Iterable[RunResult],
    *,
    metric_name: str = "best_eval",
    filters: dict[str, Any] | None = None,
    ax=None,
):
    return plot_final_metric(
        results,
        compare_by="seed",
        metric_name=metric_name,
        filters=filters,
        ax=ax,
    )


def plot_quality_vs_time(
    results: Iterable[RunResult],
    *,
    color_by: str = "gradient_method",
    marker_by: str = "execution_type",
    quality_metric: str = "best_eval",
    time_metric: str = "total_time",
    filters: dict[str, Any] | None = None,
    ax=None,
):
    import matplotlib.pyplot as plt

    rows = results_table(filter_results(results, **(filters or {})))
    rows = [
        row for row in rows
        if _is_finite_number(row.get(quality_metric))
        and _is_finite_number(row.get(time_metric))
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if not rows:
        ax.text(0.5, 0.5, "No matching runs", transform=ax.transAxes, ha="center")
        ax.set_xlabel(time_metric.replace("_", " "))
        ax.set_ylabel(quality_metric.replace("_", " "))
        return ax

    color_values = sorted({row.get(color_by) for row in rows}, key=_sort_key)
    marker_values = sorted({row.get(marker_by) for row in rows}, key=_sort_key)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    color_map = {value: colors[index % len(colors)] for index, value in enumerate(color_values)}
    marker_map = {value: markers[index % len(markers)] for index, value in enumerate(marker_values)}

    for row in rows:
        ax.scatter(
            row[time_metric],
            row[quality_metric],
            color=color_map[row.get(color_by)],
            marker=marker_map[row.get(marker_by)],
            s=42,
            alpha=0.8,
        )

    handles = [
        plt.Line2D([0], [0], color=color_map[value], marker="o", linestyle="", label=f"{color_by}={value}")
        for value in color_values
    ]
    handles.extend(
        plt.Line2D([0], [0], color="black", marker=marker_map[value], linestyle="", label=f"{marker_by}={value}")
        for value in marker_values
    )
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel(time_metric.replace("_", " "))
    ax.set_ylabel(quality_metric.replace("_", " "))
    ax.grid(True, alpha=0.25)
    return ax


def _metric_by_pair(
    rows: Iterable[dict[str, Any]],
    pair_fields: Iterable[str],
    metric_name: str,
) -> dict[tuple[Any, ...], list[float]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric_name)
        if _is_finite_number(value):
            grouped[tuple(row.get(field) for field in pair_fields)].append(float(value))
    return dict(grouped)


def paired_metric_deltas(
    results: Iterable[RunResult],
    *,
    baseline_filters: dict[str, Any],
    treatment_filters: dict[str, Any],
    metric_name: str = "best_eval",
    pair_fields: Iterable[str] = DEFAULT_PAIR_FIELDS,
) -> list[dict[str, Any]]:
    baseline_rows = results_table(filter_results(results, **baseline_filters))
    treatment_rows = results_table(filter_results(results, **treatment_filters))
    baseline = _metric_by_pair(baseline_rows, pair_fields, metric_name)
    treatment = _metric_by_pair(treatment_rows, pair_fields, metric_name)

    deltas = []
    for key in sorted(baseline.keys() & treatment.keys(), key=str):
        baseline_value = float(np.median(baseline[key]))
        treatment_value = float(np.median(treatment[key]))
        ratio = np.nan
        if baseline_value != 0:
            ratio = treatment_value / baseline_value
        deltas.append({
            **dict(zip(pair_fields, key)),
            "baseline": baseline_value,
            "treatment": treatment_value,
            "delta": treatment_value - baseline_value,
            "ratio": ratio,
            "metric": metric_name,
        })
    return deltas


def plot_paired_delta(
    results: Iterable[RunResult],
    *,
    baseline_filters: dict[str, Any],
    treatment_filters: dict[str, Any],
    metric_name: str = "best_eval",
    pair_fields: Iterable[str] = DEFAULT_PAIR_FIELDS,
    x_field: str = "gradient_method",
    value: str = "delta",
    ax=None,
):
    import matplotlib.pyplot as plt

    rows = paired_metric_deltas(
        results,
        baseline_filters=baseline_filters,
        treatment_filters=treatment_filters,
        metric_name=metric_name,
        pair_fields=pair_fields,
    )
    rows = [row for row in rows if _is_finite_number(row.get(value))]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if not rows:
        ax.text(0.5, 0.5, "No matched pairs", transform=ax.transAxes, ha="center")
        ax.set_ylabel(value)
        return ax

    x_values = sorted({row.get(x_field) for row in rows}, key=lambda item: str(item))
    data = [
        [row[value] for row in rows if row.get(x_field) == x]
        for x in x_values
    ]
    positions = np.arange(1, len(x_values) + 1)
    ax.boxplot(data, positions=positions, widths=0.55, showfliers=False)
    for position, samples in zip(positions, data):
        jitter = np.zeros(len(samples)) if len(samples) <= 1 else np.linspace(-0.08, 0.08, len(samples))
        ax.scatter(
            np.full(len(samples), position) + jitter,
            samples,
            color="black",
            s=22,
            alpha=0.75,
            zorder=3,
        )

    ax.axhline(0, color="0.35", linewidth=1, linestyle="--")
    ax.set_xticks(positions, [str(x) for x in x_values], rotation=30, ha="right")
    ax.set_xlabel(x_field)
    ax.set_ylabel(f"{value} {metric_name}".strip())
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def plot_feasibility_matrix(
    results: Iterable[RunResult],
    *,
    row_field: str = "n_qubits",
    column_field: str = "execution_type",
    filters: dict[str, Any] | None = None,
    expected_epochs: int | None = None,
    ax=None,
):
    """Plot completion rate with complete/total annotations for each cell."""

    import matplotlib.pyplot as plt

    selected = filter_results(results, **(filters or {}))
    rows = feasibility_table(
        selected,
        fields=(row_field, column_field),
        expected_epochs=expected_epochs,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    if not rows:
        ax.text(0.5, 0.5, "No matching runs", transform=ax.transAxes, ha="center")
        return ax

    row_values = sorted({row[row_field] for row in rows}, key=_sort_key)
    column_values = sorted({row[column_field] for row in rows}, key=_sort_key)
    matrix = np.full((len(row_values), len(column_values)), np.nan)
    lookup = {(row[row_field], row[column_field]): row for row in rows}
    for row_index, row_value in enumerate(row_values):
        for column_index, column_value in enumerate(column_values):
            record = lookup.get((row_value, column_value))
            if record is None:
                continue
            matrix[row_index, column_index] = record["completion_rate"]
            ax.text(
                column_index,
                row_index,
                f"{record['complete']}/{record['total']}",
                ha="center",
                va="center",
                color="white" if record["completion_rate"] < 0.55 else "black",
                fontsize=9,
            )

    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(column_values)), [str(value) for value in column_values])
    ax.set_yticks(range(len(row_values)), [str(value) for value in row_values])
    ax.set_xlabel(column_field.replace("_", " "))
    ax.set_ylabel(row_field.replace("_", " "))
    ax.figure.colorbar(image, ax=ax, label="completion rate")
    return ax


def plot_status_counts(
    results: Iterable[RunResult],
    *,
    compare_by: str = "n_qubits",
    filters: dict[str, Any] | None = None,
    expected_epochs: int | None = None,
    ax=None,
):
    """Plot stacked feasibility-status counts by one experimental field."""

    import matplotlib.pyplot as plt

    rows = feasibility_table(
        filter_results(results, **(filters or {})),
        fields=(compare_by,),
        expected_epochs=expected_epochs,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    if not rows:
        ax.text(0.5, 0.5, "No matching runs", transform=ax.transAxes, ha="center")
        return ax

    statuses = (
        "complete",
        "partial",
        "out_of_memory",
        "time_limit",
        "failed",
        "load_error",
        "empty_checkpoint",
        "not_started_or_missing",
    )
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for status in statuses:
        values = np.asarray([row[status] for row in rows], dtype=float)
        if not values.any():
            continue
        ax.bar(x, values, bottom=bottom, label=status.replace("_", " "))
        bottom += values

    ax.set_xticks(x, [str(row[compare_by]) for row in rows], rotation=30, ha="right")
    ax.set_xlabel(compare_by.replace("_", " "))
    ax.set_ylabel("run count")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)
    return ax


def print_group_summary(
    results: Iterable[RunResult],
    *,
    fields: Iterable[str] = DEFAULT_GROUP_FIELDS,
    label_fields: Iterable[str] = SUMMARY_LABEL_FIELDS,
    min_runs: int = 1,
) -> None:
    groups = comparable_groups(results, fields=fields)
    for key, runs in sorted(groups.items(), key=lambda item: str(item[0])):
        if len(runs) < min_runs:
            continue
        representative = runs[0]
        summaries = [run_summary(run) for run in runs]
        best_values = [row["best_eval"] for row in summaries if np.isfinite(row["best_eval"])]
        times = [row["time_per_epoch"] for row in summaries if np.isfinite(row["time_per_epoch"])]
        label = _label(representative, label_fields)
        print(label)
        print("  seeds:", [run.seed for run in runs])
        print("  completed:", sum(run.status == "ok" and bool(run.eval) for run in runs), "/", len(runs))
        if best_values:
            print("  best_eval median:", float(np.median(best_values)), "iqr:", tuple(np.percentile(best_values, [25, 75])))
        if times:
            print("  seconds/epoch median:", float(np.median(times)), "iqr:", tuple(np.percentile(times, [25, 75])))
