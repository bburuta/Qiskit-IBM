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
    "simulator_device",
    "noiseless_method",
    "noisy_method",
    "noisy_backend_mapping",
    "real_backend",
    "resilience_level",
    "dynamical_decoupling",
)


SUMMARY_LABEL_FIELDS = (
    "preset",
    "implementation",
    "packing",
    "execution_type",
    "gradient_method",
    "n_qubits",
    "randomness",
    "simulator_device",
    "eval_method",
)


METADATA_PATHS = {
    "run_id": "run.id",
    "label": "run.label",
    "seed": "run.seed",
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
    "learning_rate": "training.learning_rate",
    "max_iterations": "training.max_iterations",
    "gen_iterations": "training.gen_iterations",
    "disc_iterations": "training.disc_iterations",
    "precision": "backend.precision",
    "simulator_device": "backend.simulator.device",
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
    return {
        name: get_nested(config, path)
        for name, path in METADATA_PATHS.items()
    }


def _clean_metric(metric: dict[Any, Any] | None) -> dict[int, float]:
    if not metric:
        return {}
    return {
        int(epoch): float(value)
        for epoch, value in sorted(metric.items(), key=lambda item: int(item[0]))
        if value is not None
    }


def _load_training_state(training_data_file: Path):
    import torch

    return torch.load(training_data_file, weights_only=False, map_location="cpu")


def _load_config(config_file: Path) -> dict[str, Any]:
    try:
        from qgan_v2.config.loader import load_run_config

        return load_run_config(config_file)
    except Exception:
        return {}


def load_result(training_data_file: str | Path) -> RunResult:
    training_data_file = Path(training_data_file)
    state = _load_training_state(training_data_file)
    config = state.config
    metrics = state.metrics
    return RunResult(
        path=training_data_file.parent,
        config=config,
        metadata=_metadata_from_config(config),
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
                result = load_result(training_data_file)
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
            result = RunResult(
                path=run_dir,
                config=config,
                metadata=metadata,
                eval={},
                gloss={},
                dloss={},
                times={},
                status="missing_training_data",
                error="training_data.pth is missing",
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
    return sorted(values, key=lambda value: str(value))


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


def run_summary(result: RunResult, *, last_fraction: float = 0.1) -> dict[str, Any]:
    eval_values = np.asarray(list(result.eval.values()), dtype=float)
    eval_epochs = np.asarray(list(result.eval.keys()), dtype=int)
    time_values = np.asarray(list(result.times.values()), dtype=float)

    finite_eval = np.isfinite(eval_values)

    if finite_eval.any():
        finite_indexes = np.flatnonzero(finite_eval)
        best_index = int(finite_indexes[np.argmin(eval_values[finite_indexes])])
        window = max(1, int(np.ceil(len(eval_values) * last_fraction)))
        best_eval = float(eval_values[best_index])
        best_epoch = int(eval_epochs[best_index])
        final_eval = float(eval_values[-1])
        initial_eval = float(eval_values[0])
        last_window_median_eval = float(np.nanmedian(eval_values[-window:]))
        improvement = initial_eval - best_eval
    else:
        best_eval = np.nan
        best_epoch = None
        final_eval = np.nan
        initial_eval = np.nan
        last_window_median_eval = np.nan
        improvement = np.nan

    if len(time_values):
        total_time = float(np.nansum(time_values))
        time_per_epoch = total_time / len(time_values)
    else:
        total_time = np.nan
        time_per_epoch = np.nan

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
        "best_eval": best_eval,
        "final_eval": final_eval,
        "initial_eval": initial_eval,
        "last_window_median_eval": last_window_median_eval,
        "improvement": improvement,
        "epoch_of_best_eval": best_epoch,
        "total_time": total_time,
        "time_per_epoch": time_per_epoch,
        "time_to_best_eval": time_to_best_eval,
        "error": result.error,
    }


def results_table(results: Iterable[RunResult]) -> list[dict[str, Any]]:
    return [run_summary(result) for result in results]


def representative_runs(
    results: Iterable[RunResult],
    *,
    metric_name: str = "best_eval",
    lower_is_better: bool = True,
) -> dict[str, RunResult]:
    scored = []
    for result in results:
        value = run_summary(result).get(metric_name, np.nan)
        if np.isfinite(value):
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
            ax.plot(x, y, color=color or "0.35", alpha=0.18, linewidth=0.8)

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


def _summary_values(
    results: Iterable[RunResult],
    compare_by: str,
    metric_name: str,
) -> tuple[list[Any], list[list[float]]]:
    rows = results_table(results)
    values = sorted(
        {row.get(compare_by) for row in rows if row.get(compare_by) is not None},
        key=lambda value: str(value),
    )
    filtered_values = []
    data = []
    for value in values:
        samples = [
            float(row[metric_name])
            for row in rows
            if row.get(compare_by) == value and np.isfinite(row.get(metric_name, np.nan))
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
    ax=None,
):
    import matplotlib.pyplot as plt

    selected = filter_results(results, **(filters or {}))
    values, data = _summary_values(selected, compare_by, metric_name)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if not values:
        ax.text(0.5, 0.5, "No matching metric values", transform=ax.transAxes, ha="center")
        ax.set_ylabel(metric_name)
        return ax

    positions = np.arange(1, len(values) + 1)
    ax.boxplot(data, positions=positions, widths=0.55, showfliers=False)
    for position, samples in zip(positions, data):
        if len(samples) <= 1:
            jitter = np.zeros(len(samples))
        else:
            jitter = np.linspace(-0.08, 0.08, len(samples))
        ax.scatter(
            np.full(len(samples), position) + jitter,
            samples,
            color="black",
            s=22,
            alpha=0.75,
            zorder=3,
        )

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
        if np.isfinite(row.get(quality_metric, np.nan))
        and np.isfinite(row.get(time_metric, np.nan))
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if not rows:
        ax.text(0.5, 0.5, "No matching runs", transform=ax.transAxes, ha="center")
        ax.set_xlabel(time_metric.replace("_", " "))
        ax.set_ylabel(quality_metric.replace("_", " "))
        return ax

    color_values = sorted({row.get(color_by) for row in rows}, key=lambda value: str(value))
    marker_values = sorted({row.get(marker_by) for row in rows}, key=lambda value: str(value))
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
