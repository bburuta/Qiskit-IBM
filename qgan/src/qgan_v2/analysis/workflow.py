"""Reusable analysis orchestration for the results notebook.

Low-level selection, aggregation, and plotting primitives live in
``qgan_v2.analysis.results``.  This module composes those primitives into the
specific analysis sections used by ``results_analysis.ipynb`` so the notebook
can concentrate on explaining and invoking the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from qgan_v2.analysis.results import (
    SCIENTIFIC_PAIR_FIELDS,
    RunResult,
    comparison_line_colors,
    evaluation_metric_groups,
    factor_sweep_groups,
    filter_results,
    grouped_performance_table,
    is_completed,
    load_results,
    metadata_from_config,
    plot_learn,
    plot_learn_comparison,
    plot_feasibility_matrix,
    plot_final_metric,
    plot_metric_by_category_field,
    plot_metric_by_numeric_field,
    plot_paired_delta,
    plot_training_dynamics_comparison,
    results_table,
    save_figure,
    select_completed_results,
    select_main_learn_results,
    select_usable_results,
    truncate_results,
    unique_values,
)
from qgan_v2.config.battery import build_config_combinations, load_battery_file
from qgan_v2.storage.paths import get_config_filename


PRESETS = ("base", "ang", "amp")
GRADIENT_METHODS = ("PSR", "REG", "SPSA")
RANDOMNESS_LEVELS = (0, 0.1, 0.25, 0.5, 1)
TIMING_ENVIRONMENTS = ("CPU/1", "CPU/4", "GPU", "Real QPU")
TIMING_ENVIRONMENT_COLORS = {
    "CPU/1": "#1976D2",
    "CPU/4": "#EF6C00",
    "GPU": "#2E7D32",
    "Real QPU": "#7B1FA2",
}
FACET_DISPLAY_NAMES = {
    "execution_type": "Execution Type",
    "gradient_method": "Gradient Method",
    "n_qubits": "Qubit Count",
    "preset": "Encoding Preset",
    "randomness": "Input Randomness",
}
LIMITATION_ORDER = (
    "time-expensive",
    "execution time unavailable",
    "out of memory",
)

LIMITATION_CASES = (
    {
        "experiment": "amplitude q16",
        "limitation": "not transpilable",
        "detail": "The statevector preparation circuit was too large/deep to transpile.",
        "battery_reference": "train_learn_cpu.yaml / train_learn_gpu.yaml",
    },
    {
        "experiment": "noisy q16 CPU timing",
        "limitation": "out of memory",
        "detail": "The density-matrix timing jobs exceeded CPU memory.",
        "battery_reference": "train_times_cpu.yaml",
    },
    {
        "experiment": "real hardware q4",
        "limitation": "execution time unavailable",
        "detail": "Only runs with completed real-hardware timing checkpoints are available.",
        "battery_reference": "train_times_rh.yaml",
    },
    {
        "experiment": "noisy q4/q8 PSR learning dynamics",
        "limitation": "time-expensive",
        "detail": "Parameter-shift noisy learning dynamics did not finish.",
        "battery_reference": "train_learn_cpu.yaml / train_learn_gpu.yaml",
    },
    {
        "experiment": "noisy q16 learning dynamics",
        "limitation": "time-expensive",
        "detail": "Projected execution exceeded the available GPU allocation.",
        "battery_reference": "train_learn_gpu.yaml",
    },
)


def _merged_filters(
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    return {**defaults, **(overrides or {})}


def timing_configuration_key(
    result: RunResult,
    *,
    include_seed: bool = True,
    include_qubits: bool = True,
) -> tuple[Any, ...]:
    """Identify the workload behind a timing run, excluding its environment."""

    fields = [
        field
        for field in SCIENTIFIC_PAIR_FIELDS
        if (include_seed or field != "seed")
        and (include_qubits or field != "n_qubits")
    ]
    metadata = result.metadata
    values = []
    for field in fields:
        value = metadata.get(field)
        if field == "random_circuit" and metadata.get("randomness") == 0:
            value = None
        values.append(value)
    return tuple(values)


def display_rows(
    rows: Iterable[dict[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    column_labels: dict[str, str] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Display records as a dataframe when notebook dependencies are present."""

    all_rows = list(rows)
    visible = all_rows[:limit]
    if columns is not None:
        visible = [
            {column: row.get(column) for column in columns}
            for row in visible
        ]
    try:
        import pandas as pd
        from IPython.display import display

        frame = pd.DataFrame(visible)
        if column_labels:
            frame = frame.rename(columns=column_labels)
        display(frame)
    except ImportError:
        try:
            from IPython.display import Markdown, display
        except ImportError:
            for row in visible:
                print(row)
        else:
            table_columns = list(columns or (visible[0].keys() if visible else ()))

            def markdown_value(value: Any) -> str:
                if value is None or (
                    isinstance(value, (float, np.floating)) and np.isnan(value)
                ):
                    return "—"
                if isinstance(value, (float, np.floating)):
                    return f"{value:.4g}"
                return str(value).replace("|", "\\|")

            if table_columns:
                header = "| " + " | ".join(
                    (column_labels or {}).get(
                        column,
                        column.replace("_", " ").title(),
                    )
                    for column in table_columns
                ) + " |"
                separator = "| " + " | ".join("---" for _ in table_columns) + " |"
                body = [
                    "| " + " | ".join(
                        markdown_value(row.get(column))
                        for column in table_columns
                    ) + " |"
                    for row in visible
                ]
                display(Markdown("\n".join((header, separator, *body))))
    if len(all_rows) > limit:
        print(f"... {len(all_rows) - limit} more rows")
    return visible


def scientific_config_key(config: dict[str, Any]) -> tuple[Any, ...]:
    """Identify a scientific run while ignoring its simulator device."""

    metadata = metadata_from_config(config)
    if metadata["randomness"] == 0:
        # The random circuit is disabled at this level, so its configured value
        # does not distinguish scientific runs.
        metadata["random_circuit"] = None
    return tuple(metadata.get(field) for field in SCIENTIFIC_PAIR_FIELDS)


def battery_configs(
    battery_files: Iterable[str | Path],
    *,
    collapse_simulator_devices: bool = False,
) -> list[dict[str, Any]]:
    """Expand battery files, excluding fake-real validation requests."""

    configs: dict[Any, dict[str, Any]] = {}
    for battery_file in battery_files:
        default_config, variable_groups = load_battery_file(battery_file)
        for variable_values in variable_groups.values():
            for config in build_config_combinations(default_config, variable_values):
                if config["experiment"]["execution_type"] == "fake_real":
                    continue
                key = (
                    scientific_config_key(config)
                    if collapse_simulator_devices
                    else get_config_filename(config).parent.resolve()
                )
                current = configs.get(key)
                device = config["backend"]["simulator"]["device"]
                current_device = (
                    current["backend"]["simulator"]["device"]
                    if current is not None
                    else None
                )
                prefer_cpu = (
                    collapse_simulator_devices
                    and device == "CPU"
                    and current_device != "CPU"
                )
                if current is None or prefer_cpu:
                    configs[key] = config
    return list(configs.values())


def expected_battery_results(
    expected_configs: Iterable[dict[str, Any]],
    observed_results: Iterable[RunResult],
    source: str,
) -> list[RunResult]:
    """Reconcile requested configs with observed checkpoints."""

    observed_by_path = {run.path.resolve(): run for run in observed_results}
    expected_results = []
    for config in expected_configs:
        run_path = get_config_filename(config).parent.resolve()
        run = observed_by_path.get(run_path)
        requested_epochs = int(config["training"]["max_iterations"])
        expected_metadata = metadata_from_config(config)
        if run is None:
            run = RunResult(
                path=run_path,
                config=config,
                metadata={
                    **expected_metadata,
                    "max_iterations": requested_epochs,
                    "analysis_source": source,
                },
                eval={},
                gloss={},
                dloss={},
                times={},
                status="missing_training_data",
                error="planned battery execution was not run",
            )
        else:
            run = replace(
                run,
                config=config,
                metadata={
                    **run.metadata,
                    **expected_metadata,
                    "max_iterations": requested_epochs,
                    "analysis_source": source,
                },
            )
        expected_results.append(run)
    return expected_results


def incomplete_limitation(row: dict[str, Any]) -> str:
    """Map an incomplete requested run to its known resource limitation."""

    source = row.get("analysis_source")
    execution_type = row.get("execution_type")
    if source == "timing" and execution_type == "real":
        return "execution time unavailable"
    if (
        source == "timing"
        and execution_type == "noisy"
        and row.get("n_qubits") == 16
        and row.get("simulator_device") == "CPU"
    ):
        return "out of memory"
    if source == "learn" and execution_type == "noisy" and row.get("n_qubits") == 16:
        if row.get("simulator_device") == "CPU":
            return "out of memory"
        return "time-expensive"
    if source == "learn" and execution_type == "noisy" and row.get("gradient_method") == "PSR":
        return "time-expensive"
    identifier = row.get("run_id") or row.get("path") or "unknown run"
    raise ValueError(f"No limiting-factor classification for {identifier}.")


def known_oom_timing_results(results: Iterable[RunResult]) -> list[RunResult]:
    """Retain failed q16 CPU timing attempts omitted from the runnable battery."""

    selected = []
    for run in results:
        metadata = run.metadata
        if (
            metadata.get("implementation") == "qml_torch"
            and metadata.get("preset") in ("base", "ang")
            and metadata.get("execution_type") == "noisy"
            and metadata.get("n_qubits") == 16
            and metadata.get("simulator_device") == "CPU"
            and not is_completed(run)
        ):
            selected.append(replace(
                run,
                metadata={**metadata, "analysis_source": "timing"},
            ))
    return selected


@dataclass
class ResultsData:
    """The result subsets shared by all analysis figures."""

    qgan_dir: Path
    raw_learn: list[RunResult]
    raw_timing: list[RunResult]
    main_learn: list[RunResult]
    timing: list[RunResult]
    hardware: list[RunResult]
    validation: list[RunResult]

    @classmethod
    def load(cls, qgan_dir: str | Path) -> "ResultsData":
        qgan_dir = Path(qgan_dir)
        raw_learn = load_results(qgan_dir / "data" / "train")
        raw_timing = load_results(qgan_dir / "data" / "train" / "times")
        return cls(
            qgan_dir=qgan_dir,
            raw_learn=raw_learn,
            raw_timing=raw_timing,
            main_learn=select_main_learn_results(
                raw_learn,
                expected_epochs=1000,
            ),
            timing=select_completed_results(raw_timing, expected_epochs=5),
            # The results hardware case study is restricted to qml_torch.
            hardware=filter_results(
                select_usable_results(raw_learn, execution_types=("real",)),
                implementation="qml_torch",
            ),
            validation=select_usable_results(
                raw_learn,
                execution_types=("fake_real",),
            ),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "raw_learn/case-study_configs": len(self.raw_learn),
            "completed_deduplicated_simulator_runs": len(self.main_learn),
            "completed_five_epoch_timing_runs": len(self.timing),
            "usable_real_hardware_runs": len(self.hardware),
            "usable_implementation_validation_runs": len(self.validation),
            "evaluation_families": {
                name: len(runs)
                for name, runs in evaluation_metric_groups(self.main_learn).items()
            },
        }


class ResultsAnalysis:
    """Build and optionally export figures from experiment results."""

    def __init__(
        self,
        results: ResultsData,
        *,
        export_figures: bool = False,
        figure_dir: str | Path | None = None,
        gpu_model_label: str = "RTX6000",
    ) -> None:
        self.results = results
        self.export_figures = export_figures
        self.figure_dir = Path(
            figure_dir or results.qgan_dir / "figures" / "results_analysis"
        )
        self.gpu_model_label = gpu_model_label
        self._preset_output_runs: dict[str, tuple[RunResult, dict[str, Any]]] = {}

    @classmethod
    def load(
        cls,
        qgan_dir: str | Path,
        **kwargs: Any,
    ) -> "ResultsAnalysis":
        return cls(ResultsData.load(qgan_dir), **kwargs)

    def _finish(
        self,
        fig,
        stem: str,
        *,
        layout_rect: tuple[float, float, float, float] | None = None,
        tight_layout: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt

        if tight_layout:
            fig.tight_layout(rect=layout_rect)
        if self.export_figures:
            for saved_path in save_figure(fig, self.figure_dir, stem, formats=("png",)):
                print("saved:", saved_path)
        plt.show()
        plt.close(fig)

    def _evaluation_dynamics_figure(
        self,
        runs: Iterable[RunResult],
        compare_by: str,
        title: str,
        stem: str,
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_learn_comparison(
            runs,
            compare_by=compare_by,
            metric="eval",
            center="median",
            spread="iqr",
            ax=ax,
        )
        ax.set_ylabel("Evaluation Score")
        fig.suptitle(title)
        self._finish(fig, stem)
        return fig, np.asarray([ax])

    def _best_results_figure(
        self,
        runs: Iterable[RunResult],
        compare_by: str,
        title: str,
        stem: str,
    ):
        import matplotlib.pyplot as plt

        runs = list(runs)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        plot_final_metric(
            runs,
            compare_by=compare_by,
            metric_name="best_eval",
            point_color_by="seed",
            ax=axes[0],
        )
        axes[0].set_title("Best Evaluation Score")
        plot_final_metric(
            runs,
            compare_by=compare_by,
            metric_name="average_best_so_far_eval",
            point_color_by="seed",
            ax=axes[1],
        )
        axes[1].set_title("Average Best Evaluation So Far")
        fig.suptitle(title)
        self._finish(fig, stem)
        return fig, axes

    def _faceted_best_results_figure(
        self,
        groups: Iterable[tuple[Any, Iterable[RunResult]]],
        *,
        compare_by: str,
        stem: str,
        title: str | None = None,
        panel_width: float = 5,
    ):
        import matplotlib.pyplot as plt

        groups = [(label, list(runs)) for label, runs in groups]
        fig, axes = plt.subplots(
            2,
            len(groups),
            figsize=(panel_width * len(groups), 8),
            squeeze=False,
        )
        for column, (label, runs) in enumerate(groups):
            display_label = str(label)
            if isinstance(label, str) and not label.isupper():
                display_label = label.replace("_", " ").title()
            for row, metric_name, metric_title in (
                (0, "best_eval", "Best Evaluation Score"),
                (1, "average_best_so_far_eval", "Average Best Evaluation So Far"),
            ):
                plot_final_metric(
                    runs,
                    compare_by=compare_by,
                    metric_name=metric_name,
                    point_color_by="seed",
                    ax=axes[row, column],
                )
                axes[row, column].set_title(f"{display_label}: {metric_title}")
        if title is not None:
            fig.suptitle(title)
        self._finish(fig, stem)
        return fig, axes

    def timing(
        self,
        *,
        baseline_filters: dict[str, Any] | None = None,
        factor_levels: dict[str, Any] | None = None,
    ) -> list[RunResult]:
        """Backward-compatible entry point for controlled timing plots."""

        return self.controlled_timing(
            baseline_filters=baseline_filters,
            factor_levels=factor_levels,
        )

    def _timing_environment_results(
        self,
        *,
        include_real: bool = True,
    ) -> list[RunResult]:
        """Return timing runs with explicit, analysis-facing environment labels."""

        labelled = []
        for run in self.results.timing:
            execution_type = run.metadata.get("execution_type")
            label = run.metadata.get("label")
            simulator_device = run.metadata.get("simulator_device")
            run_device = run.metadata.get("run_device")
            if execution_type == "real":
                if not include_real:
                    continue
                environment = "Real QPU"
            elif simulator_device == "GPU" or run_device == "GPU":
                environment = "GPU"
            elif label == "cpu4" or run.metadata.get("cpu_threads") == 4:
                environment = "CPU/4"
            elif label is None and (
                simulator_device == "CPU" or run_device == "CPU"
            ):
                environment = "CPU/1"
            else:
                continue
            labelled.append(replace(
                run,
                metadata={**run.metadata, "timing_environment": environment},
            ))
        return labelled

    def controlled_timing(
        self,
        *,
        baseline_filters: dict[str, Any] | None = None,
        factor_levels: dict[str, Any] | None = None,
    ) -> list[RunResult]:
        """Plot controlled timing factors and the matched real-QPU case study."""

        import matplotlib.pyplot as plt

        baseline_filters = _merged_filters(
            {
                "preset": "ang",
                "implementation": "qml_torch",
                "execution_type": "noiseless",
                "gradient_method": "SPSA",
                "n_qubits": 4,
                "randomness": 0,
            },
            baseline_filters,
        )
        factor_levels = {
            "preset": PRESETS,
            "execution_type": ("noiseless", "noisy"),
            "gradient_method": GRADIENT_METHODS,
            "randomness": (0, 1),
            **(factor_levels or {}),
        }

        all_timing = self._timing_environment_results()
        primary_timing = filter_results(
            all_timing,
            execution_type=("noiseless", "noisy"),
            timing_environment=("CPU/1", "CPU/4", "GPU"),
        )

        print("timing environments:", unique_values(primary_timing, "timing_environment"))
        print("GPU model:", self.gpu_model_label)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        panels = (
            ("preset", "Runtime by Encoding Preset"),
            ("execution_type", "Runtime by Execution Type"),
            ("gradient_method", "Runtime by Gradient Method"),
            ("randomness", "Runtime by Input Randomness Level"),
        )
        for ax, (x_field, title) in zip(axes.flat, panels):
            filters = {
                field: value
                for field, value in baseline_filters.items()
                if field != x_field
            }
            if x_field in factor_levels:
                filters[x_field] = factor_levels[x_field]
            plot_metric_by_category_field(
                primary_timing,
                x_field=x_field,
                metric_name="median_time_per_epoch",
                line_by="timing_environment",
                filters=filters,
                ax=ax,
            )
            ax.set_yscale("log")
            ax.set_title(title)

        matched_hardware = filter_results(
            all_timing,
            preset="base",
            implementation="qml_torch",
            gradient_method=("PSR", "SPSA"),
            n_qubits=4,
            randomness=0,
        )
        matched_hardware = [
            replace(
                run,
                metadata={
                    **run.metadata,
                    "matched_execution": {
                        "noiseless": "Noiseless simulator",
                        "noisy": "Noisy simulator",
                        "real": "Real QPU",
                    }.get(run.metadata.get("execution_type")),
                },
            )
            for run in matched_hardware
            if (
                run.metadata.get("timing_environment") == "CPU/1"
                or run.metadata.get("execution_type") == "real"
            )
        ]
        fig.suptitle("Controlled Per-Epoch Runtime Across Experimental Factors")
        self._finish(fig, "01a_controlled_timing_factors")

        qpu_fig, qpu_ax = plt.subplots(figsize=(7, 4.5))
        plot_metric_by_category_field(
            matched_hardware,
            x_field="matched_execution",
            metric_name="median_time_per_epoch",
            line_by="gradient_method",
            ax=qpu_ax,
        )
        qpu_ax.set_yscale("log")
        qpu_ax.set_title(
            "Matched Simulator and Real QPU Timing\n"
            "(base · q4 · rand0 · qml_torch; simulator = CPU/1)"
        )
        self._finish(qpu_fig, "01b_matched_real_qpu_timing")
        return primary_timing

    def matched_timing_comparison(
        self,
        *,
        preset: str = "base",
        randomness: float = 0,
        filters: dict[str, Any] | None = None,
        execution_types: Sequence[str] = ("noiseless", "noisy"),
        environments: Sequence[str] = TIMING_ENVIRONMENTS,
    ) -> list[RunResult]:
        """Compare environment runtimes by qubit/gradient run category.

        One encoding preset and randomness level are selected at a time. The
        x-axis groups runs by qubit count and gradient method, while the y-axis
        reports median time per epoch. Real-QPU measurements are included in
        the noisy panel when available.
        """

        import matplotlib.pyplot as plt

        selected_execution_types = tuple(execution_types)
        source_execution_types = list(selected_execution_types)
        if "noisy" in selected_execution_types and "Real QPU" in environments:
            source_execution_types.append("real")
        selected = filter_results(
            self._timing_environment_results(),
            **_merged_filters(
                {
                    "preset": preset,
                    "implementation": "qml_torch",
                    "execution_type": tuple(source_execution_types),
                    "randomness": randomness,
                    "timing_environment": tuple(environments),
                },
                filters,
            ),
        )

        runs = selected

        gradient_order = ("SPSA", "PSR", "REG")

        def category_key(run: RunResult) -> tuple[Any, Any]:
            return (
                run.metadata.get("n_qubits"),
                run.metadata.get("gradient_method"),
            )

        def category_sort_key(category: tuple[Any, Any]) -> tuple[Any, Any]:
            n_qubits, gradient_method = category
            return (
                n_qubits if isinstance(n_qubits, (int, float)) else float("inf"),
                gradient_order.index(gradient_method)
                if gradient_method in gradient_order else len(gradient_order),
            )

        panel_runs = []
        for execution_type in selected_execution_types:
            panel_runs.append([
                run
                for run in runs
                if run.metadata.get("execution_type") == execution_type
                or (
                    execution_type == "noisy"
                    and run.metadata.get("execution_type") == "real"
                )
            ])

        panel_categories = [
            sorted(
                {category_key(run) for run in current_runs},
                key=category_sort_key,
            )
            for current_runs in panel_runs
        ]

        fig, axes = plt.subplots(
            1,
            len(selected_execution_types),
            figsize=(8 * len(selected_execution_types), 6),
            squeeze=False,
            sharey=True,
        )
        all_values = [
            float(np.nanmedian(list(run.times.values())))
            for run in runs
            if run.times
        ]
        positive_values = [value for value in all_values if value > 0]
        if positive_values:
            y_limits = (min(positive_values) * 0.65, max(positive_values) * 1.7)
        else:
            y_limits = (0.1, 10)

        active_environments = [
            environment
            for environment in environments
            if any(
                run.metadata.get("timing_environment") == environment
                for run in runs
            )
        ]
        annotation_offsets = {
            "CPU/4": (-5, 7, "right", "bottom"),
            "GPU": (5, -7, "left", "top"),
            "Real QPU": (5, 7, "left", "bottom"),
        }

        for ax, execution_type, current_runs, categories in zip(
            axes[0],
            selected_execution_types,
            panel_runs,
            panel_categories,
        ):
            for category_index, category in enumerate(categories):
                category_runs = [
                    run for run in current_runs if category_key(run) == category
                ]
                values_by_environment = {}
                for environment in active_environments:
                    values = [
                        float(np.nanmedian(list(run.times.values())))
                        for run in category_runs
                        if run.metadata.get("timing_environment") == environment
                        and run.times
                    ]
                    positive = [value for value in values if value > 0]
                    if positive:
                        values_by_environment[environment] = float(np.nanmedian(positive))

                if len(values_by_environment) >= 2:
                    ax.plot(
                        (category_index, category_index),
                        (
                            min(values_by_environment.values()),
                            max(values_by_environment.values()),
                        ),
                        color="0.78",
                        linewidth=1.2,
                        zorder=1,
                    )
                cpu1 = values_by_environment.get("CPU/1")
                for environment in active_environments:
                    value = values_by_environment.get(environment)
                    if value is None:
                        continue
                    ax.scatter(
                        category_index,
                        value,
                        color=TIMING_ENVIRONMENT_COLORS[environment],
                        marker="D" if environment == "Real QPU" else "o",
                        s=52 if environment == "Real QPU" else 44,
                        zorder=3,
                    )
                    if cpu1 and environment in annotation_offsets:
                        x_offset, y_offset, horizontal, vertical = (
                            annotation_offsets[environment]
                        )
                        ax.annotate(
                            f"{cpu1 / value:.1f}×",
                            (category_index, value),
                            xytext=(x_offset, y_offset),
                            textcoords="offset points",
                            ha=horizontal,
                            va=vertical,
                            fontsize=7,
                            color=TIMING_ENVIRONMENT_COLORS[environment],
                        )

            labels = [f"q{n_qubits} · {gradient}" for n_qubits, gradient in categories]
            ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
            ax.set_yscale("log")
            ax.set_ylim(*y_limits)
            ax.set_xlabel("Run (qubits · gradient method)")
            ax.set_ylabel("Median time per epoch (s, log scale)")
            ax.set_title(execution_type.replace("_", " ").title())
            ax.grid(True, axis="y", alpha=0.25)
            if not current_runs:
                ax.text(
                    0.5,
                    0.5,
                    "No matched environments",
                    transform=ax.transAxes,
                    ha="center",
                )

        handles = [
            plt.Line2D(
                [0],
                [0],
                linestyle="",
                color=TIMING_ENVIRONMENT_COLORS[environment],
                label=environment,
                marker="D" if environment == "Real QPU" else "o",
            )
            for environment in active_environments
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=len(handles),
            )
        fig.suptitle(
            f"Matched Environment Runtime — {preset.upper()} Preset",
            y=0.995,
        )
        stem_preset = "".join(
            character if character.isalnum() else "_" for character in preset.lower()
        ).strip("_")
        self._finish(
            fig,
            f"01a_{stem_preset}_matched_environment_runtime",
            layout_rect=(0, 0, 1, 0.88),
        )
        return runs

    def timing_distributions(
        self,
        *,
        compare_by: str | None = None,
        preset: str | Sequence[str] | None = None,
        execution_type: str | Sequence[str] | None = None,
        n_qubits: int | Sequence[int] | None = None,
        gradient_method: str | Sequence[str] | None = None,
        randomness: float | Sequence[float] | None = None,
        implementation: str = "qml_torch",
        environments: Sequence[str] = TIMING_ENVIRONMENTS,
        layout: tuple[int, int] | None = None,
    ) -> list[RunResult]:
        """Compare device timing distributions for selected configurations.

        Without ``compare_by``, omitted scientific arguments retain the
        controlled angle/noiseless/q4/SPSA/rand0 defaults. Each argument can
        be a sequence, creating Cartesian configuration panels as before.

        With ``compare_by``, one panel is created per execution environment
        and the compared field's levels are placed on its x-axis. All other
        omitted scientific fields are pooled, while explicit values remain
        filters. For example,
        ``compare_by="randomness"`` uses every eligible rand0 and rand1 run,
        while ``compare_by="randomness", preset="ang"`` restricts that pooled
        comparison to the angle preset. ``layout=(rows, columns)`` controls the
        subplot grid; by default all panels are placed in one horizontal row.
        """

        import matplotlib.pyplot as plt

        unknown_environments = [
            environment
            for environment in environments
            if environment not in TIMING_ENVIRONMENT_COLORS
        ]
        if unknown_environments:
            raise ValueError(
                "Unknown timing environment(s): "
                + ", ".join(map(str, unknown_environments))
            )
        if not environments:
            raise ValueError("environments must contain at least one device")

        def values_tuple(value: Any) -> tuple[Any, ...]:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                values = tuple(value)
            else:
                values = (value,)
            if not values:
                raise ValueError("timing comparison arguments cannot be empty")
            return values

        comparison_fields = (
            "preset",
            "execution_type",
            "n_qubits",
            "gradient_method",
            "randomness",
        )
        requested_values = {
            "preset": preset,
            "execution_type": execution_type,
            "n_qubits": n_qubits,
            "gradient_method": gradient_method,
            "randomness": randomness,
        }
        controlled_defaults = {
            "preset": "ang",
            "execution_type": "noiseless",
            "n_qubits": 4,
            "gradient_method": "SPSA",
            "randomness": 0,
        }
        comparison_defaults = {
            "preset": PRESETS,
            "execution_type": (
                ("noiseless", "noisy", "real")
                if "Real QPU" in environments
                else ("noiseless", "noisy")
            ),
            "gradient_method": GRADIENT_METHODS,
            "randomness": (0, 1),
        }
        if compare_by is not None and compare_by not in comparison_fields:
            allowed = ", ".join(comparison_fields)
            raise ValueError(f"compare_by must be one of: {allowed}")

        candidate_runs = filter_results(
            self._timing_environment_results(include_real="Real QPU" in environments),
            implementation=implementation,
            timing_environment=tuple(environments),
        )
        if compare_by is None:
            selection_values = {
                field: values_tuple(
                    controlled_defaults[field]
                    if requested_values[field] is None
                    else requested_values[field]
                )
                for field in comparison_fields
            }
            cases = [
                dict(zip(comparison_fields, values))
                for values in product(
                    *(selection_values[field] for field in comparison_fields)
                )
            ]
            selected_runs = filter_results(candidate_runs, **selection_values)
        else:
            explicit_filters = {
                field: values_tuple(value)
                for field, value in requested_values.items()
                if field != compare_by and value is not None
            }
            candidate_runs = filter_results(candidate_runs, **explicit_filters)
            requested_comparison = requested_values[compare_by]
            if requested_comparison is not None:
                compared_values = values_tuple(requested_comparison)
            elif compare_by == "n_qubits":
                compared_values = tuple(unique_values(candidate_runs, compare_by))
            else:
                compared_values = comparison_defaults[compare_by]
            if not compared_values:
                return []
            cases = [{compare_by: value} for value in compared_values]
            selected_runs = filter_results(
                candidate_runs,
                **{compare_by: compared_values},
            )

        def comparison_value_label(field: str, value: Any) -> str:
            if field == "randomness" and isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                return f"rand{value:g}"
            if field == "n_qubits":
                return f"q{value}"
            return str(value).replace("_", " ")

        def has_epoch_times(runs: Sequence[RunResult]) -> bool:
            return any(
                np.isfinite(value) and value > 0
                for run in runs
                for value in run.times.values()
            )

        if compare_by is None:
            panel_values = [
                case_filters
                for case_filters in cases
                if has_epoch_times(
                    filter_results(selected_runs, **case_filters)
                )
            ]
        else:
            panel_values = [
                environment
                for environment in environments
                if has_epoch_times(
                    filter_results(
                        selected_runs,
                        timing_environment=environment,
                    )
                )
            ]
        if not panel_values:
            return selected_runs

        panel_count = len(panel_values)
        if layout is None:
            row_count, column_count = 1, panel_count
        else:
            if (
                not isinstance(layout, (tuple, list))
                or len(layout) != 2
                or not all(isinstance(value, int) for value in layout)
                or any(value <= 0 for value in layout)
            ):
                raise ValueError("layout must contain two positive integers")
            row_count, column_count = layout
            if row_count * column_count < panel_count:
                raise ValueError(
                    "layout does not have enough cells for "
                    f"{panel_count} timing panels"
                )

        def panel_attribute_count(panel_value: Any) -> int:
            if compare_by is None:
                panel_runs = filter_results(selected_runs, **panel_value)
                return sum(
                    has_epoch_times(
                        filter_results(
                            panel_runs,
                            timing_environment=environment,
                        )
                    )
                    for environment in environments
                )
            panel_runs = filter_results(
                selected_runs,
                timing_environment=panel_value,
            )
            return sum(
                has_epoch_times(
                    filter_results(panel_runs, **{compare_by: value})
                )
                for value in compared_values
            )

        attribute_counts = [
            panel_attribute_count(panel_value)
            for panel_value in panel_values
        ]
        column_widths = [
            max(
                attribute_counts[index]
                for index in range(column, panel_count, column_count)
            )
            for column in range(column_count)
            if column < panel_count
        ]
        column_widths.extend([1] * (column_count - len(column_widths)))
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(max(3.0, 1.5 * sum(column_widths)), 4.1 * row_count),
            squeeze=False,
            sharey=compare_by is not None,
            gridspec_kw={"width_ratios": column_widths},
        )
        has_timing_samples = False
        flat_axes = axes.ravel()
        for panel_index, (ax, panel_value) in enumerate(
            zip(flat_axes, panel_values)
        ):
            if compare_by is None:
                case_filters = panel_value
                runs = filter_results(selected_runs, **case_filters)
                available_environments = []
                for environment in environments:
                    environment_runs = filter_results(
                        runs,
                        timing_environment=environment,
                    )
                    if has_epoch_times(environment_runs):
                        available_environments.append(
                            (environment, environment_runs)
                        )
                group_specs = [
                    (
                        position,
                        environment,
                        TIMING_ENVIRONMENT_COLORS[environment],
                        environment_runs,
                    )
                    for position, (environment, environment_runs) in enumerate(
                        available_environments,
                        start=1,
                    )
                ]
                randomness_value = case_filters["randomness"]
                randomness_label = comparison_value_label(
                    "randomness",
                    randomness_value,
                )
                ax.set_title(
                    f"{case_filters['preset']} · "
                    f"{str(case_filters['execution_type']).replace('_', ' ')} · "
                    f"q{case_filters['n_qubits']}\n"
                    f"{case_filters['gradient_method']} · {randomness_label}",
                    fontsize=10,
                )
                ax.set_xlabel("Execution environment")
            else:
                environment = panel_value
                runs = filter_results(
                    selected_runs,
                    timing_environment=environment,
                )
                available_groups = []
                for compared_value in compared_values:
                    group_runs = filter_results(
                        runs,
                        **{compare_by: compared_value},
                    )
                    if has_epoch_times(group_runs):
                        available_groups.append((compared_value, group_runs))
                group_specs = [
                    (
                        position,
                        comparison_value_label(compare_by, compared_value),
                        TIMING_ENVIRONMENT_COLORS[environment],
                        group_runs,
                    )
                    for position, (compared_value, group_runs) in enumerate(
                        available_groups,
                        start=1,
                    )
                ]
                ax.set_title(environment, fontsize=10)
                ax.set_xlabel(FACET_DISPLAY_NAMES[compare_by])
                ax.set_xlim(0.5, len(available_groups) + 0.5)

            # Keep a fixed physical width per visible category after empty
            # categories and panels have been removed. Without a box-aspect
            # constraint, the remaining axes can stretch to consume the
            # vacated horizontal space in notebook and responsive renderers.
            ax.set_box_aspect(1.6 / len(group_specs))

            positions = []
            samples_by_position = []
            colors = []
            labels = []
            run_values_by_position = []
            for position, label, color, group_runs in group_specs:
                valid_runs = [
                    [
                        float(value)
                        for _, value in sorted(run.times.items())
                        if np.isfinite(value) and value > 0
                    ]
                    for run in group_runs
                ]
                valid_runs = [values for values in valid_runs if values]
                samples = [
                    value
                    for epoch_values in valid_runs
                    for value in epoch_values
                ]
                if not samples:
                    continue
                positions.append(position)
                samples_by_position.append(samples)
                colors.append(color)
                labels.append(label)
                run_values_by_position.append((position, color, valid_runs))

            if samples_by_position:
                boxes = ax.boxplot(
                    samples_by_position,
                    positions=positions,
                    widths=0.44,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    medianprops={"color": "black", "linewidth": 1.8},
                )
                for patch, color in zip(boxes["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.22)

                for position, color, valid_runs in run_values_by_position:
                    run_count = len(valid_runs)
                    run_offsets = (
                        np.linspace(-0.17, 0.17, run_count)
                        if run_count > 1
                        else np.zeros(run_count)
                    )
                    epoch_half_width = (
                        0.04 if run_count <= 1 else min(0.025, 0.12 / run_count)
                    )
                    for epoch_values, run_offset in zip(valid_runs, run_offsets):
                        run_position = position + run_offset
                        epoch_offsets = (
                            np.linspace(
                                -epoch_half_width,
                                epoch_half_width,
                                len(epoch_values),
                            )
                            if len(epoch_values) > 1
                            else np.zeros(1)
                        )
                        ax.scatter(
                            run_position + epoch_offsets,
                            epoch_values,
                            marker="o",
                            s=11,
                            color=color,
                            edgecolor="none",
                            linewidths=0,
                            alpha=0.18,
                            zorder=2,
                        )
                        ax.scatter(
                            run_position,
                            float(np.nanmedian(epoch_values)),
                            marker="o",
                            s=32,
                            color=color,
                            edgecolor="none",
                            linewidths=0,
                            alpha=0.95,
                            zorder=4,
                        )
                        has_timing_samples = True

            if compare_by is None:
                visible_groups = [
                    (position, label)
                    for position, label, _, _ in group_specs
                    if position in positions
                ]
            else:
                visible_groups = [
                    (position, label)
                    for position, label, _, _ in group_specs
                ]
            ax.set_xticks(
                [position for position, _ in visible_groups],
                [label for _, label in visible_groups],
            )
            ax.set_yscale("log")
            if compare_by is None or panel_index == 0:
                ax.set_ylabel("Recorded epoch time (s, log scale)", fontsize=9)
            ax.xaxis.label.set_size(9)
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="both", labelsize=8)
            ax.tick_params(axis="x", rotation=25)
            if not samples_by_position:
                ax.text(0.5, 0.5, "No timing data", transform=ax.transAxes, ha="center")
        for ax in flat_axes[panel_count:]:
            fig.delaxes(ax)
        if has_timing_samples:
            epoch_handle = plt.Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                markersize=4,
                color="0.35",
                alpha=0.25,
            )
            median_handle = plt.Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                markersize=5.5,
                markeredgecolor="none",
                color="0.2",
            )
            fig.legend(
                (epoch_handle, median_handle),
                ("epoch time (shadow)", "run median"),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.90),
                ncol=2,
                fontsize=9,
            )
        fig.suptitle("Device Epoch-Time Distributions", y=0.995, fontsize=12)
        self._finish(
            fig,
            "01b_device_timing_distributions",
            layout_rect=(0, 0, 1, 0.84),
        )
        return selected_runs

    def runtime_scaling(
        self,
        *,
        filters: dict[str, Any] | None = None,
        execution_types: Sequence[str] = ("noiseless", "noisy"),
        qubits: Sequence[int] = (4, 8, 16),
    ) -> list[RunResult]:
        """Plot all workload trajectories and the battery-wide scaling trend."""

        import matplotlib.pyplot as plt

        runs = filter_results(
            self._timing_environment_results(include_real=False),
            **_merged_filters(
                {
                    "implementation": "qml_torch",
                    "execution_type": tuple(execution_types),
                    "n_qubits": tuple(qubits),
                    "timing_environment": ("CPU/1", "CPU/4", "GPU"),
                },
                filters,
            ),
        )
        environments = ("CPU/1", "CPU/4", "GPU")
        fig, axes = plt.subplots(
            len(execution_types),
            len(environments),
            figsize=(5.1 * len(environments), 4.3 * len(execution_types)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        preset_colors = dict(zip(PRESETS, comparison_line_colors(PRESETS)))
        for row, execution_type in enumerate(execution_types):
            for column, environment in enumerate(environments):
                ax = axes[row, column]
                panel_runs = filter_results(
                    runs,
                    execution_type=execution_type,
                    timing_environment=environment,
                )
                grouped: dict[tuple[Any, ...], list[RunResult]] = {}
                for run in panel_runs:
                    grouped.setdefault(
                        timing_configuration_key(run, include_qubits=False),
                        [],
                    ).append(run)
                for group in grouped.values():
                    points = sorted(
                        (
                            int(run.metadata["n_qubits"]),
                            float(np.nanmedian(list(run.times.values()))),
                        )
                        for run in group
                        if run.times
                    )
                    if not points:
                        continue
                    ax.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        color=preset_colors.get(group[0].metadata.get("preset"), "0.5"),
                        linewidth=0.8,
                        alpha=0.28,
                        marker="o",
                        markersize=2.5,
                    )

                summary_x = []
                summary_center = []
                summary_low = []
                summary_high = []
                summary_count = []
                for n_qubits in qubits:
                    samples = [
                        float(np.nanmedian(list(run.times.values())))
                        for run in panel_runs
                        if run.metadata.get("n_qubits") == n_qubits and run.times
                    ]
                    if not samples:
                        continue
                    low, center, high = np.nanpercentile(samples, (25, 50, 75))
                    summary_x.append(n_qubits)
                    summary_center.append(center)
                    summary_low.append(low)
                    summary_high.append(high)
                    summary_count.append(len(samples))
                if summary_x:
                    ax.fill_between(
                        summary_x,
                        summary_low,
                        summary_high,
                        color="black",
                        alpha=0.1,
                        linewidth=0,
                    )
                    ax.plot(
                        summary_x,
                        summary_center,
                        color="black",
                        linewidth=2.4,
                        marker="o",
                        label="battery median",
                    )
                    for x, y, count in zip(summary_x, summary_center, summary_count):
                        ax.annotate(
                            f"n={count}",
                            (x, y),
                            xytext=(0, 7),
                            textcoords="offset points",
                            ha="center",
                            fontsize=7,
                        )

                ax.set_xticks(qubits, [f"q{value}" for value in qubits])
                ax.set_yscale("log")
                ax.grid(True, alpha=0.25)
                if row == 0:
                    ax.set_title(environment)
                if column == 0:
                    ax.set_ylabel(
                        f"{execution_type.replace('_', ' ').title()}\n"
                        "Median time per epoch (s)"
                    )
                if row == len(execution_types) - 1:
                    ax.set_xlabel("Number of qubits")
                if not panel_runs:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")

        handles = [
            plt.Line2D([0], [0], color=preset_colors[preset], label=preset)
            for preset in PRESETS
        ]
        handles.append(plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.4,
            label="battery median/IQR",
        ))
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=len(handles),
        )
        fig.suptitle(
            "Runtime Scaling Across the Timing Battery\n"
            "thin lines are matched preset/gradient/randomness workloads",
            y=0.995,
        )
        self._finish(
            fig,
            "01d_runtime_scaling",
            layout_rect=(0, 0, 1, 0.91),
        )
        return runs

    def _training_cost_rows(
        self,
        *,
        filters: dict[str, Any] | None = None,
        randomness_levels: Sequence[float] = (0, 1),
    ) -> list[dict[str, Any]]:
        """Build simulator and real-QPU projected-cost records."""

        simulator_runs = filter_results(
            self._timing_environment_results(include_real=False),
            **_merged_filters(
                {
                    "preset": "ang",
                    "implementation": "qml_torch",
                    "execution_type": "noiseless",
                    "gradient_method": "SPSA",
                    "n_qubits": 4,
                    "randomness": tuple(randomness_levels),
                    "timing_environment": ("CPU/1", "CPU/4", "GPU"),
                },
                filters,
            ),
        )
        simulator_rows = results_table(simulator_runs)
        cpu1_by_randomness = {
            row["randomness"]: row["median_time_per_epoch"]
            for row in simulator_rows
            if row.get("timing_environment") == "CPU/1"
        }
        environment_order = {"CPU/1": 0, "CPU/4": 1, "GPU": 2}
        rows = []
        for row in sorted(
            simulator_rows,
            key=lambda item: (
                float(item["randomness"]),
                environment_order[item["timing_environment"]],
            ),
        ):
            median_time = float(row["median_time_per_epoch"])
            cpu1_time = cpu1_by_randomness.get(row["randomness"])
            rows.append({
                "case": f"rand{row['randomness']:g}",
                "execution": "Noiseless simulator",
                "gradient": row["gradient_method"],
                "environment": row["timing_environment"],
                "target_environment": (
                    f"Noiseless simulator · {row['timing_environment']}"
                ),
                "median_s_per_epoch": median_time,
                "speed_up_vs_cpu1": (
                    float(cpu1_time / median_time)
                    if cpu1_time is not None
                    else np.nan
                ),
                "estimated_1000_epochs_h": median_time * 1000 / 3600,
                "measured_epochs": row["measured_epochs"],
            })

        real_rows = results_table(filter_results(
            self._timing_environment_results(),
            preset="base",
            implementation="qml_torch",
            execution_type="real",
            gradient_method=("PSR", "SPSA"),
            n_qubits=4,
            randomness=0,
        ))
        for row in sorted(
            real_rows,
            key=lambda item: GRADIENT_METHODS.index(item["gradient_method"]),
        ):
            median_time = float(row["median_time_per_epoch"])
            rows.append({
                "case": "base q4 rand0",
                "execution": "Real QPU",
                "gradient": row["gradient_method"],
                "environment": "Real QPU",
                "target_environment": "Real QPU",
                "median_s_per_epoch": median_time,
                "speed_up_vs_cpu1": np.nan,
                "estimated_1000_epochs_h": median_time * 1000 / 3600,
                "measured_epochs": row["measured_epochs"],
            })
        return rows

    def training_cost_table(
        self,
        *,
        filters: dict[str, Any] | None = None,
        randomness_levels: Sequence[float] = (0, 1),
    ) -> list[dict[str, Any]]:
        """Report simulator speed-ups and real-QPU complete-training estimates."""

        table = self._training_cost_rows(
            filters=filters,
            randomness_levels=randomness_levels,
        )

        display_rows(
            table,
            columns=(
                "case",
                "gradient",
                "target_environment",
                "median_s_per_epoch",
                "speed_up_vs_cpu1",
                "estimated_1000_epochs_h",
                "measured_epochs",
            ),
            column_labels={
                "case": "Case",
                "gradient": "Gradient",
                "target_environment": "Target/environment",
                "median_s_per_epoch": "Median s/epoch",
                "speed_up_vs_cpu1": "Speed-up vs CPU/1",
                "estimated_1000_epochs_h": "Estimated 1000 epochs (h)",
                "measured_epochs": "Measured epochs",
            },
        )
        return table

    def training_cost_figure(
        self,
        *,
        filters: dict[str, Any] | None = None,
        randomness_levels: Sequence[float] = (0, 1),
    ) -> list[dict[str, Any]]:
        """Plot projected 1000-epoch cost as directly labelled lollipops."""

        import matplotlib.pyplot as plt

        rows = self._training_cost_rows(
            filters=filters,
            randomness_levels=randomness_levels,
        )
        rows = [
            row for row in rows
            if np.isfinite(row["estimated_1000_epochs_h"])
            and row["estimated_1000_epochs_h"] > 0
        ]
        fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(rows) + 1.8)))
        if rows:
            values = [float(row["estimated_1000_epochs_h"]) for row in rows]
            baseline = min(values) * 0.65
            positions = np.arange(len(rows))
            for position, row, value in zip(positions, rows, values):
                environment = row["environment"]
                color = TIMING_ENVIRONMENT_COLORS[environment]
                marker = "D" if environment == "Real QPU" else "o"
                ax.plot((baseline, value), (position, position), color=color, alpha=0.35)
                ax.scatter(value, position, color=color, marker=marker, s=62, zorder=3)
                speed_up = row["speed_up_vs_cpu1"]
                annotation = f"{value:.2g} h"
                if np.isfinite(speed_up) and environment != "CPU/1":
                    annotation += f" · {speed_up:.1f}× vs CPU/1"
                ax.annotate(
                    annotation,
                    (value, position),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )
            labels = [
                f"{row['case']} · {row['gradient']} · {row['environment']}"
                for row in rows
            ]
            ax.set_yticks(positions, labels)
            ax.invert_yaxis()
            ax.set_xscale("log")
            ax.set_xlim(baseline, max(values) * 3.6)
        else:
            ax.text(0.5, 0.5, "No timing data", transform=ax.transAxes, ha="center")
        ax.set_xlabel("Projected time for 1000 epochs (hours, log scale)")
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_title(
            "Projected Training Cost from Measured Median Epoch Time\n"
            "diamonds denote Real-QPU case studies"
        )
        self._finish(fig, "01e_projected_training_cost")
        return rows

    def _feasibility_results(
        self,
        *,
        learn_battery: str | Path | None = None,
        timing_batteries: Iterable[str | Path] | None = None,
    ) -> tuple[list[RunResult], Path, list[Path]]:
        battery_dir = self.results.qgan_dir / "configs" / "batteries" / "train"
        learn_battery = Path(
            learn_battery or battery_dir / "train_learn_gpu.yaml"
        )
        timing_batteries = [
            Path(path)
            for path in (
                timing_batteries
                or (
                    battery_dir / "train_times_cpu.yaml",
                    battery_dir / "train_times_gpu.yaml",
                    battery_dir / "train_times_rh.yaml",
                )
            )
        ]

        feasibility = expected_battery_results(
            battery_configs([learn_battery]),
            self.results.raw_learn,
            "learn",
        )
        feasibility.extend(expected_battery_results(
            battery_configs(timing_batteries),
            self.results.raw_timing,
            "timing",
        ))
        selected_paths = {run.path.resolve() for run in feasibility}
        feasibility.extend(
            run
            for run in known_oom_timing_results(self.results.raw_timing)
            if run.path.resolve() not in selected_paths
        )
        return feasibility, learn_battery, timing_batteries

    def feasibility(
        self,
        *,
        learn_battery: str | Path | None = None,
        timing_batteries: Iterable[str | Path] | None = None,
    ) -> list[RunResult]:
        import matplotlib.pyplot as plt

        feasibility, learn_file, timing_files = self._feasibility_results(
            learn_battery=learn_battery,
            timing_batteries=timing_batteries,
        )
        print("learning dynamics battery:", learn_file.name)
        print("timing batteries:", [path.name for path in timing_files])
        display_rows(
            LIMITATION_CASES,
            columns=("experiment", "limitation", "detail", "battery_reference"),
        )

        summary = results_table(feasibility)
        count_rows = []
        for source in ("learn", "timing"):
            for execution_type in ("noiseless", "noisy", "real"):
                selected = [
                    row
                    for row in summary
                    if row.get("analysis_source") == source
                    and row.get("execution_type") == execution_type
                ]
                if selected:
                    count_rows.append({
                        "source": "learning dynamics" if source == "learn" else source,
                        "execution_type": execution_type,
                        "requested": len(selected),
                        "complete": sum(row["completed_requested_budget"] for row in selected),
                        "not_complete": sum(not row["completed_requested_budget"] for row in selected),
                    })
        display_rows(count_rows)

        incomplete_rows = [
            {**row, "limitation": incomplete_limitation(row)}
            for row in summary
            if not row["completed_requested_budget"]
        ]
        qubit_order = (4, 8, 16)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_feasibility_matrix(
            feasibility,
            row_field="n_qubits",
            column_field="execution_type",
            ax=axes[0],
        )
        axes[0].set_yticks(
            np.arange(len(qubit_order)),
            [f"q{value}" for value in qubit_order],
        )
        axes[0].invert_yaxis()
        axes[0].set_title("Completion of Requested Experimental Runs")

        bottom = np.zeros(len(qubit_order))
        for limitation in LIMITATION_ORDER:
            counts = np.asarray([
                sum(
                    row["n_qubits"] == n_qubits and row["limitation"] == limitation
                    for row in incomplete_rows
                )
                for n_qubits in qubit_order
            ])
            container = axes[1].bar(
                np.arange(len(qubit_order)),
                counts,
                bottom=bottom,
                label=limitation,
            )
            axes[1].bar_label(
                container,
                labels=[str(value) if value else "" for value in counts],
                label_type="center",
                fontsize=8,
            )
            bottom += counts
        expected_incomplete = np.asarray([
            sum(row["n_qubits"] == n_qubits for row in incomplete_rows)
            for n_qubits in qubit_order
        ])
        if not np.array_equal(bottom.astype(int), expected_incomplete):
            raise AssertionError("Limitation counts do not match incomplete requests.")
        for index, total in enumerate(expected_incomplete):
            axes[1].text(index, total, str(total), ha="center", va="bottom", fontsize=9)
        axes[1].set_xticks(
            np.arange(len(qubit_order)),
            [f"q{value}" for value in qubit_order],
        )
        axes[1].set_ylabel("Number of Incomplete Requested Runs")
        axes[1].set_title("Incomplete Experimental Runs by Limiting Factor")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, axis="y", alpha=0.25)
        self._finish(fig, "01d_experimental_limitations_and_feasibility")
        return feasibility

    def preset_learning_dynamics(
        self,
        *,
        filters: dict[str, Any] | None = None,
        presets: Sequence[str] = PRESETS,
    ):
        import matplotlib.pyplot as plt

        preset_runs = filter_results(
            self.results.main_learn,
            **_merged_filters(
                {
                    "implementation": "qml_torch",
                    "execution_type": "noiseless",
                    "gradient_method": "PSR",
                    "n_qubits": 4,
                    "randomness": 0,
                },
                filters,
            ),
        )
        fig, axes = plt.subplots(
            2,
            len(presets),
            figsize=(5.3 * len(presets), 8),
            sharex="col",
            squeeze=False,
        )
        for column, preset in enumerate(presets):
            preset_name = preset.replace("_", " ").title()
            plot_training_dynamics_comparison(
                filter_results(preset_runs, preset=preset),
                compare_by="preset",
                axes=axes[:, column],
            )
            axes[0, column].set_title(f"{preset_name}: Adversarial Losses")
            axes[1, column].set_title(f"{preset_name}: Evaluation Score")
        fig.suptitle(
            "Training Dynamics Across Encoding Presets "
            "(Individual Runs, Median, and IQR)"
        )
        self._finish(fig, "02a_preset_learning_dynamics")
        return preset_runs

    def preset_best_results(
        self,
        runs: Iterable[RunResult] | None = None,
        *,
        presets: Sequence[str] = PRESETS,
    ):
        preset_runs = list(runs) if runs is not None else filter_results(
            self.results.main_learn,
            implementation="qml_torch",
            execution_type="noiseless",
            gradient_method="PSR",
            n_qubits=4,
            randomness=0,
        )
        self._faceted_best_results_figure(
            (
                (preset, filter_results(preset_runs, preset=preset))
                for preset in presets
            ),
            compare_by="preset",
            title="Evaluation Performance Across Encoding Presets",
            stem="02b_preset_best_results",
        )
        table = grouped_performance_table(
            preset_runs,
            fields=("eval_method", "preset", "n_qubits", "gradient_method", "randomness"),
        )
        display_rows(
            table,
            columns=(
                "eval_method",
                "preset",
                "completed_runs",
                "seeds",
                "median_best_eval",
                "q25_best_eval",
                "q75_best_eval",
                "median_average_best_so_far_eval",
            ),
        )
        return table

    def select_preset_outputs(
        self,
        *,
        filters: dict[str, Any] | None = None,
        presets: Sequence[str] = PRESETS,
        gradient_priority: Sequence[str] = GRADIENT_METHODS,
    ) -> list[dict[str, Any]]:
        candidates = filter_results(
            self.results.main_learn,
            **_merged_filters(
                {
                    "implementation": "qml_torch",
                    "execution_type": "noiseless",
                    "n_qubits": 4,
                },
                filters,
            ),
        )
        priority = {method: index for index, method in enumerate(gradient_priority)}
        selected_runs = {}
        for preset in presets:
            ranked = []
            for run in filter_results(candidates, preset=preset):
                summary = results_table([run])[0]
                if np.isfinite(summary["best_eval"]):
                    ranked.append((
                        float(summary["best_eval"]),
                        priority.get(run.metadata.get("gradient_method"), 99),
                        int(run.metadata.get("seed") or 0),
                        run.run_id,
                        run,
                        summary,
                    ))
            if ranked:
                selected_runs[preset] = min(ranked)[-2:]
        self._preset_output_runs = selected_runs

        manifest = []
        for preset, (run, summary) in selected_runs.items():
            manifest.append({
                "preset": preset,
                "run_id": run.run_id,
                "gradient_method": summary["gradient_method"],
                "randomness": summary["randomness"],
                "seed": summary["seed"],
                "evaluation_metric": summary["eval_method"],
                "best_eval": summary["best_eval"],
                "average_best_so_far_eval": summary["average_best_so_far_eval"],
                "epoch_of_best_eval": summary["epoch_of_best_eval"],
                "last_eval": summary["final_eval"],
                "config_file": str(run.path / "config.yaml"),
            })
        display_rows(manifest)
        return manifest

    def render_preset_outputs(self, *, random_seed: int = 0) -> list[dict[str, Any]]:
        from qgan_v2.visualization import plot_generated_output_sequence

        if not self._preset_output_runs:
            self.select_preset_outputs()
        manifest = []
        for preset, (run, _) in self._preset_output_runs.items():
            fig, record = plot_generated_output_sequence(
                run.path / "config.yaml",
                parameter_sets=("initial", "last", "best"),
                random_seed=random_seed,
            )
            manifest.append({"preset": preset, **record})
            self._finish(fig, f"02c_{preset}_initial_last_best_target")
        display_rows(manifest)
        return manifest

    def simulation_execution_dynamics(
        self,
        *,
        filters: dict[str, Any] | None = None,
        presets: Sequence[str] = PRESETS,
    ):
        import matplotlib.pyplot as plt

        runs = self._simulation_execution_runs(filters)
        fig, axes = plt.subplots(
            1,
            len(presets),
            figsize=(5.3 * len(presets), 4.8),
            sharex=True,
            squeeze=False,
        )
        for column, preset in enumerate(presets):
            plot_learn_comparison(
                filter_results(runs, preset=preset),
                compare_by="execution_type",
                metric="eval",
                ax=axes[0, column],
            )
            axes[0, column].set_title(
                f"Encoding Preset: {preset.replace('_', ' ').title()}"
            )
        fig.suptitle("Effect of Simulation Noise on Evaluation Learning Dynamics")
        self._finish(fig, "03a_noiseless_noisy_dynamics")
        return runs

    def _simulation_execution_runs(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[RunResult]:
        return filter_results(
            self.results.main_learn,
            **_merged_filters(
                {
                    "implementation": "qml_torch",
                    "gradient_method": "SPSA",
                    "n_qubits": 4,
                    "randomness": 0,
                    "execution_type": ("noiseless", "noisy"),
                },
                filters,
            ),
        )

    def simulation_execution_best_results(
        self,
        runs: Iterable[RunResult] | None = None,
        *,
        presets: Sequence[str] = PRESETS,
    ):
        runs = list(runs) if runs is not None else self._simulation_execution_runs()
        self._faceted_best_results_figure(
            (
                (preset, filter_results(runs, preset=preset))
                for preset in presets
            ),
            compare_by="execution_type",
            title="Effect of Simulation Noise on Evaluation Performance",
            stem="03b_noiseless_noisy_best_results",
        )
        return runs

    def real_hardware_comparison(
        self,
        *,
        filters: dict[str, Any] | None = None,
        eval_method: str = "kl",
    ) -> dict[str, list[RunResult]]:
        import matplotlib.pyplot as plt

        focus = _merged_filters(
            {
                "execution_type": "real",
                "implementation": "qml_torch",
                "preset": "base",
                "n_qubits": 4,
                "randomness": 0,
            },
            filters,
        )
        real_runs = filter_results(
            self.results.hardware,
            **focus,
        )
        cases = {}
        for gradient_method in unique_values(real_runs, "gradient_method"):
            hardware_group = filter_results(real_runs, gradient_method=gradient_method)
            simulator_focus = {
                field: value
                for field, value in focus.items()
                if field != "execution_type"
            }
            simulator_focus.update({
                "gradient_method": gradient_method,
                "eval_method": eval_method,
            })
            simulators = filter_results(
                self.results.main_learn,
                **simulator_focus,
            )
            budget = max(len(run.eval) for run in hardware_group)
            cases[gradient_method] = truncate_results(
                [*simulators, *hardware_group],
                max_epoch=budget - 1,
            )

        if cases:
            fig, axes = plt.subplots(
                1,
                len(cases),
                figsize=(7 * len(cases), 4.8),
                squeeze=False,
            )
            for column, (method, selected) in enumerate(cases.items()):
                plot_learn_comparison(
                    selected,
                    compare_by="execution_type",
                    metric="eval",
                    ax=axes[0, column],
                )
                axes[0, column].set_title(f"Gradient Method: {method}")
            fig.suptitle(
                "Evaluation Learning Dynamics on Matched Simulators and Quantum Hardware"
            )
            self._finish(fig, "03c_real_hardware_dynamics")

            self._faceted_best_results_figure(
                cases.items(),
                compare_by="execution_type",
                stem="03d_real_hardware_best_results",
                title=(
                    "Evaluation Performance on Matched Simulators and "
                    "Quantum Hardware"
                ),
                panel_width=6,
            )
        display_rows(
            results_table(real_runs),
            columns=(
                "run_id",
                "gradient_method",
                "completed_epochs",
                "best_eval",
                "average_best_so_far_eval",
                "median_time_per_epoch",
            ),
        )
        return cases

    def gradient_comparison(
        self,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RunResult]:
        runs = filter_results(
            self.results.main_learn,
            **_merged_filters(
                {
                    "preset": "ang",
                    "implementation": "qml_torch",
                    "execution_type": "noiseless",
                    "n_qubits": 4,
                    "randomness": 0,
                    "eval_method": "gradient",
                },
                filters,
            ),
        )
        self._evaluation_dynamics_figure(
            runs,
            "gradient_method",
            "Evaluation Learning Dynamics Across Gradient Methods",
            "04a_gradient_dynamics",
        )
        self._best_results_figure(
            runs,
            "gradient_method",
            "Evaluation Performance Across Gradient Methods",
            "04b_gradient_best_results",
        )
        return runs

    def randomness_sweep(
        self,
        *,
        filters: dict[str, Any] | None = None,
        levels: Sequence[float] = RANDOMNESS_LEVELS,
        required_seeds: Sequence[int] = (0, 1, 2),
        sweep_index: int = 0,
    ) -> list[RunResult]:
        import matplotlib.pyplot as plt

        levels = tuple(levels)
        sweeps = factor_sweep_groups(
            filter_results(self.results.main_learn, **(filters or {})),
            factor="randomness",
            required_levels=levels,
            required_seeds=required_seeds,
        )
        ordered = sorted(sweeps.items(), key=lambda item: str(item[0]))
        print("complete five-level × three-seed sweeps:", len(ordered))
        if not ordered:
            print("No fully matched randomness sweep is available.")
            return []
        if not -len(ordered) <= sweep_index < len(ordered):
            raise IndexError(
                f"sweep_index {sweep_index} is outside {len(ordered)} available sweeps"
            )

        _, runs = ordered[sweep_index]
        exemplar = runs[0]
        print("selected sweep:", {
            field: exemplar.metadata.get(field)
            for field in (
                "preset",
                "gradient_method",
                "n_qubits",
                "execution_type",
                "eval_method",
            )
        })
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = dict(
            zip(levels, comparison_line_colors(levels, field="randomness"))
        )
        level_order = {level: index for index, level in enumerate(levels)}
        for randomness in reversed(levels):
            plot_learn(
                filter_results(runs, randomness=randomness),
                metric="eval",
                center="median",
                spread="iqr",
                label=f"Input Randomness = {randomness}",
                color=colors[randomness],
                zorder=len(levels) - level_order[randomness] + 2,
                ax=ax,
            )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
        ax.set_ylabel("Evaluation Score")
        fig.suptitle("Effect of Input Randomness on Evaluation Learning Dynamics")
        self._finish(fig, "05a_randomness_complete_dynamics")

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, metric_name, title in (
            (axes[0], "best_eval", "Best Evaluation Score"),
            (axes[1], "average_best_so_far_eval", "Average Best Evaluation So Far"),
            (axes[2], "evaluation_step_volatility", "Evaluation-Step Volatility"),
        ):
            plot_final_metric(
                runs,
                compare_by="randomness",
                metric_name=metric_name,
                point_color_by="seed",
                ax=ax,
            )
            ax.set_title(title)
        fig.suptitle("Performance and Stability Across Input Randomness Levels")
        self._finish(fig, "05b_randomness_complete_best_and_volatility")
        return runs

    def render_randomness_output(
        self,
        *,
        filters: dict[str, Any] | None = None,
        random_seed: int | None = None,
        output_count: int = 1,
    ) -> dict[str, Any]:
        """Generate a fresh output from one completed run's best parameters."""

        from qgan_v2.visualization import plot_generated_output

        selected_filters = _merged_filters(
            {
                "implementation": "qml_torch",
                "execution_type": "noiseless",
                "gradient_method": "PSR",
                "n_qubits": 4,
            },
            filters,
        )
        candidates = filter_results(
            self.results.main_learn,
            **selected_filters,
        )
        if not candidates:
            raise ValueError(
                f"No completed randomness-output run matches {selected_filters}."
            )
        if len(candidates) > 1:
            run_ids = sorted(run.run_id for run in candidates)
            raise ValueError(
                "Randomness-output filters must select exactly one run; "
                f"matched {run_ids}."
            )

        run = candidates[0]
        fig, output_record = plot_generated_output(
            run.path / "config.yaml",
            parameter_set="best",
            random_seed=random_seed,
            num_outputs=output_count,
        )
        randomness = run.metadata.get("randomness")
        record = {
            "preset": run.metadata.get("preset"),
            "gradient_method": run.metadata.get("gradient_method"),
            "n_qubits": run.metadata.get("n_qubits"),
            "randomness": randomness,
            "training_seed": run.metadata.get("seed"),
            "random_input_applied": bool(randomness),
            **output_record,
        }
        display_rows([record])
        randomness_label = str(randomness).replace(".", "p")
        self._finish(
            fig,
            (
                f"05c_{record['preset']}_rand{randomness_label}_"
                f"seed{record['training_seed']}_best_output"
            ),
        )
        return record

    def randomness_pairs(
        self,
        *,
        filters: dict[str, Any] | None = None,
        baseline: float = 0,
        treatment: float = 1,
    ):
        import matplotlib.pyplot as plt

        pair_fields = tuple(
            field for field in SCIENTIFIC_PAIR_FIELDS if field != "randomness"
        )
        filters = _merged_filters(
            {
                "preset": "ang",
                "implementation": "qml_torch",
                "execution_type": "noiseless",
                "eval_method": "gradient",
            },
            filters,
        )
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, metric_name, title in (
            (axes[0], "best_eval", "Change in Best Evaluation Score"),
            (axes[1], "average_best_so_far_eval", "Change in Average Best Evaluation So Far"),
            (
                axes[2],
                "evaluation_step_volatility",
                "Change in Evaluation-Step Volatility",
            ),
        ):
            plot_paired_delta(
                self.results.main_learn,
                baseline_filters={**filters, "randomness": baseline},
                treatment_filters={**filters, "randomness": treatment},
                metric_name=metric_name,
                pair_fields=pair_fields,
                x_field="gradient_method",
                value="delta",
                ax=ax,
            )
            ax.set_title(title)
        fig.suptitle(
            f"Paired Effect of Input Randomness ({treatment} Relative to {baseline})"
        )
        self._finish(fig, "05d_randomness_rand0_rand1_paired")
        return fig, axes

    def _scaling_dynamics(
        self,
        runs: list[RunResult],
        *,
        facet_field: str,
        facet_values: Sequence[Any],
        title: str,
        stem: str,
        figsize: tuple[float, float],
        qubits_by_facet: dict[Any, Sequence[int]] | None = None,
        metric_transform: str = "none",
    ):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(facet_values), figsize=figsize, sharex=True)
        axes = np.atleast_1d(axes)
        for column, value in enumerate(facet_values):
            filters = {facet_field: value}
            if qubits_by_facet and value in qubits_by_facet:
                filters["n_qubits"] = qubits_by_facet[value]
            selected = filter_results(runs, **filters)
            plot_learn_comparison(
                selected,
                compare_by="n_qubits",
                metric="eval",
                transform=metric_transform,
                ax=axes[column],
            )
            display_value = str(value)
            if isinstance(value, str) and not value.isupper():
                display_value = value.replace("_", " ").title()
            axes[column].set_title(
                f"{FACET_DISPLAY_NAMES.get(facet_field, facet_field)}: "
                f"{display_value}"
            )
        fig.suptitle(title)
        self._finish(fig, stem)
        return fig, axes

    def _scaling_best_results(
        self,
        runs: list[RunResult],
        *,
        line_by: str,
        title: str,
        stem: str,
    ):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        plot_metric_by_numeric_field(
            runs,
            x_field="n_qubits",
            metric_name="best_eval",
            line_by=line_by,
            ax=axes[0],
        )
        axes[0].set_title("Best Evaluation Score by Number of Qubits")
        plot_metric_by_numeric_field(
            runs,
            x_field="n_qubits",
            metric_name="average_best_so_far_eval",
            line_by=line_by,
            ax=axes[1],
        )
        axes[1].set_title("Average Best Evaluation So Far by Number of Qubits")
        fig.suptitle(title)
        self._finish(fig, stem)
        return fig, axes

    def preset_scaling(
        self,
        *,
        filters: dict[str, Any] | None = None,
        presets: Sequence[str] = PRESETS,
        qubits_by_preset: dict[str, Sequence[int]] | None = None,
        metric_transform: str = "none",
    ) -> list[RunResult]:
        import matplotlib.pyplot as plt

        runs = filter_results(
            self.results.main_learn,
            **_merged_filters(
                {
                    "implementation": "qml_torch",
                    "execution_type": "noiseless",
                    "gradient_method": "SPSA",
                    "randomness": 0,
                },
                filters,
            ),
        )
        self._scaling_dynamics(
            runs,
            facet_field="preset",
            facet_values=presets,
            title="Evaluation Learning Dynamics Across Qubit Counts by Encoding Preset",
            stem="06a_preset_scaling_dynamics",
            figsize=(5.3 * len(presets), 4.8),
            qubits_by_facet=qubits_by_preset or {"amp": (4, 8)},
            metric_transform=metric_transform,
        )
        fig, axes = plt.subplots(
            2,
            len(presets),
            figsize=(5 * len(presets), 8),
            squeeze=False,
        )
        for column, preset in enumerate(presets):
            selected = filter_results(runs, preset=preset)
            plot_metric_by_numeric_field(
                selected,
                x_field="n_qubits",
                metric_name="best_eval",
                ax=axes[0, column],
            )
            preset_name = preset.replace("_", " ").title()
            axes[0, column].set_title(
                f"{preset_name}: Best Evaluation Score"
            )
            plot_metric_by_numeric_field(
                selected,
                x_field="n_qubits",
                metric_name="average_best_so_far_eval",
                ax=axes[1, column],
            )
            axes[1, column].set_title(
                f"{preset_name}: Average Best Evaluation So Far"
            )
        fig.suptitle("Performance Scaling Across Qubit Counts by Encoding Preset")
        self._finish(fig, "06b_preset_scaling_best_results")
        return runs

    def execution_scaling(
        self,
        *,
        filters: dict[str, Any] | None = None,
        execution_types: Sequence[str] = ("noiseless", "noisy"),
        metric_transform: str = "none",
    ) -> list[RunResult]:
        runs = filter_results(
            self.results.main_learn,
            **_merged_filters(
                {
                    "preset": "ang",
                    "implementation": "qml_torch",
                    "gradient_method": "SPSA",
                    "randomness": 0,
                    "execution_type": tuple(execution_types),
                },
                filters,
            ),
        )
        self._scaling_dynamics(
            runs,
            facet_field="execution_type",
            facet_values=execution_types,
            title="Evaluation Learning Dynamics Across Qubit Counts by Execution Type",
            stem="06c_execution_scaling_dynamics",
            figsize=(7 * len(execution_types), 4.8),
            metric_transform=metric_transform,
        )
        self._scaling_best_results(
            runs,
            line_by="execution_type",
            title="Performance Scaling Across Qubit Counts by Execution Type",
            stem="06d_execution_scaling_best_results",
        )
        return runs

    def gradient_scaling(
        self,
        *,
        filters: dict[str, Any] | None = None,
        gradient_methods: Sequence[str] = GRADIENT_METHODS,
        metric_transform: str = "none",
    ) -> list[RunResult]:
        runs = filter_results(
            self.results.main_learn,
            **_merged_filters(
                {
                    "preset": "ang",
                    "implementation": "qml_torch",
                    "execution_type": "noiseless",
                    "randomness": 0,
                },
                filters,
            ),
        )
        self._scaling_dynamics(
            runs,
            facet_field="gradient_method",
            facet_values=gradient_methods,
            title="Evaluation Learning Dynamics Across Qubit Counts by Gradient Method",
            stem="06e_gradient_scaling_dynamics",
            figsize=(6 * len(gradient_methods), 4.8),
            metric_transform=metric_transform,
        )
        self._scaling_best_results(
            runs,
            line_by="gradient_method",
            title="Performance Scaling Across Qubit Counts by Gradient Method",
            stem="06f_gradient_scaling_best_results",
        )
        return runs

    def randomness_scaling(
        self,
        *,
        filters: dict[str, Any] | None = None,
        randomness_levels: Sequence[float] = (0, 1),
        sweep_levels: Sequence[float] = RANDOMNESS_LEVELS,
        sweep_qubits: Sequence[int] = (4, 8),
        metric_transform: str = "none",
    ) -> list[RunResult]:
        all_runs = filter_results(
            self.results.main_learn,
            **_merged_filters(
                {
                    "preset": "ang",
                    "implementation": "qml_torch",
                    "execution_type": "noiseless",
                    "gradient_method": "SPSA",
                },
                filters,
            ),
        )
        sweep_runs = filter_results(
            all_runs,
            randomness=tuple(sweep_levels),
            n_qubits=tuple(sweep_qubits),
        )
        self._scaling_dynamics(
            sweep_runs,
            facet_field="randomness",
            facet_values=sweep_levels,
            title=(
                "Evaluation Learning Dynamics for q4 and q8 Across the Complete "
                "Randomness Sweep"
            ),
            stem="06g_randomness_complete_sweep_q4_q8_dynamics",
            figsize=(4.8 * len(sweep_levels), 4.8),
            metric_transform=metric_transform,
        )

        runs = filter_results(all_runs, randomness=tuple(randomness_levels))
        self._scaling_dynamics(
            runs,
            facet_field="randomness",
            facet_values=randomness_levels,
            title="Evaluation Learning Dynamics Across Qubit Counts by Input Randomness",
            stem="06h_randomness_scaling_dynamics",
            figsize=(7 * len(randomness_levels), 4.8),
            metric_transform=metric_transform,
        )
        self._scaling_best_results(
            runs,
            line_by="randomness",
            title="Performance Scaling Across Qubit Counts by Input Randomness",
            stem="06i_randomness_scaling_best_results",
        )
        return runs

    def implementation_validation(
        self,
        *,
        filters: dict[str, Any] | None = None,
        presets: Sequence[str] = PRESETS,
    ) -> list[RunResult]:
        import matplotlib.pyplot as plt

        runs = filter_results(
            self.results.validation,
            **_merged_filters(
                {
                    "execution_type": "fake_real",
                    "n_qubits": 4,
                    "gradient_method": "SPSA",
                    "randomness": 0,
                    "seed": 0,
                },
                filters,
            ),
        )
        fig, axes = plt.subplots(
            1,
            len(presets),
            figsize=(6.7 * len(presets), 4.8),
            sharex=True,
            squeeze=False,
        )
        for column, preset in enumerate(presets):
            plot_learn_comparison(
                filter_results(runs, preset=preset),
                compare_by="implementation_packing",
                metric="eval",
                ax=axes[0, column],
            )
            axes[0, column].set_title(
                f"Encoding Preset: {preset.replace('_', ' ').title()}"
            )
        fig.suptitle("Validation of Generator Implementations and Packing Strategies")
        self._finish(fig, "07_implementation_packing_validation")
        display_rows(
            results_table(runs),
            columns=(
                "run_id",
                "preset",
                "implementation_packing",
                "completed_epochs",
                "best_eval",
                "average_best_so_far_eval",
            ),
        )
        return runs


__all__ = [
    "GRADIENT_METHODS",
    "LIMITATION_CASES",
    "PRESETS",
    "RANDOMNESS_LEVELS",
    "ResultsAnalysis",
    "ResultsData",
    "battery_configs",
    "display_rows",
    "expected_battery_results",
    "incomplete_limitation",
    "scientific_config_key",
]
