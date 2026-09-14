"""Reusable analysis orchestration for the results notebook.

Low-level selection, aggregation, and plotting primitives live in
``qgan_v2.analysis.results``.  This module composes those primitives into the
specific analysis sections used by ``tutorial_experiments.ipynb`` so the notebook
can concentrate on explaining and invoking the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    load_results,
    metadata_from_config,
    plot_convergence,
    plot_convergence_comparison,
    plot_feasibility_matrix,
    plot_final_metric,
    plot_metric_by_category_field,
    plot_metric_by_numeric_field,
    plot_paired_delta,
    plot_training_dynamics_comparison,
    results_table,
    save_figure,
    select_completed_results,
    select_main_convergence_results,
    select_usable_results,
    truncate_results,
    unique_values,
)
from qgan_v2.config.battery import build_config_combinations, load_battery_file
from qgan_v2.storage.paths import get_config_filename


PRESETS = ("base", "ang", "amp")
GRADIENT_METHODS = ("PSR", "REG", "SPSA")
RANDOMNESS_LEVELS = (0, 0.1, 0.25, 0.5, 1)
FACET_DISPLAY_NAMES = {
    "execution_type": "Execution Type",
    "gradient_method": "Gradient Method",
    "preset": "Encoding Preset",
    "randomness": "Input Randomness",
}
LIMITATION_ORDER = (
    "time-expensive",
    "execution time unavailable",
    "out of memory",
    "other unfinished",
)

LIMITATION_CASES = (
    {
        "n_qubits": 4,
        "experiment": "noisy PSR",
        "limitation": "time-expensive",
        "detail": "Parameter-shift noisy convergence did not finish.",
        "battery_reference": "train_conv_cpu.yaml / train_conv_gpu.yaml",
    },
    {
        "n_qubits": 4,
        "experiment": "real hardware",
        "limitation": "execution time unavailable",
        "detail": "Only runs with completed real-hardware timing checkpoints are available.",
        "battery_reference": "train_times_rh.yaml",
    },
    {
        "n_qubits": 8,
        "experiment": "noisy PSR",
        "limitation": "time-expensive",
        "detail": "Parameter-shift noisy convergence did not finish.",
        "battery_reference": "train_conv_cpu.yaml / train_conv_gpu.yaml",
    },
    {
        "n_qubits": 16,
        "experiment": "noisy simulation on GPU",
        "limitation": "time-expensive",
        "detail": "Projected execution exceeded the available GPU allocation.",
        "battery_reference": "train_conv_gpu.yaml",
    },
    {
        "n_qubits": 16,
        "experiment": "amplitude preset",
        "limitation": "not transpilable",
        "detail": "The statevector preparation circuit was too large/deep to transpile.",
        "battery_reference": "train_conv_cpu.yaml / train_conv_gpu.yaml",
    },
    {
        "n_qubits": 16,
        "experiment": "noisy simulation on CPU",
        "limitation": "out of memory",
        "detail": "The density-matrix simulation exceeded CPU memory.",
        "battery_reference": "train_conv_cpu.yaml",
    },
)


def _merged_filters(
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    return {**defaults, **(overrides or {})}


def display_rows(
    rows: Iterable[dict[str, Any]],
    *,
    columns: Sequence[str] | None = None,
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

        display(pd.DataFrame(visible))
    except ImportError:
        for row in visible:
            print(row)
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
    if source == "convergence" and execution_type == "noisy" and row.get("n_qubits") == 16:
        if row.get("simulator_device") == "CPU":
            return "out of memory"
        return "time-expensive"
    if source == "convergence" and execution_type == "noisy" and row.get("gradient_method") == "PSR":
        return "time-expensive"
    return "other unfinished"


@dataclass
class ResultsData:
    """The result subsets shared by all analysis figures."""

    qgan_dir: Path
    raw_convergence: list[RunResult]
    raw_timing: list[RunResult]
    main_convergence: list[RunResult]
    timing: list[RunResult]
    hardware: list[RunResult]
    validation: list[RunResult]

    @classmethod
    def load(cls, qgan_dir: str | Path) -> "ResultsData":
        qgan_dir = Path(qgan_dir)
        raw_convergence = load_results(qgan_dir / "data" / "train")
        raw_timing = load_results(qgan_dir / "data" / "train" / "times")
        return cls(
            qgan_dir=qgan_dir,
            raw_convergence=raw_convergence,
            raw_timing=raw_timing,
            main_convergence=select_main_convergence_results(
                raw_convergence,
                expected_epochs=1000,
            ),
            timing=select_completed_results(raw_timing, expected_epochs=5),
            hardware=select_usable_results(raw_convergence, execution_types=("real",)),
            validation=select_usable_results(
                raw_convergence,
                execution_types=("fake_real",),
            ),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "raw_convergence/case-study_configs": len(self.raw_convergence),
            "completed_deduplicated_simulator_runs": len(self.main_convergence),
            "completed_five_epoch_timing_runs": len(self.timing),
            "usable_real_hardware_runs": len(self.hardware),
            "usable_implementation_validation_runs": len(self.validation),
            "evaluation_families": {
                name: len(runs)
                for name, runs in evaluation_metric_groups(self.main_convergence).items()
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

    def _finish(self, fig, stem: str) -> None:
        import matplotlib.pyplot as plt

        fig.tight_layout()
        if self.export_figures:
            for saved_path in save_figure(fig, self.figure_dir, stem):
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
        plot_convergence_comparison(
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
            metric_name="epoch_of_best_eval",
            point_color_by="seed",
            ax=axes[1],
        )
        axes[1].set_title("Epoch of Best Evaluation")
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
                (1, "epoch_of_best_eval", "Epoch of Best Evaluation"),
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
        factor_levels = {"randomness": (0, 1), **(factor_levels or {})}

        primary_timing = [
            run
            for run in self.results.timing
            if (
                run.metadata.get("simulator_device") == "GPU"
                or run.metadata.get("run_device") == "CPU"
            )
            and run.metadata.get("label") in (None, "cpu4")
        ]
        for run in primary_timing:
            if run.metadata.get("simulator_device") == "GPU":
                environment = f"GPU ({self.gpu_model_label})"
            elif run.metadata.get("label") == "cpu4":
                environment = "CPU/4"
            else:
                environment = "CPU/1"
            run.metadata["timing_environment"] = environment

        print("timing environments:", unique_values(primary_timing, "timing_environment"))
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
        fig.suptitle("Computational Cost Across Experimental Factors")
        self._finish(fig, "01a_timing_main_factors")
        return primary_timing

    def _feasibility_results(
        self,
        *,
        convergence_battery: str | Path | None = None,
        timing_batteries: Iterable[str | Path] | None = None,
    ) -> tuple[list[RunResult], Path, list[Path]]:
        battery_dir = self.results.qgan_dir / "configs" / "batteries" / "train"
        convergence_battery = Path(
            convergence_battery or battery_dir / "train_conv_gpu.yaml"
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
            battery_configs([convergence_battery]),
            self.results.raw_convergence,
            "convergence",
        )
        feasibility.extend(expected_battery_results(
            battery_configs(timing_batteries),
            self.results.raw_timing,
            "timing",
        ))
        return feasibility, convergence_battery, timing_batteries

    def feasibility(
        self,
        *,
        convergence_battery: str | Path | None = None,
        timing_batteries: Iterable[str | Path] | None = None,
    ) -> list[RunResult]:
        import matplotlib.pyplot as plt

        feasibility, convergence_file, timing_files = self._feasibility_results(
            convergence_battery=convergence_battery,
            timing_batteries=timing_batteries,
        )
        print("convergence battery:", convergence_file.name)
        print("timing batteries:", [path.name for path in timing_files])
        display_rows(sorted(LIMITATION_CASES, key=lambda row: (row["n_qubits"], row["experiment"])))

        summary = results_table(feasibility)
        count_rows = []
        for source in ("convergence", "timing"):
            for execution_type in ("noiseless", "noisy", "real"):
                selected = [
                    row
                    for row in summary
                    if row.get("analysis_source") == source
                    and row.get("execution_type") == execution_type
                ]
                if selected:
                    count_rows.append({
                        "source": source,
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
        self._finish(fig, "01b_experimental_limitations_and_feasibility")
        return feasibility

    def preset_learning_dynamics(
        self,
        *,
        filters: dict[str, Any] | None = None,
        presets: Sequence[str] = PRESETS,
    ):
        import matplotlib.pyplot as plt

        preset_runs = filter_results(
            self.results.main_convergence,
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
            self.results.main_convergence,
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
                "median_epoch_of_best_eval",
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
            self.results.main_convergence,
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
            plot_convergence_comparison(
                filter_results(runs, preset=preset),
                compare_by="execution_type",
                metric="eval",
                ax=axes[0, column],
            )
            axes[0, column].set_title(
                f"Encoding Preset: {preset.replace('_', ' ').title()}"
            )
        fig.suptitle("Effect of Simulation Noise on Evaluation Convergence")
        self._finish(fig, "03a_noiseless_noisy_dynamics")
        return runs

    def _simulation_execution_runs(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[RunResult]:
        return filter_results(
            self.results.main_convergence,
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
                self.results.main_convergence,
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
                plot_convergence_comparison(
                    selected,
                    compare_by="execution_type",
                    metric="eval",
                    ax=axes[0, column],
                )
                axes[0, column].set_title(f"Gradient Method: {method}")
            fig.suptitle(
                "Evaluation Convergence on Matched Simulators and Quantum Hardware"
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
                "epoch_of_best_eval",
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
            self.results.main_convergence,
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
            "Evaluation Convergence Across Gradient Methods",
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
            filter_results(self.results.main_convergence, **(filters or {})),
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
            plot_convergence(
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
        fig.suptitle("Effect of Input Randomness on Evaluation Convergence")
        self._finish(fig, "05a_randomness_complete_dynamics")

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, metric_name, title in (
            (axes[0], "best_eval", "Best Evaluation Score"),
            (axes[1], "epoch_of_best_eval", "Epoch of Best Evaluation"),
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
            self.results.main_convergence,
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
            (axes[1], "epoch_of_best_eval", "Change in Epoch of Best Evaluation"),
            (
                axes[2],
                "evaluation_step_volatility",
                "Change in Evaluation-Step Volatility",
            ),
        ):
            plot_paired_delta(
                self.results.main_convergence,
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
    ):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(facet_values), figsize=figsize, sharex=True)
        axes = np.atleast_1d(axes)
        for column, value in enumerate(facet_values):
            filters = {facet_field: value}
            if qubits_by_facet and value in qubits_by_facet:
                filters["n_qubits"] = qubits_by_facet[value]
            selected = filter_results(runs, **filters)
            plot_convergence_comparison(
                selected,
                compare_by="n_qubits",
                metric="eval",
                ax=axes[column],
            )
            axes[column].set_ylabel("Evaluation Score")
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
            metric_name="epoch_of_best_eval",
            line_by=line_by,
            ax=axes[1],
        )
        axes[1].set_title("Epoch of Best Evaluation by Number of Qubits")
        fig.suptitle(title)
        self._finish(fig, stem)
        return fig, axes

    def preset_scaling(
        self,
        *,
        filters: dict[str, Any] | None = None,
        presets: Sequence[str] = PRESETS,
        qubits_by_preset: dict[str, Sequence[int]] | None = None,
    ) -> list[RunResult]:
        import matplotlib.pyplot as plt

        runs = filter_results(
            self.results.main_convergence,
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
            title="Evaluation Convergence Across Qubit Counts by Encoding Preset",
            stem="06a_preset_scaling_dynamics",
            figsize=(5.3 * len(presets), 4.8),
            qubits_by_facet=qubits_by_preset or {"amp": (4, 8)},
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
                metric_name="epoch_of_best_eval",
                ax=axes[1, column],
            )
            axes[1, column].set_title(
                f"{preset_name}: Epoch of Best Evaluation"
            )
        fig.suptitle("Performance Scaling Across Qubit Counts by Encoding Preset")
        self._finish(fig, "06b_preset_scaling_best_results")
        return runs

    def execution_scaling(
        self,
        *,
        filters: dict[str, Any] | None = None,
        execution_types: Sequence[str] = ("noiseless", "noisy"),
    ) -> list[RunResult]:
        runs = filter_results(
            self.results.main_convergence,
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
            title="Evaluation Convergence Across Qubit Counts by Execution Type",
            stem="06c_execution_scaling_dynamics",
            figsize=(7 * len(execution_types), 4.8),
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
    ) -> list[RunResult]:
        runs = filter_results(
            self.results.main_convergence,
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
            title="Evaluation Convergence Across Qubit Counts by Gradient Method",
            stem="06e_gradient_scaling_dynamics",
            figsize=(6 * len(gradient_methods), 4.8),
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
    ) -> list[RunResult]:
        runs = filter_results(
            self.results.main_convergence,
            **_merged_filters(
                {
                    "preset": "ang",
                    "implementation": "qml_torch",
                    "execution_type": "noiseless",
                    "gradient_method": "SPSA",
                    "randomness": tuple(randomness_levels),
                },
                filters,
            ),
        )
        self._scaling_dynamics(
            runs,
            facet_field="randomness",
            facet_values=randomness_levels,
            title="Evaluation Convergence Across Qubit Counts by Input Randomness",
            stem="06g_randomness_scaling_dynamics",
            figsize=(7 * len(randomness_levels), 4.8),
        )
        self._scaling_best_results(
            runs,
            line_by="randomness",
            title="Performance Scaling Across Qubit Counts by Input Randomness",
            stem="06h_randomness_scaling_best_results",
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
            plot_convergence_comparison(
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
                "epoch_of_best_eval",
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
