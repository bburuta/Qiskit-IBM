import math
from pathlib import Path

import pytest

from qgan_v2.analysis.results import RunResult, is_completed
from qgan_v2.analysis.workflow import (
    LIMITATION_CASES,
    ResultsAnalysis,
    ResultsData,
    expected_battery_results,
    incomplete_limitation,
    known_oom_timing_results,
    scientific_config_key,
    timing_configuration_key,
)


def make_config(*, device="CPU", randomness=0):
    return {
        "run": {"id": "run", "seed": 2},
        "experiment": {
            "implementation": "ang",
            "execution_type": "noisy",
            "gradient_method": "PSR",
            "n_qubits": 4,
        },
        "implementation": {
            "name": "qml_torch",
            "discriminator_packing": "separate",
        },
        "encoding": {
            "random_circuit": 7,
            "randomness": randomness,
            "batch_size": 4,
            "eval_batch_size": 8,
            "eval_method": "gradient",
        },
        "training": {
            "learning_rate": 0.01,
            "max_iterations": 1000,
            "gen_iterations": 1,
            "disc_iterations": 1,
        },
        "backend": {
            "precision": 0.01,
            "simulator": {
                "device": device,
                "noiseless_method": "statevector",
                "noisy_method": "density_matrix",
                "noisy_backend_mapping": "hardware",
            },
        },
    }


def make_timing_result(
    tmp_path,
    run_id,
    *,
    environment="CPU/1",
    execution_type="noiseless",
    gradient_method="SPSA",
    preset=None,
    n_qubits=4,
    randomness=0,
    seconds=2.0,
    epochs=5,
):
    is_gpu = environment == "GPU"
    return RunResult(
        path=tmp_path / run_id,
        config={},
        metadata={
            "run_id": run_id,
            "label": "cpu4" if environment == "CPU/4" else None,
            "preset": preset or ("base" if execution_type == "real" else "ang"),
            "implementation": "qml_torch",
            "execution_type": execution_type,
            "gradient_method": gradient_method,
            "n_qubits": n_qubits,
            "randomness": randomness,
            "run_device": "GPU" if is_gpu else "CPU",
            "simulator_device": "GPU" if is_gpu else "CPU",
            "cpu_threads": 4 if environment == "CPU/4" else 0,
            "max_iterations": epochs,
        },
        eval={epoch: 1.0 for epoch in range(epochs)},
        gloss={},
        dloss={},
        times={epoch: seconds for epoch in range(epochs)},
    )


def test_scientific_config_key_ignores_device_and_unused_random_circuit():
    cpu = make_config(device="CPU", randomness=0)
    gpu = make_config(device="GPU", randomness=0)
    gpu["encoding"]["random_circuit"] = 99

    assert scientific_config_key(cpu) == scientific_config_key(gpu)


def test_expected_battery_results_creates_a_missing_result(monkeypatch, tmp_path):
    config = make_config()
    run_path = tmp_path / "run"
    monkeypatch.setattr(
        "qgan_v2.analysis.workflow.get_config_filename",
        lambda value: run_path / "config.yaml",
    )

    result = expected_battery_results([config], [], "convergence")[0]

    assert result.path == run_path
    assert result.status == "missing_training_data"
    assert result.metadata["analysis_source"] == "convergence"


def test_expected_result_uses_battery_budget_without_hardware_special_case(
    monkeypatch,
    tmp_path,
):
    config = make_config()
    config["experiment"]["execution_type"] = "real"
    run_path = tmp_path / "run"
    observed = RunResult(
        path=run_path,
        config=config,
        metadata={"execution_type": "real", "max_iterations": 1000},
        eval={0: 1.0, 1: 0.5},
        gloss={},
        dloss={},
        times={},
    )
    monkeypatch.setattr(
        "qgan_v2.analysis.workflow.get_config_filename",
        lambda value: run_path / "config.yaml",
    )

    result = expected_battery_results([config], [observed], "convergence")[0]

    assert result.metadata["max_iterations"] == 1000
    assert not is_completed(result)


def test_expected_result_is_complete_at_or_above_battery_budget(monkeypatch, tmp_path):
    config = make_config()
    config["training"]["max_iterations"] = 5
    run_path = tmp_path / "run"
    observed = RunResult(
        path=run_path,
        config=config,
        metadata={"max_iterations": 1000, "simulator_device": "GPU"},
        eval={epoch: float(epoch) for epoch in range(6)},
        gloss={},
        dloss={},
        times={},
    )
    monkeypatch.setattr(
        "qgan_v2.analysis.workflow.get_config_filename",
        lambda value: run_path / "config.yaml",
    )

    result = expected_battery_results([config], [observed], "timing")[0]

    assert result.metadata["max_iterations"] == 5
    assert result.metadata["simulator_device"] == "CPU"
    assert is_completed(result)


def test_default_feasibility_sources_use_one_convergence_and_three_timing_batteries(
    monkeypatch,
    tmp_path,
):
    data = ResultsData(tmp_path, [], [], [], [], [], [])
    analysis = ResultsAnalysis(data)
    expanded = []

    def record_batteries(paths, **kwargs):
        expanded.append([Path(path).name for path in paths])
        return []

    monkeypatch.setattr("qgan_v2.analysis.workflow.battery_configs", record_batteries)

    results, convergence_battery, timing_batteries = analysis._feasibility_results()

    assert results == []
    assert convergence_battery.name == "train_conv_gpu.yaml"
    assert expanded == [["train_conv_gpu.yaml"], [
        "train_times_cpu.yaml",
        "train_times_gpu.yaml",
        "train_times_rh.yaml",
    ]]
    assert [path.name for path in timing_batteries] == expanded[1]


def test_incomplete_limitation_classifies_known_cases():
    assert incomplete_limitation({
        "analysis_source": "timing",
        "execution_type": "real",
    }) == "execution time unavailable"
    assert incomplete_limitation({
        "analysis_source": "timing",
        "execution_type": "noisy",
        "n_qubits": 16,
        "simulator_device": "CPU",
    }) == "out of memory"
    assert incomplete_limitation({
        "analysis_source": "convergence",
        "execution_type": "noisy",
        "gradient_method": "SPSA",
        "n_qubits": 16,
        "simulator_device": "CPU",
    }) == "out of memory"
    assert incomplete_limitation({
        "analysis_source": "convergence",
        "execution_type": "noisy",
        "gradient_method": "SPSA",
        "n_qubits": 16,
        "simulator_device": "GPU",
    }) == "time-expensive"


def test_known_oom_timing_results_selects_incomplete_base_and_angle_q16(tmp_path):
    base = make_timing_result(
        tmp_path,
        "base-oom",
        execution_type="noisy",
        preset="base",
        n_qubits=16,
    )
    angle = make_timing_result(
        tmp_path,
        "angle-oom",
        execution_type="noisy",
        preset="ang",
        n_qubits=16,
    )
    amplitude = make_timing_result(
        tmp_path,
        "amplitude-not-transpilable",
        execution_type="noisy",
        preset="amp",
        n_qubits=16,
    )
    q8 = make_timing_result(
        tmp_path,
        "q8",
        execution_type="noisy",
        preset="base",
        n_qubits=8,
    )
    completed = make_timing_result(
        tmp_path,
        "completed",
        execution_type="noisy",
        preset="base",
        n_qubits=16,
    )
    for run in (base, angle, amplitude, q8):
        run.eval = {}
        run.times = {}

    selected = known_oom_timing_results((base, angle, amplitude, q8, completed))

    assert [run.run_id for run in selected] == ["base-oom", "angle-oom"]
    assert all(run.metadata["analysis_source"] == "timing" for run in selected)


def test_limitation_cases_join_experiment_and_qubit_scope():
    assert [row["experiment"] for row in LIMITATION_CASES] == [
        "amplitude q16",
        "noisy q16 CPU timing",
        "real hardware q4",
        "noisy q4/q8 PSR convergence",
        "noisy q16 convergence",
    ]
    assert all("n_qubits" not in row for row in LIMITATION_CASES)
    assert [row["limitation"] for row in LIMITATION_CASES] == [
        "not transpilable",
        "out of memory",
        "execution time unavailable",
        "time-expensive",
        "time-expensive",
    ]


def test_timing_environment_labels_real_hardware_explicitly(tmp_path):
    timing = [
        make_timing_result(tmp_path, "cpu1"),
        make_timing_result(tmp_path, "cpu4", environment="CPU/4"),
        make_timing_result(tmp_path, "gpu", environment="GPU"),
        make_timing_result(tmp_path, "real", execution_type="real"),
    ]
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], [], timing, [], []))

    labelled = {
        run.run_id: run.metadata["timing_environment"]
        for run in analysis._timing_environment_results()
    }

    assert labelled == {
        "cpu1": "CPU/1",
        "cpu4": "CPU/4",
        "gpu": "GPU",
        "real": "Real QPU",
    }


def test_timing_configuration_key_ignores_environment_and_optionally_qubits(
    tmp_path,
):
    cpu = make_timing_result(tmp_path, "cpu")
    gpu = make_timing_result(tmp_path, "gpu", environment="GPU")
    q8 = make_timing_result(tmp_path, "q8")
    q8.metadata["n_qubits"] = 8

    assert timing_configuration_key(cpu) == timing_configuration_key(gpu)
    assert timing_configuration_key(cpu) != timing_configuration_key(q8)
    assert timing_configuration_key(
        cpu,
        include_qubits=False,
    ) == timing_configuration_key(q8, include_qubits=False)

def test_training_cost_table_has_speedups_randomness_and_real_cases(
    monkeypatch,
    tmp_path,
):
    timing = []
    for randomness in (0, 1):
        for environment, seconds in (("CPU/1", 8.0), ("CPU/4", 2.0), ("GPU", 1.0)):
            timing.append(make_timing_result(
                tmp_path,
                f"rand{randomness}-{environment}",
                environment=environment,
                randomness=randomness,
                seconds=seconds,
            ))
    timing.extend((
        make_timing_result(
            tmp_path,
            "real-psr",
            execution_type="real",
            gradient_method="PSR",
            seconds=30.0,
            epochs=3,
        ),
        make_timing_result(
            tmp_path,
            "real-spsa",
            execution_type="real",
            gradient_method="SPSA",
            seconds=20.0,
            epochs=4,
        ),
    ))
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], [], timing, [], []))
    monkeypatch.setattr(
        "qgan_v2.analysis.workflow.display_rows",
        lambda rows, **kwargs: rows,
    )

    table = analysis.training_cost_table()

    assert len(table) == 8
    assert {row["case"] for row in table[:6]} == {"rand0", "rand1"}
    gpu_row = next(
        row for row in table
        if row["case"] == "rand0" and row["environment"] == "GPU"
    )
    assert gpu_row["speed_up_vs_cpu1"] == pytest.approx(8.0)
    assert gpu_row["estimated_1000_epochs_h"] == pytest.approx(1000 / 3600)
    assert gpu_row["measured_epochs"] == 5
    real_rows = [row for row in table if row["execution"] == "Real QPU"]
    assert [row["gradient"] for row in real_rows] == ["PSR", "SPSA"]
    assert [row["measured_epochs"] for row in real_rows] == [3, 4]
    assert all(math.isnan(row["speed_up_vs_cpu1"]) for row in real_rows)


def test_new_timing_figures_show_matched_raw_and_projected_data(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("matplotlib")
    timing = []
    for environment, seconds in (("CPU/1", 8.0), ("CPU/4", 2.0), ("GPU", 1.0)):
        run = make_timing_result(
            tmp_path,
            environment,
            environment=environment,
            seconds=seconds,
        )
        run.times = {0: seconds * 2, 1: seconds, 2: seconds * 0.9}
        timing.append(run)
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], [], timing, [], []))
    figures = []
    monkeypatch.setattr(
        analysis,
        "_finish",
        lambda figure, stem, **kwargs: figures.append((figure, stem)),
    )

    matched = analysis.matched_timing_comparison(
        preset="ang",
        execution_types=("noiseless",),
    )
    raw = analysis.timing_distributions(
        preset="ang",
        execution_type="noiseless",
        n_qubits=4,
        gradient_method="SPSA",
        randomness=0,
    )
    projected = analysis.training_cost_figure(randomness_levels=(0,))

    assert len(matched) == 3
    assert len(raw) == 3
    assert len(projected) == 3
    assert [stem for _, stem in figures] == [
        "01a_ang_matched_environment_runtime",
        "01b_device_timing_distributions",
        "01e_projected_training_cost",
    ]
    matched_axis = figures[0][0].axes[0]
    assert matched_axis.get_xscale() == "linear"
    assert matched_axis.get_yscale() == "log"
    assert [label.get_text() for label in matched_axis.get_xticklabels()] == [
        "q4 · SPSA",
    ]
    assert {text.get_text() for text in matched_axis.texts} >= {"4.0×", "8.0×"}
    raw_figure = figures[1][0]
    raw_axis = raw_figure.axes[0]
    assert raw_axis.get_yscale() == "log"
    assert len(raw_axis.collections) == 6
    assert raw_axis.collections[1].get_offsets()[0, 1] == pytest.approx(8.0)
    assert raw_axis.collections[0].get_alpha() == pytest.approx(0.18)
    assert raw_axis.collections[1].get_sizes()[0] == pytest.approx(32)
    assert raw_axis.collections[1].get_alpha() == pytest.approx(0.95)
    median_vertices = raw_axis.collections[1].get_paths()[0].vertices
    marker_width = max(vertex[0] for vertex in median_vertices) - min(
        vertex[0] for vertex in median_vertices
    )
    marker_height = max(vertex[1] for vertex in median_vertices) - min(
        vertex[1] for vertex in median_vertices
    )
    assert marker_width == pytest.approx(marker_height)
    assert [text.get_text() for text in raw_figure.legends[0].get_texts()] == [
        "epoch time (shadow)",
        "run median",
    ]
    assert figures[2][0].axes[0].get_xscale() == "log"


def test_timing_distributions_accepts_scientific_selectors_and_sequences(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("matplotlib")
    timing = []
    for randomness in (0, 1):
        for environment, seconds in (("CPU/1", 4.0), ("GPU", 1.0)):
            timing.append(make_timing_result(
                tmp_path,
                f"rand{randomness}-{environment}",
                environment=environment,
                preset="base",
                execution_type="noisy",
                n_qubits=8,
                gradient_method="PSR",
                randomness=randomness,
                seconds=seconds,
            ))
    timing.append(make_timing_result(tmp_path, "unmatched-default"))
    timing.append(make_timing_result(
        tmp_path,
        "real-rand0",
        environment="Real QPU",
        execution_type="real",
        randomness=0,
    ))
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], [], timing, [], []))
    figures = []
    monkeypatch.setattr(
        analysis,
        "_finish",
        lambda figure, stem, **kwargs: figures.append((figure, stem)),
    )

    selected = analysis.timing_distributions(
        preset="base",
        execution_type="noisy",
        n_qubits=8,
        gradient_method="PSR",
        randomness=(0, 1),
        environments=("CPU/1", "GPU"),
    )

    figure, stem = figures[0]
    assert len(selected) == 4
    assert stem == "01b_device_timing_distributions"
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == [
        "base · noisy · q8\nPSR · rand0",
        "base · noisy · q8\nPSR · rand1",
    ]
    assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == [
        "CPU/1",
        "GPU",
    ]

    pooled = analysis.timing_distributions(compare_by="randomness")

    pooled_figure, pooled_stem = figures[1]
    assert len(pooled) == 6
    assert pooled_stem == "01b_device_timing_distributions"
    assert [axis.get_title() for axis in pooled_figure.axes] == [
        "CPU/1",
        "GPU",
        "Real QPU",
    ]
    assert {
        axis.get_subplotspec().rowspan.start
        for axis in pooled_figure.axes
    } == {0}
    assert [label.get_text() for label in pooled_figure.axes[0].get_xticklabels()] == [
        "rand0",
        "rand1",
    ]
    assert [
        label.get_text()
        for label in pooled_figure.axes[2].get_xticklabels()
    ] == ["rand0"]
    assert pooled_figure.axes[0].get_xlim() == pytest.approx((0.5, 2.5))
    assert pooled_figure.axes[2].get_xlim() == pytest.approx((0.5, 1.5))
    assert (
        pooled_figure.axes[0]
        .get_subplotspec()
        .get_gridspec()
        .get_width_ratios()
    ) == [2, 2, 1]


def test_timing_distributions_compare_by_uses_other_arguments_as_filters(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("matplotlib")
    timing = [
        make_timing_result(
            tmp_path,
            f"{preset}-rand{randomness}",
            preset=preset,
            randomness=randomness,
        )
        for preset in ("base", "ang")
        for randomness in (0, 1)
    ]
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], [], timing, [], []))
    monkeypatch.setattr(analysis, "_finish", lambda *args, **kwargs: None)

    selected = analysis.timing_distributions(
        compare_by="randomness",
        preset="base",
        environments=("CPU/1",),
    )

    assert {run.run_id for run in selected} == {"base-rand0", "base-rand1"}


def test_timing_distributions_rejects_unknown_compare_by(tmp_path):
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], [], [], [], []))

    with pytest.raises(ValueError, match="compare_by must be one of"):
        analysis.timing_distributions(compare_by="device")


def test_timing_distributions_accepts_custom_layout(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    timing = [
        make_timing_result(tmp_path, "cpu1"),
        make_timing_result(tmp_path, "cpu4", environment="CPU/4"),
        make_timing_result(tmp_path, "gpu", environment="GPU"),
        make_timing_result(
            tmp_path,
            "real",
            environment="Real QPU",
            execution_type="real",
        ),
    ]
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], [], timing, [], []))
    figures = []
    monkeypatch.setattr(
        analysis,
        "_finish",
        lambda figure, stem, **kwargs: figures.append(figure),
    )

    analysis.timing_distributions(compare_by="randomness", layout=(2, 2))

    assert {
        axis.get_subplotspec().rowspan.start
        for axis in figures[0].axes
    } == {0, 1}
    assert {
        axis.get_subplotspec().colspan.start
        for axis in figures[0].axes
    } == {0, 1}
    with pytest.raises(ValueError, match="not have enough cells"):
        analysis.timing_distributions(compare_by="randomness", layout=(1, 3))


def test_timing_distributions_does_not_render_empty_panels(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], [], [], [], []))
    figures = []
    monkeypatch.setattr(
        analysis,
        "_finish",
        lambda figure, stem, **kwargs: figures.append(figure),
    )

    selected = analysis.timing_distributions(compare_by="randomness")
    selected_qubits = analysis.timing_distributions(compare_by="n_qubits")

    assert selected == []
    assert selected_qubits == []
    assert figures == []


def test_matched_timing_groups_qubits_and_gradients_and_adds_real_to_noisy(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("matplotlib")
    timing = []
    for gradient_method in ("SPSA", "PSR"):
        for environment, seconds in (("CPU/1", 8.0), ("GPU", 1.0)):
            timing.append(make_timing_result(
                tmp_path,
                f"q4-{gradient_method}-{environment}",
                environment=environment,
                execution_type="noisy",
                gradient_method=gradient_method,
                preset="base",
                seconds=seconds,
            ))
    timing.append(make_timing_result(
        tmp_path,
        "q16-SPSA-GPU",
        environment="GPU",
        execution_type="noisy",
        gradient_method="SPSA",
        preset="base",
        n_qubits=16,
        seconds=100.0,
    ))
    for environment, seconds in (("CPU/1", 4.0), ("GPU", 2.0)):
        timing.append(make_timing_result(
            tmp_path,
            f"q4-REG-noiseless-{environment}",
            environment=environment,
            execution_type="noiseless",
            gradient_method="REG",
            preset="base",
            seconds=seconds,
        ))
    for gradient_method, seconds in (("SPSA", 20.0), ("PSR", 30.0)):
        timing.append(make_timing_result(
            tmp_path,
            f"real-{gradient_method}",
            environment="Real QPU",
            execution_type="real",
            gradient_method=gradient_method,
            preset="base",
            seconds=seconds,
        ))

    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], [], timing, [], []))
    figures = []
    monkeypatch.setattr(
        analysis,
        "_finish",
        lambda figure, stem, **kwargs: figures.append((figure, stem)),
    )

    selected = analysis.matched_timing_comparison(preset="base")

    figure, stem = figures[0]
    assert len(selected) == 9
    assert stem == "01a_base_matched_environment_runtime"
    assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == [
        "q4 · REG",
    ]
    assert [label.get_text() for label in figure.axes[1].get_xticklabels()] == [
        "q4 · SPSA",
        "q4 · PSR",
        "q16 · SPSA",
    ]
    assert figure.axes[0].get_title() == "Noiseless"
    assert figure.axes[1].get_title() == "Noisy"
    assert len(figure.axes[0].collections) == 2
    assert len(figure.axes[1].collections) == 7
    assert {
        float(collection.get_offsets()[0, 0])
        for collection in figure.axes[1].collections
    } == {0.0, 1.0, 2.0}
    assert {text.get_text() for text in figure.axes[1].texts} >= {
        "8.0×",
        "0.4×",
        "0.3×",
    }


def test_randomness_sweep_filters_candidates_before_grouping(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    runs = [
        RunResult(tmp_path / preset, {}, {"preset": preset}, {}, {}, {}, {})
        for preset in ("amp", "ang", "base")
    ]
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], runs, [], [], []))
    captured = {}

    def capture_groups(results, **kwargs):
        captured["presets"] = [run.metadata["preset"] for run in results]
        return {}

    monkeypatch.setattr(
        "qgan_v2.analysis.workflow.factor_sweep_groups",
        capture_groups,
    )

    selected = analysis.randomness_sweep(filters={"preset": "base"})

    assert selected == []
    assert captured["presets"] == ["base"]


def test_randomness_scaling_puts_complete_q4_q8_sweep_first(monkeypatch, tmp_path):
    runs = []
    for randomness in (0, 0.1, 0.25, 0.5, 1):
        for n_qubits in (4, 8):
            runs.append(
                RunResult(
                    tmp_path / f"q{n_qubits}-rand{randomness}",
                    {},
                    {
                        "preset": "ang",
                        "implementation": "qml_torch",
                        "execution_type": "noiseless",
                        "gradient_method": "PSR",
                        "n_qubits": n_qubits,
                        "randomness": randomness,
                    },
                    {0: 1.0},
                    {},
                    {},
                    {},
                )
            )
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], runs, [], [], []))
    dynamics_calls = []

    monkeypatch.setattr(
        analysis,
        "_scaling_dynamics",
        lambda selected, **kwargs: dynamics_calls.append((selected, kwargs)),
    )
    monkeypatch.setattr(analysis, "_scaling_best_results", lambda *args, **kwargs: None)

    selected = analysis.randomness_scaling(
        filters={"gradient_method": "PSR"},
        randomness_levels=(0, 1),
        metric_transform="normalize",
    )

    first_runs, first_options = dynamics_calls[0]
    assert len(first_runs) == 10
    assert first_options["facet_values"] == (0, 0.1, 0.25, 0.5, 1)
    assert first_options["metric_transform"] == "normalize"
    assert {run.metadata["n_qubits"] for run in first_runs} == {4, 8}
    assert {run.metadata["randomness"] for run in selected} == {0, 1}


def test_render_randomness_output_uses_one_run_and_best_parameters(
    monkeypatch,
    tmp_path,
):
    metadata = {
        "preset": "base",
        "implementation": "qml_torch",
        "execution_type": "noiseless",
        "gradient_method": "PSR",
        "n_qubits": 4,
        "randomness": 0.5,
        "seed": 2,
    }
    run = RunResult(tmp_path / "selected", {}, metadata, {}, {}, {}, {})
    analysis = ResultsAnalysis(ResultsData(tmp_path, [], [], [run], [], [], []))
    plotted = {}

    def plot_output(config_file, **kwargs):
        plotted.update(config_file=config_file, **kwargs)
        return object(), {
            "run_id": "selected",
            "random_seed": kwargs["random_seed"],
            "num_outputs": kwargs["num_outputs"],
        }

    monkeypatch.setattr(
        "qgan_v2.visualization.plot_generated_output",
        plot_output,
    )
    monkeypatch.setattr("qgan_v2.analysis.workflow.display_rows", lambda rows: rows)
    monkeypatch.setattr(analysis, "_finish", lambda fig, stem: None)

    record = analysis.render_randomness_output(
        filters={"preset": "base", "randomness": 0.5, "seed": 2},
        random_seed=None,
        output_count=4,
    )

    assert plotted == {
        "config_file": run.path / "config.yaml",
        "parameter_set": "best",
        "random_seed": None,
        "num_outputs": 4,
    }
    assert record["training_seed"] == 2
    assert record["random_input_applied"] is True
    assert record["num_outputs"] == 4
