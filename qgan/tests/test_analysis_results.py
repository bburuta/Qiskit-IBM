from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qgan_v2.analysis.results import (
    RunResult,
    aggregate_metric,
    classify_run_status,
    comparison_line_colors,
    deduplicate_simulator_runs,
    factor_sweep_groups,
    grouped_performance_table,
    is_completed,
    load_results,
    metadata_from_config,
    paired_metric_deltas,
    plot_training_dynamics_comparison,
    run_summary,
    select_main_learn_results,
    transform_metric_values,
    unique_values,
)
from qgan_v2.config.loader import save_config_file


def make_result(
    run_id,
    *,
    epochs=1000,
    seed=0,
    randomness=0,
    device="CPU",
    execution_type="noiseless",
    error=None,
    status="ok",
):
    metadata = {
        "run_id": run_id,
        "preset": "ang",
        "implementation": "qml_torch",
        "packing": "separate",
        "execution_type": execution_type,
        "gradient_method": "SPSA",
        "n_qubits": 4,
        "random_circuit": 1,
        "randomness": randomness,
        "batch_size": 4,
        "eval_batch_size": 8,
        "eval_method": "gradient",
        "learning_rate": 0.005,
        "max_iterations": 1000,
        "gen_iterations": 1,
        "disc_iterations": 1,
        "precision": 0.01,
        "noiseless_method": "statevector",
        "noisy_method": "density_matrix",
        "noisy_backend_mapping": "hardware",
        "seed": seed,
        "run_device": device,
        "simulator_device": device,
    }
    return RunResult(
        path=Path(run_id),
        config={},
        metadata=metadata,
        eval={epoch: float(epochs - epoch) for epoch in range(epochs)},
        gloss={},
        dloss={},
        times={epoch: 2.0 for epoch in range(epochs)},
        status=status,
        error=error,
    )


def test_completion_requires_the_full_requested_budget():
    assert is_completed(make_result("complete"))
    assert is_completed(make_result("continued", epochs=1001))
    assert not is_completed(make_result("partial", epochs=999))


def test_main_learn_deduplicates_cpu_and_gpu_copies():
    cpu = make_result("cpu", device="CPU")
    gpu = make_result("gpu", device="GPU")
    gpu.metadata["random_circuit"] = None  # Older rand0 configs omitted this unused field.

    assert deduplicate_simulator_runs([gpu, cpu]) == [cpu]
    assert select_main_learn_results([gpu, cpu]) == [cpu]


@pytest.mark.parametrize(
    "experiment, expected",
    [
        ({"implementation": "base"}, "base"),
        ({"preset": "ang"}, "ang"),
        ({"preset": "amp", "implementation": "base"}, "amp"),
    ],
)
def test_metadata_reads_legacy_presets_and_prefers_the_current_field(experiment, expected):
    assert metadata_from_config({"experiment": experiment})["preset"] == expected


def test_legacy_checkpoint_deduplicates_when_adjacent_yaml_fails_validation(
    monkeypatch, tmp_path,
):
    cpu_config = {
        "run": {"id": "cpu", "seed": 0, "device": "CPU"},
        "experiment": {
            "preset": "base",
            "execution_type": "noiseless",
            "n_qubits": 4,
            "gradient_method": "PSR",
        },
        "implementation": {"name": "qml_torch", "discriminator_packing": "separate"},
        "encoding": {"randomness": 0, "random_circuit": 1},
        "training": {"max_iterations": 1000},
        "backend": {"simulator": {"device": "CPU"}},
    }
    gpu_config = deepcopy(cpu_config)
    gpu_config["run"].update(id="gpu", device="GPU")
    gpu_config["backend"]["simulator"]["device"] = "GPU"
    gpu_config["experiment"]["implementation"] = gpu_config["experiment"].pop("preset")
    gpu_config["encoding"].pop("random_circuit")
    states = {}
    for config in (cpu_config, gpu_config):
        run_dir = tmp_path / config["run"]["id"]
        run_dir.mkdir()
        # These older configs lack fields required by current run validation.
        save_config_file(config, run_dir / "config.yaml")
        checkpoint = run_dir / "training_data.pth"
        checkpoint.touch()
        states[checkpoint] = SimpleNamespace(
            config=config,
            metrics=SimpleNamespace(
                eval={epoch: 1.0 for epoch in range(1000)},
                gloss={}, dloss={}, times={},
            ),
        )
    monkeypatch.setattr("qgan_v2.analysis.results._load_training_state", states.__getitem__)

    runs = load_results(tmp_path)

    assert len(runs) == 2
    assert all(run.status == "ok" and run.metadata["preset"] == "base" for run in runs)
    selected = select_main_learn_results(runs)
    assert [run.run_id for run in selected] == ["cpu"]
    assert gpu_config["experiment"] == {
        "implementation": "base", "execution_type": "noiseless",
        "n_qubits": 4, "gradient_method": "PSR",
    }


def test_factor_sweep_requires_every_level_for_every_seed():
    levels = (0, 0.1, 0.25, 0.5, 1)
    complete = [
        make_result(f"run-{level}-{seed}", randomness=level, seed=seed)
        for level in levels
        for seed in (0, 1, 2)
    ]

    groups = factor_sweep_groups(
        complete,
        factor="randomness",
        required_levels=levels,
    )
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == 15

    assert not factor_sweep_groups(
        complete[:-1],
        factor="randomness",
        required_levels=levels,
    )


def test_summary_uses_a_fixed_last_100_window_and_median_epoch_time():
    result = make_result("summary", epochs=1000)
    result.times[0] = 1000.0
    summary = run_summary(result)

    assert summary["last_window_median_eval"] == pytest.approx(50.5)
    assert summary["evaluation_step_volatility"] == pytest.approx(1.0)
    assert summary["median_time_per_epoch"] == pytest.approx(2.0)
    assert summary["projected_1000_epoch_time"] == pytest.approx(2000.0)
    assert summary["measured_epochs"] == 1000


def test_average_best_so_far_rewards_early_achievements_despite_regression():
    early = make_result("early", epochs=5)
    early.eval = {0: 1.0, 1: 0.2, 2: 0.8, 3: 0.9, 4: 0.1}
    late = make_result("late", epochs=5)
    late.eval = {0: 1.0, 1: 0.8, 2: 0.9, 3: 0.2, 4: 0.1}

    assert run_summary(early)["best_eval"] == run_summary(late)["best_eval"]
    assert run_summary(early)["average_best_so_far_eval"] == pytest.approx(0.34)
    assert run_summary(late)["average_best_so_far_eval"] == pytest.approx(0.58)


@pytest.mark.parametrize(
    "evaluations, expected",
    [
        ({3: 0.1, 0: 1.0, 2: 0.9, 1: 0.2}, 0.375),
        ({0: np.nan, 1: 1.0, 2: np.inf, 3: 0.2, 4: 0.9}, 1.4 / 3),
        ({0: 0.2}, 0.2),
        ({}, np.nan),
        ({0: np.nan, 1: np.inf, 2: -np.inf}, np.nan),
    ],
)
def test_average_best_so_far_uses_finite_evaluations_in_epoch_order(
    evaluations, expected
):
    result = make_result("summary")
    result.eval = evaluations
    value = run_summary(result)["average_best_so_far_eval"]
    if np.isnan(expected):
        assert np.isnan(value)
    else:
        assert value == pytest.approx(expected)


def test_average_best_so_far_is_aggregated_and_used_in_paired_differences():
    baseline = make_result("baseline", epochs=3, randomness=0)
    baseline.eval = {0: 1.0, 1: 0.2, 2: 0.9}
    treatment = make_result("treatment", epochs=3, randomness=1)
    treatment.eval = {0: 1.0, 1: 0.8, 2: 0.2}
    runs = [baseline, treatment]

    grouped = grouped_performance_table(runs, fields=("randomness",))
    assert grouped[0]["median_average_best_so_far_eval"] == pytest.approx(1.4 / 3)
    assert grouped[1]["median_average_best_so_far_eval"] == pytest.approx(2.0 / 3)
    assert "median_epoch_of_best_eval" not in grouped[0]
    deltas = paired_metric_deltas(
        runs,
        baseline_filters={"randomness": 0},
        treatment_filters={"randomness": 1},
        metric_name="average_best_so_far_eval",
        pair_fields=("seed",),
    )
    assert len(deltas) == 1
    assert deltas[0]["delta"] == pytest.approx(0.2)


def test_metric_value_transforms_are_per_run_and_leave_raw_values_unchanged():
    values = np.asarray([2.0, 4.0, 6.0])

    assert np.allclose(transform_metric_values(values, "none"), values)
    assert np.allclose(transform_metric_values(values, "normalize"), [0.0, 0.5, 1.0])
    assert np.allclose(
        transform_metric_values(values, "standardize"),
        (values - values.mean()) / values.std(),
    )
    assert np.allclose(values, [2.0, 4.0, 6.0])


def test_aggregate_metric_normalizes_each_run_before_seed_aggregation():
    first = make_result("first", epochs=3)
    second = make_result("second", epochs=3)
    first.eval = {0: 10.0, 1: 5.0, 2: 0.0}
    second.eval = {0: 100.0, 1: 50.0, 2: 0.0}

    aggregate = aggregate_metric([first, second], transform="normalize")

    assert np.allclose(aggregate["center"], [1.0, 0.5, 0.0])


def test_metric_value_transform_rejects_unknown_mode():
    with pytest.raises(ValueError, match="transform must be one of"):
        transform_metric_values([1.0], "scaled")


def test_failure_classification_recognizes_oom():
    failed = make_result(
        "oom",
        epochs=0,
        status="missing_training_data",
        error="RuntimeError: CUDA out of memory",
    )
    assert classify_run_status(failed) == "out_of_memory"


def test_numeric_categories_are_ordered_numerically():
    assert unique_values(
        [
            make_result("q16", randomness=16),
            make_result("q4", randomness=4),
            make_result("q8", randomness=8),
        ],
        "randomness",
    ) == [4, 8, 16]

    gradients = []
    for method in ("SPSA", "REG", "PSR"):
        result = make_result(method)
        result.metadata["gradient_method"] = method
        gradients.append(result)
    assert unique_values(gradients, "gradient_method") == ["PSR", "REG", "SPSA"]


def test_numeric_comparison_colors_follow_value_spacing():
    colors_module = pytest.importorskip("matplotlib.colors")
    levels = (0, 0.1, 0.25, 0.5, 1)
    colors = comparison_line_colors(levels, field="randomness")

    expected = [
        colors_module.to_rgb(value)
        for value in ("#E6A400", "#E65100", "#A60026", "#7626B8", "#174EA6")
    ]
    assert len(colors) == len(levels)
    assert np.allclose([color[:3] for color in colors], expected)


def test_qubit_comparison_colors_progress_from_yellow_to_red():
    colors_module = pytest.importorskip("matplotlib.colors")
    colors = comparison_line_colors((4, 8, 16), field="n_qubits")

    expected = [
        colors_module.to_rgb(value)
        for value in ("#E6A400", "#E65100", "#A60026")
    ]
    assert np.allclose([color[:3] for color in colors], expected)
    assert sum(colors[0][:3]) > sum(colors[-1][:3])


def test_qubit_comparison_colors_do_not_shift_when_q16_is_missing():
    colors_module = pytest.importorskip("matplotlib.colors")
    colors = comparison_line_colors((4, 8), field="n_qubits")

    expected = [
        colors_module.to_rgb(value)
        for value in ("#E6A400", "#E65100")
    ]
    assert np.allclose([color[:3] for color in colors], expected)


def test_multi_setting_dynamics_plot_keeps_losses_and_evaluation():
    pytest.importorskip("matplotlib")
    first = make_result("first", epochs=3)
    second = make_result("second", epochs=3)
    first.metadata["gradient_method"] = "PSR"
    second.metadata["gradient_method"] = "SPSA"
    first.gloss = second.gloss = {0: 1.0, 1: 0.5, 2: 0.25}
    first.dloss = second.dloss = {0: 0.2, 1: 0.4, 2: 0.8}

    figure, axes = plot_training_dynamics_comparison(
        [first, second], compare_by="gradient_method"
    )

    assert len(axes) == 2
    assert axes[0].get_ylabel() == "Adversarial loss"
    assert axes[1].get_ylabel() == "Evaluation score"
    assert {line.get_label() for line in axes[0].lines if not line.get_label().startswith("_")} == {
        "PSR — generator",
        "PSR — discriminator",
        "SPSA — generator",
        "SPSA — discriminator",
    }
    assert {line.get_label() for line in axes[1].lines if not line.get_label().startswith("_")} == {
        "PSR",
        "SPSA",
    }
    figure.clear()
