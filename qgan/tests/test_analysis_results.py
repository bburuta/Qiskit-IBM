from pathlib import Path

import numpy as np
import pytest

from qgan_v2.analysis.results import (
    RunResult,
    classify_run_status,
    comparison_line_colors,
    deduplicate_simulator_runs,
    factor_sweep_groups,
    is_completed,
    plot_training_dynamics_comparison,
    run_summary,
    select_main_convergence_results,
    unique_values,
)


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


def test_main_convergence_deduplicates_cpu_and_gpu_copies():
    cpu = make_result("cpu", device="CPU")
    gpu = make_result("gpu", device="GPU")
    gpu.metadata["random_circuit"] = None  # Older rand0 configs omitted this unused field.

    assert deduplicate_simulator_runs([gpu, cpu]) == [cpu]
    assert select_main_convergence_results([gpu, cpu]) == [cpu]


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
