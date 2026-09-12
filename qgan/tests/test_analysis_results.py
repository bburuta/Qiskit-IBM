from pathlib import Path

import pytest

from qgan_v2.analysis.results import (
    RunResult,
    classify_run_status,
    deduplicate_simulator_runs,
    factor_sweep_groups,
    is_completed,
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
