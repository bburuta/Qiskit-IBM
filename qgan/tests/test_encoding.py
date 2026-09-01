import pytest

pytest.importorskip("torch")
pytest.importorskip("qiskit")

import numpy as np
from qiskit.quantum_info import Statevector

from qgan_v2.circuits.encoding import (
    create_randomizer_circuit,
    generate_amp_randomizer,
    generate_ang_randomizer,
)


def test_amp_randomizer_mode_0_has_no_parameters_or_gates():
    circuit = generate_amp_randomizer(3, 0, randomness=0.1, seed=0)

    assert circuit.num_parameters == 0
    assert len(circuit.data) == 0


def test_amp_randomizer_mode_1_uses_one_ry_parameter_per_qubit():
    circuit = generate_amp_randomizer(3, 1, randomness=0.1, seed=0)

    assert circuit.num_parameters == 3
    assert circuit.count_ops()["ry"] == 3


def test_amp_randomizer_mode_2_uses_efficient_su2_with_two_repetitions():
    circuit = generate_amp_randomizer(3, 2, randomness=0.1, seed=0)

    assert circuit.num_parameters > 3
    assert circuit.count_ops()["ry"] > 0
    assert circuit.count_ops()["rz"] > 0


def test_amp_randomizer_mode_3_prepares_fixed_random_statevector():
    circuit = generate_amp_randomizer(3, 3, randomness=1, seed=0)

    assert circuit.num_parameters == 0
    assert len(circuit.data) > 0


@pytest.mark.parametrize(
    ("randomness", "expected_fidelity"),
    [
        (0, 1),
        (0.5, 0.5),
        (1, 0),
    ],
)
def test_amp_randomizer_mode_3_controls_distance_from_zero_state(randomness, expected_fidelity):
    circuit = generate_amp_randomizer(3, 3, randomness=randomness, seed=0)

    statevector = Statevector.from_instruction(circuit).data
    fidelity_to_zero_state = abs(statevector[0]) ** 2

    assert np.isclose(fidelity_to_zero_state, expected_fidelity)
    assert np.isclose(np.linalg.norm(statevector), 1)


def test_amp_randomizer_mode_3_rejects_randomness_outside_unit_interval():
    with pytest.raises(ValueError, match="randomness must be in \\[0, 1\\]"):
        generate_amp_randomizer(3, 3, randomness=1.1, seed=0)


def test_ang_randomizer_mode_0_has_no_parameters_or_gates():
    circuit = generate_ang_randomizer(3, 0)

    assert circuit.num_parameters == 0
    assert len(circuit.data) == 0


def test_ang_randomizer_nonzero_modes_use_angle_ry_randomizer():
    circuit = generate_ang_randomizer(3, 2)

    assert circuit.num_parameters == 3
    assert circuit.count_ops()["ry"] == 3


def test_create_randomizer_circuit_disables_randomizer_when_randomness_is_zero():
    config = {
        "encoding": {
            "type": "amplitude",
            "randomness": 0,
            "random_circuit": 2,
        },
        "experiment": {"n_qubits": 3},
        "run": {"seed": 0},
    }

    circuit = create_randomizer_circuit(config)

    assert circuit.num_parameters == 0
    assert len(circuit.data) == 0


def test_create_randomizer_circuit_uses_configured_amp_mode_when_randomness_is_nonzero():
    config = {
        "encoding": {
            "type": "amplitude",
            "randomness": 0.1,
            "random_circuit": 2,
        },
        "experiment": {"n_qubits": 3},
        "run": {"seed": 0},
    }

    circuit = create_randomizer_circuit(config)

    assert circuit.num_parameters > 3
