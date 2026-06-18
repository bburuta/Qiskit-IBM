import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector



#- Quantum circuit dataset -#

# Build specific distribution circuit
def create_specific_distribution_circuits(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Real circuit")
    qc.h(range(n_qubits-1))
    qc.cx(n_qubits-2, n_qubits-1)

    return [qc]


# Get probability distributions from quantum circuits
def circuits_to_probs(circuits):
    probs = []
    for circuit in circuits:
        probs.append(Statevector(circuit).probabilities())

    return np.array(probs)



#- Quantum circuit management -#

# Create quantum dataset circuits depending on source type
def create_quantum_dataset_circuits(config):
    source = config['dataset']['source']
    n_qubits = config['experiment']['n_qubits']

    if source == 'specific_distribution':
        return create_specific_distribution_circuits(n_qubits)
    else:
        raise ValueError(f"Unknown quantum dataset source: {source}")

# TODO show circuits and probability distributions