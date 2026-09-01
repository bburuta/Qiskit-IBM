import torch
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import random_statevector

from qgan_v2.datasets.images import get_images_dataset
from qgan_v2.datasets.quantum import create_quantum_dataset_circuits



#- Amplitude encoding -#

# Amplitude encoding. Get quantum circuits from amplitudes
def generate_amp_circuits(n_qubits, X_amplitudes):
    qcs = []
    for amplitudes in X_amplitudes:
        qc = QuantumCircuit(n_qubits, name="Real circuit")
        qc.prepare_state(state=amplitudes.detach().cpu().numpy(),
                        qubits=qc.qubits,
                        normalize=False)
        qcs.append(qc)
    return qcs
    

# Create RY random input generator
def generate_ry_randomizer(n_qubits):
    random_weights = ParameterVector('θ_r', n_qubits)
    qc = QuantumCircuit(n_qubits, name="Randomizer")

    for q in range(n_qubits):
        qc.ry(random_weights[q], q)

    return qc


# Create EfficientSU2 random input generator
def generate_efficient_su2_randomizer(n_qubits):
    return efficient_su2(
        n_qubits,
        entanglement="reverse_linear",
        reps=2,
        parameter_prefix='θ_r',
        name='Randomizer',
    ).decompose()


# Create a fixed random statevector input circuit at a controlled distance from |0...0>
def generate_statevector_randomizer(n_qubits, randomness, seed):
    if randomness < 0 or randomness > 1:
        raise ValueError("Statevector randomizer randomness must be in [0, 1].")

    dimension = 2 ** n_qubits
    statevector = random_statevector(dimension, seed=seed).data

    random_direction = statevector.copy()
    random_direction[0] = 0
    direction_norm = np.linalg.norm(random_direction)
    if direction_norm == 0:
        random_direction[1] = 1
        direction_norm = 1

    random_direction = random_direction / direction_norm
    angle = randomness * np.pi / 2
    interpolated_state = np.zeros(dimension, dtype=complex)
    interpolated_state[0] = np.cos(angle)
    interpolated_state += np.sin(angle) * random_direction

    qc = QuantumCircuit(n_qubits, name="Randomizer")
    qc.prepare_state(
        state=interpolated_state,
        qubits=qc.qubits,
        normalize=False,
    )

    return qc


# Create random input generator for amplitude-style encodings
def generate_amp_randomizer(n_qubits, random_circuit, randomness, seed):
    if random_circuit == 0:
        return QuantumCircuit(n_qubits, name="Randomizer")
    if random_circuit == 1:
        return generate_ry_randomizer(n_qubits)
    if random_circuit == 2:
        return generate_efficient_su2_randomizer(n_qubits)
    if random_circuit == 3:
        return generate_statevector_randomizer(n_qubits, randomness, seed)

    raise ValueError(f"Unknown random circuit type: {random_circuit}")



#- Angle encoding -#

# Angle encoding. Get variational quantum circuit for angle encoding
def generate_ang_circuit(n_qubits):
    real_weights = ParameterVector('θ_r', n_qubits)
    qc = QuantumCircuit(n_qubits, name="Real circuit")
    param_index = 0

    for q in range(n_qubits):
        qc.ry(real_weights[param_index], q); param_index += 1

    return qc


# Create angle random input generator
def generate_ang_randomizer(n_qubits, random_circuit):
    if random_circuit != 0:
        return generate_ang_circuit(n_qubits)
    return QuantumCircuit(n_qubits, name="Randomizer")



#- Create encoding circuits -#

# Transform images torch matrices to probability distributions
def images_to_prob(images, intensity_power):
    x = images.flatten(start_dim=1)

    if torch.any(x < 0):
        raise ValueError("Pixel intensities must be non-negative.")

    weights = x ** intensity_power

    totals = weights.sum(dim=1, keepdim=True)
    if torch.any(totals == 0):
        raise ValueError("At least one image is all zero.")

    probs = weights / totals

    return probs


# Transform images torch matrices to amplitudes
def images_to_amp(images, intensity_power):
    probs = images_to_prob(images, intensity_power)
    amplitudes = torch.sqrt(probs)

    return amplitudes


# Create real circuits depending on dataset and encoding type
def create_real_circuits(config):
    dataset_type = config['dataset']['type']
    encoding = config['encoding']['type']
    n_qubits = config['experiment']['n_qubits']

    if dataset_type == 'quantum':
        if encoding == 'direct_circuit':
            real_circuits = create_quantum_dataset_circuits(config)
        else:
            raise ValueError(f"Encoding incompatible with {dataset_type} datasets: {encoding}")
        
    elif dataset_type == 'classical':   
        if encoding == 'angle':
            real_circuits = [generate_ang_circuit(n_qubits)]
        elif encoding == 'amplitude':
            X = torch.as_tensor(get_images_dataset(config))
            X_amplitudes = images_to_amp(X, config['encoding']['contrast'])
            real_circuits = generate_amp_circuits(n_qubits, X_amplitudes)
        else:
            raise ValueError(f"Encoding incompatible with {dataset_type} datasets: {encoding}")
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return real_circuits


# Create real circuits depending on dataset and encoding type
def create_randomizer_circuit(config):
    encoding = config['encoding']['type']
    n_qubits = config['experiment']['n_qubits']
    randomness = config['encoding']['randomness']
    random_circuit = 0 if randomness == 0 else config['encoding']['random_circuit']
    seed = config['run']['seed']

    if encoding in ['direct_circuit', "amplitude"]:
        randomizer_circuit = generate_amp_randomizer(n_qubits, random_circuit, randomness, seed)
    elif encoding == 'angle':
        randomizer_circuit = generate_ang_randomizer(n_qubits, random_circuit)
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")
    
    return randomizer_circuit
