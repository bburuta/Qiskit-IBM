import torch

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qgan_v4.datasets.images import get_images_dataset
from qgan_v4.datasets.quantum import create_quantum_dataset_circuits



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
    

# Create random input generator
def generate_amp_randomizer(n_qubits, randomness):
    if randomness != 0:
        # # Almost fully random circuit, but expensive TODO
        # qc = efficient_su2(n_qubits,
        #                   entanglement="reverse_linear",
        #                   reps=3, # Number of layers
        #                   parameter_prefix='θ_r',
        #                   name='Randomizer').decompose()

        # Low randomness circuit, cheaper
        disc_weights = ParameterVector('θ_r', n_qubits)
        qc = QuantumCircuit(n_qubits, name="Randomizer")
        param_index = 0

        for q in range(n_qubits):
            qc.ry(disc_weights[param_index], q); param_index += 1
        
    else:
        qc = QuantumCircuit(n_qubits)
    
    return qc



#- Angle encoding -#

# Angle encoding. Get variational quantum circuit for angle encoding
def generate_ang_circuit(n_qubits):
    real_weights = ParameterVector('θ_r', n_qubits)
    qc = QuantumCircuit(n_qubits, name="Real circuit")
    param_index = 0

    for q in range(n_qubits):
        qc.ry(real_weights[param_index], q); param_index += 1

    return qc


# Create random input generator
def generate_ang_randomizer(n_qubits, randomness):
    if randomness != 0:
        qc = generate_ang_circuit(n_qubits)
    else:
        qc = QuantumCircuit(n_qubits)
    
    return qc



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

    if encoding in ['direct_circuit', "amplitude"]:
        randomizer_circuit = generate_amp_randomizer(n_qubits, randomness)
    elif encoding == 'angle':   
        randomizer_circuit = generate_ang_randomizer(n_qubits, randomness)
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")
    
    return randomizer_circuit
