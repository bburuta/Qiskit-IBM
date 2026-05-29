import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector



#- Amplitude embedding -#

# Transform images to probability distributions
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


# Transform images to amplitudes
def images_to_amp(images, intensity_power):
    probs = images_to_prob(images, intensity_power)
    amplitudes = torch.sqrt(probs)

    return amplitudes


# Base mode. Get specific probability distribution quantum circuit
def generate_prob_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Real circuit")
    qc.h(range(n_qubits-1))
    qc.cx(n_qubits-2, n_qubits-1)
    return [qc]


# Amplitude embedding. Get quantum circuits from amplitudes
def generate_amp_circuits(n_qubits, X_amplitudes = None):
    qcs = []
    for amplitudes in X_amplitudes:
        qc = QuantumCircuit(n_qubits, name="Real circuit")
        qc.prepare_state(state=amplitudes.detach().cpu().numpy(),
                        qubits=qc.qubits,
                        normalize=False)
        qcs.append(qc)
    return qcs
    


#- Angle embedding -#

# Angle embedding. Get variational quantum circuit for angle embedding
def generate_real_circuit(n_qubits):
    real_weights = ParameterVector('θ_r', n_qubits)
    qc = QuantumCircuit(n_qubits, name="Real circuit")
    param_index = 0

    for q in range(n_qubits):
        qc.ry(real_weights[param_index], q); param_index += 1

    return qc



_, X_amplitudes = images_to_amp(matrices, config['embedding_options']['contrast'],)

    if len(real_circuits) != len(X): # TODO poner otro if a 'base' == 1 o sino hacer (sistema hashes no porq backend y training data queremos mantener) que dataset y circuits vayan de la mano, pa ang / amp sera mejor tambien
        raise ValueError("Number of real circuits and number of real samples do not match.")