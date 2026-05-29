import torch
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# --- 1. High-Performance Backend Configuration ---
# Targeting GPU with single precision and cuQuantum optimization.
# 'batched_shots_gpu' enables parallel execution of the batch on the GPU.
backend_options = {
    'method': 'statevector',
    'device': 'GPU',
    'precision': 'single',       # Significant speedup 
    'cuStateVec_enable': True,   # NVIDIA library optimization [9]
    'batched_shots_gpu': True,   # Parallelize batch on GPU [9]
    'blocking_enable': False     # Disable chunking; simulation fits in VRAM 
}
backend = AerSimulator(**backend_options)

# --- 2. Algorithmic Optimization: Adjoint Gradient ---
# Replaces O(M) Parameter Shift with O(1) Adjoint Differentiation 
gradient = ReverseEstimatorGradient(estimator=backend)

# --- 3. QNN Definition ---
# The EstimatorQNN now uses the optimized backend and gradient method.
# Ensure 'input_params' and 'weight_params' are defined as per your circuit.
qnn_g = EstimatorQNN(
    circuit=generator_circuit,
    estimator=backend,
    gradient=gradient, 
    input_params=input_params,   # Assuming these are defined
    weight_params=weight_params
)
# TorchConnector handles the integration with PyTorch's autograd
model_g = TorchConnector(qnn_g)

# --- 4. Pre-compilation for KL Divergence Tracking ---
# Instead of instantiating Statevector() every loop, we pre-transpile
# a circuit that explicitly saves probabilities.
prob_circuit = generator_circuit.copy()
prob_circuit.save_probabilities() # GPU-native instruction [8]
transpiled_prob_circ = transpile(prob_circuit, backend)

# --- 5. Optimized Training Loop ---
for epoch in range(current_epoch, train_config['max_iterations']+1):

    # --- Quantum Discriminator Updates ---
    for disc_train_step in range(D_STEPS):
        # Optimization: Ensure closure_d handles batching correctly
        disc_loss = optimizer_d.step(closure_d)
        if (disc_train_step == D_STEPS-1):
            dloss[epoch] = disc_loss.detach().cpu().numpy()

    # --- Quantum Generator Updates ---
    for gen_train_step in range(G_STEPS):
        # The optimizer uses ReverseEstimatorGradient implicitly via model_g
        gen_loss = optimizer_g.step(closure_g)
        if (gen_train_step == G_STEPS-1):
            gloss[epoch] = gen_loss.detach().cpu().numpy()

    # --- Optimized KL Tracking ---
    # We avoid creating a new simulation instance. We run the pre-transpiled
    # circuit on the persistent GPU backend.
    with torch.no_grad():
        # Get current weights efficiently
        gen_params_tensor = torch.nn.utils.parameters_to_vector(model_g.parameters())
        gen_params_np = gen_params_tensor.cpu().numpy()
        
        # Execute on GPU backend; retrieve only small probability vector
        # parameter_binds maps the circuit parameters to the current numpy weights
        job = backend.run(
            transpiled_prob_circ, 
            parameter_binds=[{p: v for p, v in zip(prob_circuit.parameters, gen_params_np)}]
        )
        result = job.result()
        
        # Data transfer is minimal: 2^N floats instead of complex statevector
        probs_np = result.data(0)['probabilities'] 
        
        # Compute KL using PyTorch (can be done on GPU if tensors are moved there)
        gen_distribution_tensor = torch.from_numpy(probs_np)
        current_kl = torch.nn.functional.kl_div(
            input=gen_distribution_tensor.log(), 
            target=real_distribution_tensor, 
            reduction='sum'
        ).item()
        
        kl_div[epoch] = current_kl
        
        if min_kl_div > current_kl:
            min_kl_div = current_kl
            best_gen_params = gen_params_np