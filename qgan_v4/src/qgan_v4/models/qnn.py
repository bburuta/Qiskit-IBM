from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.library import SaveProbabilities
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN



#- QNN circuits -#

# Compose circuits
def compose_circuits(*circuits):
    n_qubits = circuits[0].num_qubits
    qc = QuantumCircuit(n_qubits)

    for circuit in circuits:
        qc.compose(circuit, inplace=True)

    return qc


# Get composed circuits
def get_composed_circuits(generator_circuit, discriminator_circuit, randomizer_circuit, real_circuits):
    # Connect real data and discriminator
    real_disc_circuits = []
    for real_circuit in real_circuits:
        real_disc_circuit = compose_circuits(real_circuit, discriminator_circuit)
        real_disc_circuits.append(real_disc_circuit)

    # Connect random input and generator
    ran_gen_circuit = compose_circuits(randomizer_circuit, generator_circuit)

    # Connect generator and discriminator
    gen_disc_circuit = compose_circuits(ran_gen_circuit, discriminator_circuit)

    return ran_gen_circuit, gen_disc_circuit, real_disc_circuits


# Set up evaluation circuits for angle encoding
def prepare_eval_circuit_ang(ran_gen_circuit):
    ran_gen_circuit.measure_all()
    return ran_gen_circuit


# Set up evaluation circuits for amplitude encoding
def prepare_eval_circuit_amp(ran_gen_circuit):
    ran_gen_circuit.append(SaveProbabilities(ran_gen_circuit.num_qubits), ran_gen_circuit.qubits)
    return ran_gen_circuit



#- QNN methods -#

# Gradient computation method
def get_gradient_method(gradient_method, estimator, seed):
    if gradient_method == 'SPSA':
        return SPSAEstimatorGradient(estimator=estimator, seed=seed)
    elif gradient_method == 'PSR':
        return ParamShiftEstimatorGradient(estimator=estimator)
    elif gradient_method == 'REG':
        return ReverseEstimatorGradient()
    else:
        raise ValueError(f"Unknown gradient method: {gradient_method}")


# Get observables
def get_observables(n_qubits):
    # Observables
    last_qubit = [("Z" + "I" * (n_qubits - 1), 1.0)]
    obs_train = SparsePauliOp.from_list(last_qubit)

    all_qubits = [
        ("I" * i + "Z" + "I" * (n_qubits - 1 - i), 1.0)
        for i in range(n_qubits)
    ]
    obs_eval = SparsePauliOp.from_list(all_qubits)

    return obs_train, obs_eval



#- QNN transpilation -#

# Train circuits and observables transpilation
def transpile_train_circuits(gen_disc_circuit, real_disc_circuits, obs, pm):
    transpiled_circuits = pm.run([gen_disc_circuit, *real_disc_circuits])
    gen_disc_circuit_transpiled = transpiled_circuits[0]
    real_disc_circuits_transpiled = transpiled_circuits[1:]

    obs_real_disc = [
        obs.apply_layout(real_disc_circuit_transpiled.layout)
        for real_disc_circuit_transpiled in real_disc_circuits_transpiled
    ]
    obs_gen_disc = [obs.apply_layout(gen_disc_circuit_transpiled.layout)]

    return real_disc_circuits_transpiled, gen_disc_circuit_transpiled, obs_real_disc, obs_gen_disc


# Evaluation circuits and observables transpilation
def transpile_eval_circuits(ran_gen_circuit, obs, eval_pm):
    gen_eval_circuit_transpiled = eval_pm.run(ran_gen_circuit)
    obs_gen_eval = obs.apply_layout(gen_eval_circuit_transpiled.layout)

    return gen_eval_circuit_transpiled, obs_gen_eval



#- QNN creation -#

# Get parameters by prefix
def split_params_by_prefix(params):
    disc_params = [param for param in params if param.name.startswith("θ_d")]
    gen_params = [param for param in params if param.name.startswith("θ_g")]
    other_params = [
        param for param in params
        if not param.name.startswith("θ_d") and not param.name.startswith("θ_g")
    ]
    return disc_params, gen_params, other_params


# Create train QNNs
def generate_train_qnns(real_disc_circuits_transpiled, gen_disc_circuit_transpiled, obs_real_disc, obs_gen_disc, gradient, estimator):
    gen_disc_params = list(gen_disc_circuit_transpiled.parameters)
    disc_params, gen_params, random_params = split_params_by_prefix(gen_disc_params)
    training_precision = estimator.options.default_precision

    # Specify QNN to update generator parameters
    gen_qnn = EstimatorQNN(
        circuit=gen_disc_circuit_transpiled,
        input_params=disc_params + random_params, # fixed parameters (discriminator + random parameters)
        weight_params=gen_params, # parameters to update (generator parameters)
        estimator=estimator,
        observables=obs_gen_disc,
        gradient=gradient,
        default_precision=training_precision,
    )

    # Specify QNN to update discriminator parameters regarding to fake data
    disc_fake_qnn = EstimatorQNN(
        circuit=gen_disc_circuit_transpiled,
        input_params=gen_params + random_params, # fixed parameters (generator + random parameters)
        weight_params=disc_params, # parameters to update (discriminator parameters)
        estimator=estimator,
        observables=obs_gen_disc,
        gradient=gradient,
        default_precision=training_precision,
    )

    # Specify QNN to update discriminator parameters regarding to real data
    disc_real_qnns = []
    for real_disc_circuit_transpiled, obs in zip(real_disc_circuits_transpiled, obs_real_disc):
        real_disc_params = list(real_disc_circuit_transpiled.parameters)
        real_disc_weights, _, real_inputs = split_params_by_prefix(real_disc_params)
        disc_real_qnn = EstimatorQNN(
            circuit=real_disc_circuit_transpiled,
            input_params=real_inputs, # fixed parameters (real data parameters)
            weight_params=real_disc_weights, # parameters to update (discriminator parameters)
            estimator=estimator,
            observables=obs,
            gradient=gradient,
            default_precision=training_precision,
        )
        disc_real_qnns.append(disc_real_qnn)

    return gen_qnn, disc_fake_qnn, disc_real_qnns


# Create evaluation QNN for angle encoding
def generate_eval_qnn_ang(gen_eval_circuit_transpiled, obs_gen_eval, gradient, eval_estimator):
    gen_eval_params = list(gen_eval_circuit_transpiled.parameters)
    _, gen_params, random_params = split_params_by_prefix(gen_eval_params)
    eval_precision = eval_estimator.options.default_precision

    gen_eval_qnn = EstimatorQNN(
        circuit=gen_eval_circuit_transpiled,
        input_params=random_params, # fixed parameters (random parameters)
        weight_params=gen_params, # parameters to update (generator parameters)
        estimator=eval_estimator,
        observables=obs_gen_eval,
        gradient=gradient,
        default_precision=eval_precision,
    )

    return gen_eval_qnn
