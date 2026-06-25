import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient

from qgan_v4.models.qnn import compose_circuits, split_params_by_prefix


#- Packed circuit creation -#

# Repeat one parameterized circuit while sharing its trainable parameters
def pack_repeated_circuit(template, name, batch_size):
    packed = QuantumCircuit(template.num_qubits * batch_size, name=f"packed_{name}")

    for copy_index in range(batch_size):
        qubits = range(copy_index * template.num_qubits, (copy_index + 1) * template.num_qubits)
        packed.compose(template, qubits=qubits, inplace=True)

    return packed


# Repeat an input circuit with independent parameters for every copy
def pack_input_circuit(template, name, batch_size):
    packed = QuantumCircuit(template.num_qubits * batch_size, name=f"packed_{name}")
    template_params = list(template.parameters)
    copy_params = []

    for copy_index in range(batch_size):
        params = ParameterVector(f"{name}_{copy_index}", len(template_params))
        renamed = template.assign_parameters(dict(zip(template_params, params)), inplace=False)
        qubits = range(copy_index * template.num_qubits, (copy_index + 1) * template.num_qubits)
        packed.compose(renamed, qubits=qubits, inplace=True)
        copy_params.append(list(params))

    return packed, copy_params


# Pack selected fixed circuits without adding input parameters
def pack_fixed_circuits(circuits, indexes, name):
    batch_size = len(indexes)
    n_qubits = circuits[0].num_qubits
    packed = QuantumCircuit(n_qubits * batch_size, name=f"packed_{name}")

    for copy_index, circuit_index in enumerate(indexes):
        circuit = circuits[int(circuit_index)]
        qubits = range(copy_index * n_qubits, (copy_index + 1) * n_qubits)
        packed.compose(circuit, qubits=qubits, inplace=True)

    return packed, [[] for _ in range(batch_size)]


# Compose packed input and trainable circuits on the same qubits
def join_packed_circuits(input_circuit, train_circuit, name):
    circuit = QuantumCircuit(input_circuit.num_qubits, name=f"joined_{name}")
    circuit.compose(input_circuit, inplace=True)
    circuit.compose(train_circuit, inplace=True)
    return circuit


# Place real and fake discriminator branches on separate qubits
def join_disc_circuits(real_circuit, fake_circuit, name):
    n_real_qubits = real_circuit.num_qubits
    circuit = QuantumCircuit(
        n_real_qubits + fake_circuit.num_qubits,
        name=f"joined_{name}",
    )
    circuit.compose(real_circuit, qubits=range(n_real_qubits), inplace=True)
    circuit.compose(
        fake_circuit,
        qubits=range(n_real_qubits, circuit.num_qubits),
        inplace=True,
    )
    return circuit


# Create the packed randomizer, generator and discriminator circuit
def create_fake_circuit(randomizer, generator, discriminator, batch_size):
    gen_disc = compose_circuits(generator, discriminator)
    repeated_gen_disc = pack_repeated_circuit(gen_disc, "gen_disc", batch_size)
    random_circuit, random_params = pack_input_circuit(randomizer, "random", batch_size)
    circuit = join_packed_circuits(random_circuit, repeated_gen_disc, "fake")
    return circuit, random_params


# Create the packed angle-encoded real discriminator circuit
def create_angle_real_circuit(real_circuit, discriminator, batch_size):
    repeated_disc = pack_repeated_circuit(discriminator, "disc", batch_size)
    real_input_circuit, real_params = pack_input_circuit(real_circuit, "real", batch_size)
    circuit = join_packed_circuits(real_input_circuit, repeated_disc, "real")
    return circuit, real_params


# Create the packed fixed real discriminator circuit
def create_fixed_real_circuit(real_circuits, real_indexes, discriminator):
    batch_size = len(real_indexes)
    repeated_disc = pack_repeated_circuit(discriminator, "disc", batch_size)
    fixed_circuit, fixed_params = pack_fixed_circuits(real_circuits, real_indexes, "real")
    circuit = join_packed_circuits(fixed_circuit, repeated_disc, "real")
    return circuit, fixed_params


# Join one fixed real branch and one fake branch for direct-circuit training
def create_direct_disc_circuit(randomizer, generator, discriminator, real_circuit):
    fake_circuit, random_params = create_fake_circuit(
        randomizer,
        generator,
        discriminator,
        batch_size=1,
    )
    real_disc_circuit, _ = create_fixed_real_circuit(
        [real_circuit],
        [0],
        discriminator,
    )
    circuit = join_disc_circuits(real_disc_circuit, fake_circuit, "direct_disc")
    return circuit, random_params


# Join half real and half fake branches for angle discriminator training
def create_angle_disc_circuit(
    randomizer,
    generator,
    discriminator,
    real_circuit,
    batch_size,
):
    half_batch = batch_size // 2
    real_disc_circuit, real_params = create_angle_real_circuit(
        real_circuit,
        discriminator,
        half_batch,
    )
    fake_disc_circuit, random_params = create_fake_circuit(
        randomizer,
        generator,
        discriminator,
        half_batch,
    )
    circuit = join_disc_circuits(real_disc_circuit, fake_disc_circuit, "angle_disc")
    return circuit, real_params + random_params


#- Observables -#

# Create one output-qubit Z observable for every packed copy
def get_observables(n_qubits, batch_size):
    return [
        SparsePauliOp.from_sparse_list(
            [("Z", [copy_index * n_qubits + n_qubits - 1], 1.0)],
            num_qubits=n_qubits * batch_size,
        )
        for copy_index in range(batch_size)
    ]


#- Transpilation -#

# Recover the physical qubits selected for every packed copy
def get_layout_groups(circuit, transpiled_circuit, n_qubits, batch_size):
    if (
        transpiled_circuit.layout is None
        or transpiled_circuit.layout.initial_layout is None
    ):
        return [
            list(range(copy_index * n_qubits, (copy_index + 1) * n_qubits))
            for copy_index in range(batch_size)
        ]

    virtual_to_physical = transpiled_circuit.layout.initial_layout.get_virtual_bits()
    return [
        [
            virtual_to_physical[circuit.qubits[copy_index * n_qubits + local_index]]
            for local_index in range(n_qubits)
        ]
        for copy_index in range(batch_size)
    ]


# Transpile one packed circuit and preserve its execution metadata
def transpile_packed_circuit(circuit, input_params, n_qubits, batch_size, pass_manager):
    transpiled_circuit = pass_manager.run(circuit)
    observables = [
        obs.apply_layout(transpiled_circuit.layout)
        for obs in get_observables(n_qubits, batch_size)
    ]
    disc_params, gen_params, _ = split_params_by_prefix(
        list(transpiled_circuit.parameters)
    )

    return {
        "circuit": transpiled_circuit,
        "observables": observables,
        "disc_params": disc_params,
        "gen_params": gen_params,
        "input_params": input_params,
        "batch_size": batch_size,
        "layout_groups": get_layout_groups(
            circuit,
            transpiled_circuit,
            n_qubits,
            batch_size,
        ),
    }


# Prepare the shared generator and fake discriminator job
def prepare_gen_job(randomizer, generator, discriminator, batch_size, pass_manager):
    circuit, input_params = create_fake_circuit(randomizer, generator, discriminator, batch_size)
    return transpile_packed_circuit(circuit, input_params, randomizer.num_qubits, batch_size, pass_manager)


# Prepare the separate angle real discriminator job
def prepare_angle_real_job(real_circuit, discriminator, batch_size, pass_manager):
    circuit, input_params = create_angle_real_circuit(real_circuit, discriminator, batch_size)
    return transpile_packed_circuit(circuit, input_params, real_circuit.num_qubits, batch_size, pass_manager)


# Prepare a fixed real discriminator job for the selected circuit indexes
def prepare_fixed_real_job(real_circuits, real_indexes, discriminator, pass_manager):
    circuit, input_params = create_fixed_real_circuit(real_circuits, real_indexes, discriminator)
    return transpile_packed_circuit(circuit, input_params, real_circuits[0].num_qubits, len(real_indexes), pass_manager)


# Prepare the joined direct-circuit discriminator job
def prepare_direct_disc_job(
    randomizer,
    generator,
    discriminator,
    real_circuit,
    pass_manager,
):
    circuit, input_params = create_direct_disc_circuit(
        randomizer,
        generator,
        discriminator,
        real_circuit,
    )
    return transpile_packed_circuit(
        circuit,
        input_params,
        randomizer.num_qubits,
        batch_size=2,
        pass_manager=pass_manager,
    )


# Prepare the joined angle discriminator job
def prepare_angle_disc_job(
    randomizer,
    generator,
    discriminator,
    real_circuit,
    batch_size,
    pass_manager,
):
    circuit, input_params = create_angle_disc_circuit(
        randomizer,
        generator,
        discriminator,
        real_circuit,
        batch_size,
    )
    return transpile_packed_circuit(
        circuit,
        input_params,
        randomizer.num_qubits,
        batch_size,
        pass_manager,
    )


#- Parameter values -#

# Build one parameter row in transpiled circuit order
def parameter_values(job, disc_values=(), gen_values=(), input_values=()):
    values = dict(zip(job["disc_params"], disc_values))
    values.update(zip(job["gen_params"], gen_values))

    if job["input_params"][0]:
        for copy_index, params in enumerate(job["input_params"]):
            values.update(zip(params, input_values[copy_index]))

    return np.asarray([[values[param] for param in job["circuit"].parameters]])


# Apply loss gradients to circuit parameter gradients
def combine_gradients(circuit_gradients, loss_gradients):
    circuit_gradients = np.asarray(circuit_gradients, dtype=float)
    loss_gradients = np.asarray(loss_gradients, dtype=float).reshape(-1)
    return np.tensordot(loss_gradients, circuit_gradients, axes=(0, 0))


#- Execution and gradients -#

# Execute all packed observables in one Estimator call
def packed_values(estimator, job, values):
    pub = (job["circuit"], job["observables"], values)
    result = estimator.run([pub]).result()
    return np.asarray(result[0].data.evs, dtype=float).reshape(-1)


# Compute one selected-parameter gradient for every packed observable
def packed_gradients(gradient, job, values, category):
    params = job[f"{category}_params"]
    result = gradient.run(
        [job["circuit"]] * job["batch_size"],
        job["observables"],
        [values[0]] * job["batch_size"],
        parameters=[params] * job["batch_size"],
    ).result()
    return [np.asarray(grad, dtype=float) for grad in result.gradients]


# Create the primitive gradient method used by runtime_packed
def get_gradient_method(config, estimator):
    method = config["experiment"]["gradient_method"]
    seed = config["run"]["seed"]

    if method == "SPSA":
        return SPSAEstimatorGradient(
            estimator=estimator,
            epsilon=0.01,
            seed=seed,
        )
    if method == "PSR":
        return ParamShiftEstimatorGradient(estimator=estimator)
    raise ValueError("runtime_packed supports gradient_method 'PSR' and 'SPSA'. Use qml_torch for 'REG'.")
