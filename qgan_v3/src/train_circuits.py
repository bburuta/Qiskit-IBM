from qiskit.circuit.library import real_amplitudes, efficient_su2
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit import qpy



#- Create quantum circuits -#

# Create random input generator
def generate_randomizer(n_qubits, randomness):
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


# Create generator
def generate_generator(n_qubits, circuit_type):
    if circuit_type == 1:
        qc = real_amplitudes(n_qubits,
                            reps=3, # Number of layers
                            parameter_prefix='θ_g',
                            name='Generator').decompose()
    
    else: 
        qc = efficient_su2(n_qubits,
                      entanglement="reverse_linear",
                      reps=1, # Number of layers
                      parameter_prefix='θ_g',
                      name='Generator').decompose()
    
    return qc


# Create discriminator
def generate_discriminator(n_qubits, circuit_type):
    if circuit_type == 1:
        qc = efficient_su2(n_qubits,
                        entanglement="reverse_linear",
                        reps=1, # Number of layers
                        parameter_prefix='θ_d',
                        name='Discriminator').decompose()

        param_index = qc.num_parameters
        for i in reversed(range(n_qubits - 1)):
            qc.cx(i, n_qubits - 1)
        #qc.rx(disc_weights[param_index], N_QUBITS-1); param_index += 1
        qc.ry(Parameter("θ_d["+str(param_index)+"]"), n_qubits-1); param_index += 1
        qc.rz(Parameter("θ_d["+str(param_index)+"]"), n_qubits-1); param_index += 1
    
    else:
        qc = real_amplitudes(n_qubits,
                            reps=3, # Number of layers
                            parameter_prefix='θ_d',
                            name='Discriminator').decompose()
        
        param_index = qc.num_parameters
        for i in reversed(range(n_qubits - 1)):
            qc.cx(i, n_qubits - 1)
        qc.ry(Parameter("θ_d["+str(param_index)+"]"), n_qubits-1)
    
    return qc


# Create circuits file
def create_train_circuits_file(filename):
    randomizer_circuit = generate_randomizer()
    generator_circuit = generate_generator()
    discriminator_circuit = generate_discriminator()

    with open(filename, 'wb') as fd:
        qpy.dump([randomizer_circuit, generator_circuit, discriminator_circuit], fd)

    print("Circuits file created.")


# Load circuits from file
def load_train_circuits_file(config, filename):
    if config['data_management']['create_circuits'] or not os.path.exists(filename):
        create_train_circuits_file(filename)

    with open(filename, 'rb') as fd:
        circuits = qpy.load(fd)

    randomizer_circuit = circuits[-3]
    generator_circuit = circuits[-2]
    discriminator_circuit = circuits[-1]

    return randomizer_circuit, generator_circuit, discriminator_circuit