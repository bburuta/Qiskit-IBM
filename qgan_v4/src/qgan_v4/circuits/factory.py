import os

from qiskit.circuit.library import real_amplitudes, efficient_su2
from qiskit.circuit import Parameter
from qiskit import qpy

from qgan_v4.circuits.encoding import create_real_circuits, create_randomizer_circuit
from qgan_v4.storage.paths import get_circuits_filename



#- Create quantum circuits -#

# Create generator
def generate_generator(n_qubits, circuit_type):
    if circuit_type == 'real_amplitudes':
        qc = real_amplitudes(n_qubits,
                            reps=3, # Number of layers
                            parameter_prefix='θ_g',
                            name='Generator').decompose()
    
    elif circuit_type == 'efficient_su2':
        qc = efficient_su2(n_qubits,
                      entanglement="reverse_linear",
                      reps=1, # Number of layers
                      parameter_prefix='θ_g',
                      name='Generator').decompose()
    else:
        raise ValueError(f"Unknown generator circuit: {circuit_type}")
    
    return qc


# Create discriminator
def generate_discriminator(n_qubits, circuit_type):
    if circuit_type == 'efficient_su2':
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
    
    elif circuit_type == 'real_amplitudes':
        qc = real_amplitudes(n_qubits,
                            reps=3, # Number of layers
                            parameter_prefix='θ_d',
                            name='Discriminator').decompose()
        
        param_index = qc.num_parameters
        for i in reversed(range(n_qubits - 1)):
            qc.cx(i, n_qubits - 1)
        qc.ry(Parameter("θ_d["+str(param_index)+"]"), n_qubits-1)
    else:
        raise ValueError(f"Unknown discriminator circuit: {circuit_type}")
    
    return qc



# Create circuits file
def create_circuits_file(circuits, filename):
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'wb') as fd:
        qpy.dump(circuits, fd)

    print("Circuits file created.")


# Load circuits from file
def load_train_circuits_file(filename):
    with open(filename, 'rb') as fd:
        circuits = qpy.load(fd)

    return circuits



#- Circuits management -#

# Get circuits
def get_circuits(config, save_file=False):
    n_qubits = config['experiment']['n_qubits']

    # Save circuits in file
    if save_file:
        filename = get_circuits_filename(config)

        # Reset circuits file
        if config['circuits']['reset'] or not os.path.exists(filename):
            generator_circuit = generate_generator(n_qubits, config['circuits']['generator'])
            discriminator_circuit = generate_discriminator(n_qubits, config['circuits']['discriminator'])
            randomizer_circuit = create_randomizer_circuit(config)
            real_circuits = create_real_circuits(config)
            
            create_circuits_file([generator_circuit, discriminator_circuit, randomizer_circuit, *real_circuits], filename)

        # Load backend file
        circuits = load_train_circuits_file(filename)

        generator_circuit = circuits[0]
        discriminator_circuit = circuits[1]
        randomizer_circuit = circuits[2]
        real_circuits = circuits[3:]
    else:
        generator_circuit = generate_generator(n_qubits, config['circuits']['generator'])
        discriminator_circuit = generate_discriminator(n_qubits, config['circuits']['discriminator'])
        randomizer_circuit = create_randomizer_circuit(config)
        real_circuits = create_real_circuits(config)

    return [generator_circuit, discriminator_circuit, randomizer_circuit, real_circuits]
