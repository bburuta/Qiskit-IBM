# %% [markdown]
# Version con TorchConnector, parametros guardados junto al state del modelo (pero para visualizar tambien se guardan aparte), gradientes actualizadas sin insercción a mano pero al necesitar unico modelo: discriminador fusionado en dos necesita el doble de qubits (necesario para entrenar distribuciones de probabilidad), step te devuelve el loss (no es necesario calcularlo en un paso extra)

# ## Introduction
# 
# The Quantum Generative Adversarial Network (QGAN) [[1]](https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-machine-learning/qgan.ipynb)  [[2]](https://arxiv.org/abs/1406.2661) we propose consists of two Quantum Neural Network (QNN) [[3]](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/01_neural_networks.html): a generator and a discriminator. The generator is responsible for creating synthetic data samples. The discriminator evaluates the authenticity of the created samples by distinguishing between real and generated data. Through an adversarial training process, both networks continuously improve, leading to the generation of increasingly realistic data. 
# This fully quantum approach benefits from the strengths of quantum state preparation and gradient calculation combined with classical optimizators [[4]](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam).
# The data used to train the QGAN in this implementation is a probability distributions.
# 
# This implementation uses aer_simulator_statevector.


# ## Implementation (statevector simulation)

# %% Instalation
#--- INSTALATION INSTRUCTIONS ---#

# For linux 64-bit systems,
#uname -a

# Conda quick installation
#mkdir -p ~/miniconda3
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
#bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
#rm ~/miniconda3/miniconda.sh

# Create enviroment with conda
#conda create -n myenv python=3.10
#conda activate myenv
#pip install qiskit==1.4.5 qiskit-machine-learning==0.8.4 'qiskit-machine-learning[sparse]' torch
# IMPORTANT: Make sure you are on 3.10
# May need to restart the kernel after instalation

#--- Imports ---#
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import random_statevector, Statevector, SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.primitives import StatevectorEstimator
from qiskit import qpy

from qiskit_machine_learning.neural_networks import EstimatorQNN # Downgrade to qiskit 1.x so is compatible with qiskit-machine-learning 0.8.2
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient
from qiskit_machine_learning.connectors import TorchConnector

import numpy as np
import torch
import time
import os


# %% Configuration
#- Configuration -#

# Training configuration dict
train_config = {
    'execution_type': "simulation",
    'n_qubits': 4,
    'seed': 1,
    'id': None, # For different circuits or training parameters
    'reset_data': 0,

    'create_circuits': False, # Create circuits manually or load from file
    'gradient_method': "PSR", # PSR or SPSA
    'max_iterations': 1000,
    'gen_iterations': 1, # !!!!!!!!!!!!!!!! PRUEBA DISTINTOS
    'disc_iterations': 1,
    'print_progress_iterations': 10,
    'training_data_file': None, # Automatically created with manage_files function
    'circuits_file': None # Automatically created with manage_files function
}

# File management
def manage_files(data_folder_name = 'data', implementation_name = 'fullyq_torchc', execution_type_name = 'sim', training_data_file_name = 'training_data', circuits_file_name = 'circuits'):
    data_folder = data_folder_name + '/' + implementation_name + '/' + execution_type_name + '/' + 'q' + str(train_config['n_qubits']) + '/' + 'seed' + str(train_config['seed']) + '/'
    if train_config['id'] is not None:
        data_folder = data_folder + '/' + str(train_config['id']) + '/' 
    training_data_file = data_folder + training_data_file_name + '.pth'
    circuits_file = data_folder + circuits_file_name + '.qpy'

    # Create folders if they do not exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    return training_data_file, circuits_file

train_config['training_data_file'], train_config['circuits_file'] = manage_files()



# %% Create QCs
#- Create quantum circuits -#

# Create real data sample circuit
def generate_real_circuit():
    n_qubits = train_config['n_qubits']

    # sv = random_statevector(2**N_QUBITS, seed=SEED)
    # qc = QuantumCircuit(N_QUBITS)
    # qc.prepare_state(sv, qc.qubits, normalize=True)

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits-1))
    qc.cx(n_qubits-2, n_qubits-1)
    return qc


# Create generator
def generate_generator():
    n_qubits = train_config['n_qubits']

    qc = RealAmplitudes(n_qubits,
                        reps=3, # Number of layers
                        parameter_prefix='θ_g',
                        name='Generator')
    
    return qc.decompose()


# Create discriminator
def generate_discriminator():
    n_qubits = train_config['n_qubits']

    qc = EfficientSU2(n_qubits,
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
    
    return qc


# Create quantum circuits
def create_circuits():
    real_circuit = generate_real_circuit()
    generator_circuit = generate_generator()
    discriminator_circuit = generate_discriminator()

    with open(train_config['circuits_file'], 'wb') as fd:
        qpy.dump([real_circuit, generator_circuit, discriminator_circuit], fd)

# Load circuits
if train_config['create_circuits']:
    create_circuits()

try:
    with open(train_config['circuits_file'], 'rb') as fd:
        circuits = qpy.load(fd)
except FileNotFoundError:
    print("Circuits file not found. Creating new circuits file.")
    create_circuits()
    with open(train_config['circuits_file'], 'rb') as fd:
        circuits = qpy.load(fd)
    
real_circuit = circuits[0]
generator_circuit = circuits[1]
discriminator_circuit = circuits[2]


# %% Set up models
#- Set up training quantum circuits -#
def generate_training_circuits(real_circuit, generator, discriminator):
    n_qubits = train_config['n_qubits']

    # Connect real circuit and discriminator
    real_disc_circuit = QuantumCircuit(n_qubits)
    real_disc_circuit.compose(real_circuit, inplace=True)
    real_disc_circuit.compose(discriminator, inplace=True)

    # Connect generator and discriminator
    gen_disc_circuit = QuantumCircuit(n_qubits)
    gen_disc_circuit.compose(generator.decompose(), inplace=True)
    gen_disc_circuit.compose(discriminator, inplace=True)

    # Combine both discriminator circuits
    real_gen_disc_circuit = QuantumCircuit(n_qubits*2)
    real_gen_disc_circuit.compose(gen_disc_circuit, qubits=range(0,n_qubits), inplace=True)
    real_gen_disc_circuit.compose(real_disc_circuit, qubits=range(n_qubits,(n_qubits*2)), inplace=True)

    # Use EstimatorQNN to compile the circuit and handle gradient calculation
    estimator = StatevectorEstimator()

    if train_config['gradient_method'] == 'SPSA':
        gradient = SPSAEstimatorGradient(estimator=estimator)
    else:
        gradient = ParamShiftEstimatorGradient(estimator=estimator)

    # specify QNN to update generator parameters
    H1 = SparsePauliOp.from_list([("Z" + "I"*(n_qubits-1), 1.0)])
    n_disc_params = discriminator.num_parameters

    gen_qnn = EstimatorQNN(circuit=gen_disc_circuit,
                        input_params=gen_disc_circuit.parameters[:n_disc_params], # fixed parameters (discriminator parameters)
                        weight_params=gen_disc_circuit.parameters[n_disc_params:], # parameters to update (generator parameters)
                        estimator=estimator,
                        observables=[H1],
                        gradient=gradient,
                        default_precision=0.0,
                        input_gradients = True
                        )


    # specify QNN to update discriminator parameters
    H21 = SparsePauliOp.from_list([("Z" + "I"*(n_qubits-1) + "I"*(n_qubits), 1.0)]) # First discriminator output
    H22 = SparsePauliOp.from_list([(("I"*(n_qubits) + "Z" + "I"*(n_qubits-1)), 1.0)]) # Second discriminator output

    disc_qnn = EstimatorQNN(circuit=real_gen_disc_circuit,
                            input_params=gen_disc_circuit.parameters[n_disc_params:],
                            weight_params=gen_disc_circuit.parameters[:n_disc_params], # parameters to update (discriminator parameters)
                            estimator=estimator,
                            observables=[H21, H22],
                            gradient=gradient,
                            default_precision=0.0,
                            input_gradients = True
                            )
    
    return gen_qnn, disc_qnn

gen_qnn, disc_qnn = generate_training_circuits(real_circuit, generator_circuit, discriminator_circuit)
    


# %% Restore states
#- Restore parameters and model states -#

# Reset all data training
def reset_data(gen_qnn, disc_qnn):
    np.random.seed(train_config['seed'])

    init_gen_params = np.random.uniform(low=-np.pi, high=np.pi, size=(gen_qnn.num_weights,))
    init_disc_params = np.random.uniform(low=-np.pi, high=np.pi, size=(disc_qnn.num_weights,))

    model_g = TorchConnector(gen_qnn, initial_weights = init_gen_params)
    model_d = TorchConnector(disc_qnn, initial_weights = init_disc_params)

    model_g.train() 
    model_d.train()

    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=0.01)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.01)


    torch.save({
        'init_gen_params': init_gen_params,
        'init_disc_params': init_disc_params,
        'gen_params': torch.nn.utils.parameters_to_vector(model_g.parameters()).detach().numpy(),
        'disc_params': torch.nn.utils.parameters_to_vector(model_d.parameters()).detach().numpy(),
        'best_gen_params': init_gen_params,
        'model_g_state': model_g.state_dict(),
        'model_d_state': model_d.state_dict(),
        'optimizer_g_state': optimizer_g.state_dict(),
        'optimizer_d_state': optimizer_d.state_dict(),
        'current_epoch': 0,
        "metrics": {
            "gloss": {},
            "dloss": {},
            "kl_div": {},
        },
        'random_state': np.random.get_state()
    }, train_config['training_data_file'])


# Load parameters and training states
if train_config['reset_data']:
    reset_data(gen_qnn, disc_qnn)

try:
    params = torch.load(train_config['training_data_file'], weights_only=False)
except FileNotFoundError:
    print("Training data file not found. Resetting parameters.")
    reset_data(gen_qnn, disc_qnn)
    params = torch.load(train_config['training_data_file'], weights_only=False)

np.random.set_state(params['random_state'])

model_g = TorchConnector(gen_qnn)
model_d = TorchConnector(disc_qnn)
optimizer_g = torch.optim.Adam(model_g.parameters())
optimizer_d = torch.optim.Adam(model_d.parameters())

model_g.load_state_dict(params['model_g_state'])
model_d.load_state_dict(params['model_d_state'])
optimizer_g.load_state_dict(params['optimizer_g_state'])
optimizer_d.load_state_dict(params['optimizer_d_state'])

f_loss = torch.nn.MSELoss(reduction="sum")

def closure_g():
    optimizer_g.zero_grad()  # Initialize/clear gradients
    disc_params = torch.nn.utils.parameters_to_vector(model_d.parameters()).detach() #disc_params = optimizer_d.param_groups[0]['params'][0].detach()
    loss = f_loss(model_g(disc_params), torch.ones(1)) # 1-> Good guess
    loss.backward()  # Backward pass
    return loss

def closure_d():
    optimizer_d.zero_grad()
    gen_params = torch.nn.utils.parameters_to_vector(model_g.parameters()).detach() #gen_params = optimizer_g.param_groups[0]['params'][0].detach()
    loss = f_loss(model_d(gen_params), torch.tensor([1.0, 0.0]))
    loss.backward()
    return loss


current_epoch = params['current_epoch']
gloss = params['metrics']['gloss']
gen_loss = list(gloss)[-1] if (gloss) else None
dloss = params['metrics']['dloss']
disc_loss = list(dloss)[-1] if (dloss) else None
kl_div = params['metrics']['kl_div']
min_kl_div = np.min(list(kl_div.values())) if (kl_div) else float('inf')



# %% Interrupter
#- Manage training interruption -#
import signal

# Class to manage training interruption
class Interrupter:
    def __init__(self):
        self.kill_now = False
        self.interrupt_count = 0

        # Intercept the Ctrl+C signal
        signal.signal(signal.SIGINT, self.handle_signal)
        # Intercept the termination signal (useful for Docker/systems)
        #signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, signum, frame):
        self.interrupt_count += 1
        
        if self.interrupt_count == 1:
            # First Press: Enable graceful exit
            self.kill_now = True
            print("\nInterrupter: Termination signal received. The loop will stop after the current iteration. (Press Ctrl+C again to force quit)")
        
        elif self.interrupt_count >= 2:
            # Second Press: Force quit immediately
            print("\nInterrupter: [!] Force quit triggered! Terminating immediately.")
            # Restore default signal handler to avoid recursion
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            # Raise the exception to stop execution right here
            raise KeyboardInterrupt
        


# %% Training
#- Training -#

D_STEPS = train_config['disc_iterations']
G_STEPS = train_config['gen_iterations']

real_distribution_tensor = torch.from_numpy(Statevector(real_circuit).probabilities()) # Retrieve real data probability distribution 

interrupter = Interrupter()

if train_config['print_progress_iterations']:
    TABLE_HEADERS = "Epoch | Generator cost | Discriminator cost | KL Div. | Best KL Div. | Time |"
    print(TABLE_HEADERS)
start_time = time.time()

#--- Training loop ---#
try: # In case of interruption
    for epoch in range(current_epoch, train_config['max_iterations']+1):

        #--- Quantum discriminator parameter updates ---#
        for disc_train_step in range(D_STEPS):
            # Calculate discriminator gradients and update parameters
            disc_loss = optimizer_d.step(closure_d)

            # Calculate discriminator cost
            dloss[epoch] = disc_loss

        #--- Quantum generator parameter updates ---#
        for gen_train_step in range(G_STEPS):
            # Calculate generator gradient and update parameters
            gen_loss = optimizer_g.step(closure_g)

            # Save generator cost
            gloss[epoch] = gen_loss

        #--- Track KL and save best performing generator weights ---#
        gen_params_np = torch.nn.utils.parameters_to_vector(model_g.parameters()).detach().numpy()
        gen_distribution_tensor = torch.from_numpy(Statevector(generator_circuit.assign_parameters(gen_params_np)).probabilities()) # Retrieve probability distribution of generator with current parameters.


        # 3. Move to GPU (The speedup for large vectors is massive)
        # if torch.cuda.is_available():
        #     p = p.cuda()
        #     q = q.cuda()

        # Performance measurement function: uses Kullback Leibler Divergence to measures the distance between two distributions
        current_kl = torch.nn.functional.kl_div(input=gen_distribution_tensor.log(), target=real_distribution_tensor, reduction='sum').numpy() # reduction="batchnoseque" pa batches
        kl_div[epoch] = current_kl
        if min_kl_div > current_kl:
            min_kl_div = current_kl
            best_gen_params = gen_params_np # New best

        #--- Print progress ---#
        if train_config['print_progress_iterations'] and (epoch % train_config['print_progress_iterations'] == 0):
            for header, val in zip(TABLE_HEADERS.split('|'),
                                (epoch, gen_loss, disc_loss, current_kl, min_kl_div, (time.time() - start_time))):
                print(f"{val:.3g} ".rjust(len(header)), end="|")
            start_time = time.time()
            print()

        # In case of interruption
        if interrupter.kill_now:
            print("Interrupter: Graceful exit triggered. Breaking loop.")
            break
            
#--- Save parameters and optimizer states data ---#
finally:
    torch.save({
        'init_gen_params': params['init_gen_params'],
        'init_disc_params': params['init_disc_params'],
        'best_gen_params': best_gen_params,
        'gen_params': torch.nn.utils.parameters_to_vector(model_g.parameters()),
        'disc_params': torch.nn.utils.parameters_to_vector(model_d.parameters()),
        'model_g_state': model_g.state_dict(),
        'model_d_state': model_d.state_dict(),
        'optimizer_g_state': optimizer_g.state_dict(),
        'optimizer_d_state': optimizer_d.state_dict(),
        'current_epoch': epoch+1,
        "metrics": {
            "gloss": gloss,
            "dloss": dloss,
            "kl_div": kl_div,
        },
        'random_state': np.random.get_state()
    }, train_config['training_data_file'])
    
    kl_div_data = list(kl_div.values())
    print("Training complete:", "\n   Data path:", train_config['training_data_file'], "\n   Best KLDiv:", np.min(kl_div_data), "in epoch", np.argmin(kl_div_data), "\n   Improvement:", kl_div_data[0]-np.min(kl_div_data))