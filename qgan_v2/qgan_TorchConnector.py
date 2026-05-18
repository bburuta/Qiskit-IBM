# %% [markdown]
# Resumen: 
# - Implementacion:
#     - Training: noiseless, noisy y real
#     - Evaluation: noiseless y noisy
# - Simulación via qiskit_aer (en GPUs)
# - Optimicación pytorch para evaluacion en multiples GPUs

# %% [markdown]
# ## Implementation: Probability Distributions with Torch Connector

# %%
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
#pip install qiskit qiskit-machine-learning 'qiskit-machine-learning[sparse]' qiskit_aer qiskit-aer-gpu qiskit_algorithms torch matplotlib pylatexenc ipykernelc pyyaml
# IMPORTANT: Make sure you are on 3.10
# May need to restart the kernel after instalation

#--- Imports ---#
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import random_statevector, Statevector, SparsePauliOp
from qiskit.circuit.library import real_amplitudes, efficient_su2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qpy

from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient
from qiskit_machine_learning.connectors import TorchConnector

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as EstimatorV2_sim, SamplerV2 as SamplerV2_sim
from qiskit_aer.quantum_info import AerStatevector
from qiskit_aer.library import SaveProbabilities

from qiskit_ibm_runtime import EstimatorV2 as EstimatorV2_rh, QiskitRuntimeService, Session
from qiskit_ibm_runtime.options import EstimatorOptions # TODO

from qiskit_algorithms.gradients import ReverseEstimatorGradient

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import signal
import datetime as dt
import pickle
import yaml
import argparse

# %%
#- Parameter management for python scripts -#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fully Quantum GAN"
    )
    parser.add_argument("-c", "--config_path", required=True, type=str)
    args = parser.parse_args()

    configuration_file_path = args.config_path

# %%
#- Load configuration file -#

#configuration_file_path = "../data/test/qgan_TorchConnector/q4/noiseless/CPU/PSR/seed0/id0/config.yaml"
config_path = os.path.dirname(configuration_file_path) + "/"

# Load config file
def load_config_file(filename):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)

    train_c = data['train_config']
    backend_c = data['backend_config']

    return train_c, backend_c


train_config, backend_config = load_config_file(configuration_file_path)

# %%
#- Create backend -#

# # Save account
# QiskitRuntimeService.save_account(
#     token="",
#     instance="crn:v1:bluemix:public:quantum-computing:eu-de:a/cb804b30dfcb48b890393bfd6e41e9c2:4cb40c64-a531-4c13-b39c-e04c31185259::",
#     set_as_default = True,
#     overwrite=True
# )

# Save backend file
def create_backend_file(backend, filename):
    backend_dict = {
        'timestamp': dt.datetime.now(dt.timezone.utc),
        'configuration': backend.configuration(),
        'properties': backend.properties(),
        'target': backend.target,
        'options': backend.options,
        'noise_model': None if train_config["execution_type"] == "noiseless" else backend.options.noise_model
    }

    with open(filename, "wb") as f:
        pickle.dump(backend_dict, f)

    print("Backend file created.")


# Load backend file (just for simulation)
def load_backend_file(filename):
    if train_config['reset_backend'] or not os.path.exists(filename):
        # Get real backend info
        if train_config['execution_type'] == "noisy" and backend_config['reset_backend']:
            service = QiskitRuntimeService(channel=backend_config['channel'])
            real_backend = service.backend(backend_config['name']) #backend = service.least_busy(min_num_qubits=30)
            backend = AerSimulator.from_backend(real_backend, **backend_config['sim_options']) # Get current backend state
        else:
            backend = AerSimulator(**backend_config['sim_options'])

        create_backend_file(backend, filename)

    with open(filename, "rb") as f:
            backend_dict = pickle.load(f)

    return backend_dict


# Create backend
if train_config['execution_type'] == "real":
    # Get real backend
    service = QiskitRuntimeService(channel=backend_config['channel'])
    backend = service.backend(backend_config['name']) #backend = service.least_busy(min_num_qubits=30)

    # Save backend info
    create_backend_file(backend, config_path + "backend.pkl")

    # Create session
    session = Session(backend=backend)

    # Create estimator
    estimator = EstimatorV2_rh(session=session) # ? TODO qiskit-ibm-runtime

else:
    # Load backend configuration
    backend_dict = load_backend_file(config_path + "backend.pkl")

    # Create backend
    backend = AerSimulator(
        configuration=backend_dict['configuration'],
        properties=backend_dict['properties'],
        target=backend_dict['target'],
        **backend_dict['options']
    )

    # Create Estimator for simulation
    estimator = EstimatorV2_sim(
        options = {
            "default_precision": backend_config["train_precision"],
            "backend_options": backend.options,
        })


# Transpilation method
pm = generate_preset_pass_manager(
    optimization_level=3,
    backend=backend,
    seed_transpiler=train_config['seed']
)


# Backend and pass manager for noiseless evaluation (do not need to execute evaluation in a nosiy environment)
eval_backend = AerSimulator(**backend_config['eval_options'])
eval_pm = generate_preset_pass_manager(optimization_level=3, backend=eval_backend, seed_transpiler=train_config['seed'])



# Select device torch
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" # before torch import to select specific devices
#import torch
if train_config['device'] == "GPU" and torch.cuda.is_available():
    print(f"GPUs available to PyTorch: {torch.cuda.device_count()}")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if backend_config["sim_options"]["precision"] == "double":
    dtype = torch.float64
else:
    dtype = torch.float32


# Print backend properties
print(backend)

# %%
#- Create quantum circuits -#

# Create real data sample circuit
def generate_real_circuit():
    n_qubits = train_config['n_qubits']

    # sv = random_statevector(2**n_qubits, seed=train_config['seed'])
    # qc = QuantumCircuit(n_qubits)
    # qc.prepare_state(sv, qc.qubits, normalize=True)

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits-1))
    qc.cx(n_qubits-2, n_qubits-1)
    return qc


# Create generator
def generate_generator():
    n_qubits = train_config['n_qubits']

    qc = real_amplitudes(n_qubits,
                        reps=3, # Number of layers
                        parameter_prefix='θ_g',
                        name='Generator')
    
    return qc.decompose()


# Create discriminator
def generate_discriminator():
    n_qubits = train_config['n_qubits']

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
    
    return qc


# Create circuits file
def create_circuits_file(filename):
    real_circuit = generate_real_circuit()
    generator_circuit = generate_generator()
    discriminator_circuit = generate_discriminator()

    with open(filename, 'wb') as fd:
        qpy.dump([real_circuit, generator_circuit, discriminator_circuit], fd)

    print("Circuits file created.")


# Load circuits from file
def load_circuits_file(filename):
    if train_config['create_circuits'] or not os.path.exists(filename):
        create_circuits_file(filename)

    with open(filename, 'rb') as fd:
        circuits = qpy.load(fd)

    return circuits[0], circuits[1], circuits[2]
    
    
real_circuit, generator_circuit, discriminator_circuit = load_circuits_file(config_path + "circuits.qpy")

# %%
#- Set up training quantum circuits -#
def generate_training_circuits(real_circuit, generator_circuit, discriminator_circuit):
    n_qubits = train_config['n_qubits']

    # Connect real data and discriminator
    real_disc_circuit = QuantumCircuit(n_qubits)
    real_disc_circuit.compose(real_circuit, inplace=True)
    real_disc_circuit.compose(discriminator_circuit, inplace=True)

    # Connect generator and discriminator
    gen_disc_circuit = QuantumCircuit(n_qubits)
    gen_disc_circuit.compose(generator_circuit, inplace=True)
    gen_disc_circuit.compose(discriminator_circuit, inplace=True)


    # Gradient computation method
    if train_config['gradient_method'] == 'SPSA':
        gradient = SPSAEstimatorGradient(estimator=estimator, seed=train_config['seed'])
    elif train_config['gradient_method'] == 'REG':
        gradient = ReverseEstimatorGradient()
    else:
        gradient = ParamShiftEstimatorGradient(estimator=estimator)


    # Observables
    H1 = SparsePauliOp.from_list([("Z" + "I"*(n_qubits-1), 1.0)])


    # Transpilation
    real_disc_circuit_transpiled, gen_disc_circuit_transpiled = pm.run([real_disc_circuit, gen_disc_circuit])
    obs_real_disc = [H1.apply_layout(real_disc_circuit_transpiled.layout)]
    obs_gen_disc = [H1.apply_layout(gen_disc_circuit_transpiled.layout)]


    N_DPARAMS = discriminator_circuit.num_parameters

    # specify QNN to update generator parameters
    gen_qnn = EstimatorQNN(circuit=gen_disc_circuit_transpiled,
                        input_params=gen_disc_circuit_transpiled.parameters[:N_DPARAMS], # fixed parameters (discriminator parameters)
                        weight_params=gen_disc_circuit_transpiled.parameters[N_DPARAMS:], # parameters to update (generator parameters)
                        estimator=estimator,
                        observables=obs_gen_disc,
                        gradient=gradient,
                        default_precision=backend_config["train_precision"],
                        #pass_manager=pm, # Not needed, already tranpsiled
                        input_gradients=True
                        )

    # specify QNN to update discriminator parameters regarding to fake data
    disc_fake_qnn = EstimatorQNN(circuit=gen_disc_circuit_transpiled,
                            input_params=gen_disc_circuit_transpiled.parameters[N_DPARAMS:], # fixed parameters (generator parameters)
                            weight_params=gen_disc_circuit_transpiled.parameters[:N_DPARAMS], # parameters to update (discriminator parameters)
                            estimator=estimator,
                            observables=obs_gen_disc,
                            gradient=gradient,
                            default_precision=backend_config["train_precision"],
                            #pass_manager=pm, # Not needed, already tranpsiled
                            input_gradients=True
                            )

    # specify QNN to update discriminator parameters regarding to real data
    disc_real_qnn = EstimatorQNN(circuit=real_disc_circuit_transpiled,
                            input_params=[], # no input parameters
                            weight_params=real_disc_circuit_transpiled.parameters[:N_DPARAMS], # parameters to update (discriminator parameters)
                            estimator=estimator,
                            observables=obs_real_disc,
                            gradient=gradient,
                            default_precision=backend_config["train_precision"],
                            #pass_manager=pm, # Not needed, already tranpsiled
                            input_gradients=True
                            )
    
    # specify Generator evaluator

    # Create Sampler for evaluation, to use with TorchConnector
    sampler = SamplerV2_sim(
        default_shots = backend_config['eval_options']['shots'],
        options = {"backend_options": eval_backend.options}
    )
    # # Noiseless sampler (not BaseV2, cannot be used for TorchConnector)
    # sampler = Sampler_sim(
    #     backend_options = backend.options,
    #     run_options = {
    #         'shots': None,
    #     },
    #     skip_transpilation=True
    # )

    # specify QNN to evaluate generator
    gen_eval_circuit = generator_circuit.copy()
    gen_eval_circuit.measure_all()
    gen_eval_transpiled = SamplerQNN(circuit=gen_eval_circuit,
                            input_params=[], # no input parameters
                            weight_params=gen_eval_circuit.parameters,
                            sampler=sampler,
                            gradient=gradient,
                            pass_manager=eval_pm,
                            input_gradients=False # For evaluation
                            )

    # Noiseless pubs version
    gen_noiseless_eval_circuit = generator_circuit.copy()
    gen_noiseless_eval_circuit.append(SaveProbabilities(gen_noiseless_eval_circuit.num_qubits), gen_noiseless_eval_circuit.qubits)
    gen_noiseless_eval_transpiled = eval_pm.run(gen_noiseless_eval_circuit)
    

    return gen_qnn, disc_fake_qnn, disc_real_qnn, gen_eval_transpiled, gen_noiseless_eval_transpiled

gen_qnn, disc_fake_qnn, disc_real_qnn, gen_eval_transpiled, gen_noiseless_eval_transpiled = generate_training_circuits(real_circuit, generator_circuit, discriminator_circuit)

# %%
#f_loss = torch.nn.MSELoss(reduction="sum")
class FLoss(torch.nn.Module):
    def __init__(self):
        super(FLoss, self).__init__()

    def forward(self, x, label):
        loss = -x * label
        return loss.mean()
    
f_loss = FLoss()


class JoinedDiscriminator(torch.nn.Module):
    def __init__(self, disc_real_qnn, disc_fake_qnn, initial_weights=None):
        super().__init__()
        self.model_dr = TorchConnector(disc_real_qnn)
        self.model_df = TorchConnector(disc_fake_qnn, initial_weights=initial_weights)
        self.tie_weights()

    def tie_weights(self):
        if 'weight' in self.model_dr._parameters:
            self.model_dr._parameters.pop('weight')
        if '_weights' in self.model_dr._parameters:
            self.model_dr._parameters.pop('_weights')
        self.model_dr._parameters['weight'] = self.model_df.weight
        self.model_dr._parameters['_weights'] = self.model_df.weight

    def parameters(self, recurse=True):
        return self.model_df.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self.model_df.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def state_dict(self, *args, **kwargs):
        return {
            'model_df_state': self.model_df.state_dict(*args, **kwargs),
        }

    def load_state_dict(self, state_dict, strict=True):
        result = self.model_df.load_state_dict(state_dict['model_df_state'], strict=strict)
        self.tie_weights()
        return result

    def _apply(self, fn):
        super()._apply(fn)
        self.tie_weights()
        return self

    def forward(self, fake_input):
        real_output = self.model_dr()
        fake_output = self.model_df(fake_input)
        return real_output, fake_output


# %%
#- Restore parameters and model states -#

# Create training data file
def create_training_data_file(n_gen_params, n_disc_params, filename):
    np.random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])

    init_gen_params = np.random.uniform(low=-np.pi, high=np.pi, size=(n_gen_params,)) * 0.1 # Start from near 0 parameters to mitigate drastic changes at the start
    init_disc_params = np.random.uniform(low=-np.pi, high=np.pi, size=(n_disc_params,)) * 0.1

    gen_params = torch.tensor(init_gen_params, requires_grad=True, dtype=dtype)
    disc_params = torch.tensor(init_disc_params, requires_grad=True, dtype=dtype)

    params = {
        'init_gen_params': init_gen_params,
        'init_disc_params': init_disc_params,
        'gen_params': gen_params,
        'disc_params': disc_params,
        'best_gen_params': init_gen_params,
        'current_epoch': 0,
        "metrics": {
            "gloss": {},
            "dloss": {},
            "eval": {},
            'times': {},
        },
        'np_random_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state()
    }

    model_g = TorchConnector(gen_qnn, initial_weights=init_gen_params)
    model_d = JoinedDiscriminator(disc_real_qnn, disc_fake_qnn, initial_weights=init_disc_params)
    eval_g = TorchConnector(gen_eval_transpiled)

    if 'weight' in eval_g._parameters:
        eval_g._parameters.pop('weight')
    if '_weights' in eval_g._parameters:
        eval_g._parameters.pop('_weights')
    eval_g._parameters['weight'] = model_g.weight
    eval_g._parameters['_weights'] = model_g.weight

    model_g.train() 
    model_d.train()
    eval_g.eval()

    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=0.005)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.005)

    params['model_g_state'] = model_g.state_dict()
    params['model_d_state'] = model_d.state_dict()
    params['eval_g_state'] = eval_g.state_dict()
    

    params['optimizer_g_state'] = optimizer_g.state_dict()
    params['optimizer_d_state'] = optimizer_d.state_dict()

    torch.save(params, filename)

    print("Training data file created.")


# Load parameters and training states from file
def load_training_data_file(filename):
    if train_config['reset_training_data'] or not os.path.exists(filename):
        create_training_data_file(generator_circuit.num_parameters, discriminator_circuit.num_parameters, filename)

    params = torch.load(filename, weights_only=False, map_location=device)

    return params


params = load_training_data_file(config_path + "training_data.pth")

np.random.set_state(params['np_random_state'])
torch.set_rng_state(params['torch_rng_state'])

gen_params = params['gen_params']
disc_params = params['disc_params']

current_epoch = params['current_epoch']
epoch = current_epoch - 1
gloss = params['metrics']['gloss']
gen_loss = list(gloss.values())[-1] if gloss else None
dloss = params['metrics']['dloss']
disc_loss = list(dloss.values())[-1] if dloss else None
eval = params['metrics']['eval']
min_eval = np.min(list(eval.values())) if (eval) else float('inf')
best_gen_params = params['best_gen_params']
times = params['metrics']['times']


model_g = TorchConnector(gen_qnn)    
model_d = JoinedDiscriminator(disc_real_qnn, disc_fake_qnn)
eval_g = TorchConnector(gen_eval_transpiled)

model_g.load_state_dict(params['model_g_state'])
model_d.load_state_dict(params['model_d_state'])
eval_g.load_state_dict(params['eval_g_state'])

model_g.to(device)
model_d.to(device)
eval_g.to(device)

if 'weight' in eval_g._parameters:
    eval_g._parameters.pop('weight')
if '_weights' in eval_g._parameters:
    eval_g._parameters.pop('_weights')
eval_g._parameters['weight'] = model_g.weight
eval_g._parameters['_weights'] = model_g.weight

optimizer_g = torch.optim.Adam(model_g.parameters())
optimizer_d = torch.optim.Adam(model_d.parameters())


optimizer_g.load_state_dict(params['optimizer_g_state'])
optimizer_d.load_state_dict(params['optimizer_d_state'])


# %%
#- Manage training interruption -#

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

# %%
#- Evualuation method -#

# Evaluation method: KL-Div of generated (ger_dist) and real (target) sample
def evaluate(gen_dist, target):
    return torch.nn.functional.kl_div(
        input = gen_dist.clamp_min(1e-10).log(), #torch.nn.functional.log_softmax(gen_dist, dim=-1),
        target = target, 
        reduction = 'sum' 
    ).item()

# %%
#- Forward and backward pass -#

# Discriminator pass
def disc_pass():
    optimizer_d.zero_grad()

    # Calculate discriminator loss with real and generated data
    gen_params = torch.nn.utils.parameters_to_vector(model_g.parameters()).detach() #gen_params = optimizer_g.param_groups[0]['params'][0].detach()
    real_output, fake_output = model_d(gen_params)
    real_loss = f_loss(real_output, torch.ones_like(real_output)) # 1-> Real guess (correct)
    fake_loss = f_loss(fake_output, -torch.ones_like(fake_output)) # -1-> Fake guess (correct)

    disc_total_loss = real_loss + fake_loss
    disc_total_loss.backward()

    optimizer_d.step()

    # Calculate discriminator cost
    disc_loss = (real_loss.item() + fake_loss.item() -2)/4

    return disc_loss

# Generator pass
def gen_pass():
    optimizer_g.zero_grad()

    # Calculate generator gradient
    disc_params = torch.nn.utils.parameters_to_vector(model_d.parameters()).detach() #disc_params = optimizer_d.param_groups[0]['params'][0].detach()
    gen_output = model_g(disc_params)
    gen_loss = f_loss(gen_output, torch.ones_like(gen_output)) # 1-> Real guess (decieved)
    gen_loss.backward()  # Backward pass

    optimizer_g.step()

    # Calculate generator cost
    gen_loss = (gen_loss.item() -1)/2
    
    return gen_loss

# Copy parameters
def copy_params():
    return torch.nn.utils.parameters_to_vector(model_g.parameters()).detach().cpu().numpy().copy()


#Evaluate performance
if backend_config['eval_precision'] != 0.0:
    # TorchConnector version
    def evaluation(real_distribution_tensor):
        gen_dist = eval_g()

        # Performance measurement function: uses Kullback Leibler Divergence to measures the distance between two distributions
        current_eval = evaluate(gen_dist, real_distribution_tensor)

        return current_eval
    
else:
    # Noiseless pubs version
    def evaluation(real_distribution_tensor):
        # Get generator parameters as numpy
        gen_params = torch.nn.utils.parameters_to_vector(model_g.parameters()).detach() #gen_params = optimizer_g.param_groups[0]['params'][0].detach()

        # Get fake samples
        parameter_binds = {gen_noiseless_eval_transpiled.parameters[i]: [gen_params[i]] for i in range(len(gen_noiseless_eval_transpiled.parameters))}
        job = eval_backend.run([gen_noiseless_eval_transpiled], parameter_binds=[parameter_binds])
        result = job.result()
        all_counts = result.data(0)['probabilities']
        gen_dist = torch.tensor(all_counts, dtype=dtype, device=device)

        # Performance measurement function: uses Kullback Leibler Divergence to measures the distance between two distributions
        current_eval = evaluate(gen_dist, real_distribution_tensor)

        return current_eval


# %%
#- Training -#

D_STEPS = train_config['disc_iterations']
G_STEPS = train_config['gen_iterations']

real_distribution_tensor = torch.tensor(Statevector(real_circuit).probabilities(), dtype=dtype, device=device) # Retrieve real data probability distribution 

interrupter = Interrupter()

if train_config['print_progress_iterations']:
    TABLE_HEADERS = "Epoch | Generator cost | Discriminator cost | Eval | Best eval | Time |"
    print(TABLE_HEADERS)

prev_times = 0
start_time = time.time()

#--- Training loop ---#
try: # In case of interruption
    for epoch in range(current_epoch, train_config['max_iterations']+1):

        #--- Quantum discriminator parameter updates ---#
        for disc_train_step in range(D_STEPS):
            disc_loss = disc_pass()
            dloss[epoch] = disc_loss


        #--- Quantum generator parameter updates ---#
        for gen_train_step in range(G_STEPS):
            gen_loss = gen_pass()
            gloss[epoch] = gen_loss


        #--- Track KL and save best performing generator weights ---#
        current_eval = evaluation(real_distribution_tensor)
        eval[epoch] = current_eval
        if min_eval > current_eval:
            min_eval = current_eval
            best_gen_params = copy_params() # New best
        

        # Calculate time
        cur_time = (time.time() - start_time)
        times[epoch] = cur_time
        start_time = time.time()


        #--- Print progress ---#
        if train_config['print_progress_iterations'] and (epoch % train_config['print_progress_iterations'] == 0):
            now_times = sum(times.values())
            for header, val in zip(TABLE_HEADERS.split('|'),
                                (epoch, gen_loss, disc_loss, current_eval, min_eval, now_times - prev_times)):
                print(f"{val:.3g} ".rjust(len(header)), end="|")
            print()

            prev_times = now_times


        # In case of interruption
        if interrupter.kill_now:
            print("Interrupter: Graceful exit triggered. Breaking loop.")
            break
            
#--- Save parameters and optimizer states data ---#
finally:
    params = {
        'init_gen_params': params['init_gen_params'],
        'init_disc_params': params['init_disc_params'],
        'best_gen_params': best_gen_params,
        'gen_params': torch.nn.utils.parameters_to_vector(model_g.parameters()).detach().cpu(),
        'disc_params': torch.nn.utils.parameters_to_vector(model_d.parameters()).detach().cpu(),
        'current_epoch': epoch+1,
        "metrics": {
            "gloss": gloss,
            "dloss": dloss,
            "eval": eval,
            'times': times,
        },
        'np_random_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'optimizer_g_state': optimizer_g.state_dict(),
        'optimizer_d_state': optimizer_d.state_dict(),
    }

    params['model_g_state'] = model_g.state_dict()
    params['model_d_state'] = model_d.state_dict()
    params['eval_g_state'] = eval_g.state_dict()
    
    torch.save(params, config_path + "training_data.pth")
    
    if train_config['execution_type'] == "real":
        session.close()

    eval_data = list(eval.values()) if eval else [0]
    print("Training complete:", "\n   Data path:", config_path, "\n   Best eval:", np.min(eval_data), "in epoch", np.argmin(eval_data), "\n   Improvement:", eval_data[0]-np.min(eval_data), "\n   Total time:", sum(times.values()))



