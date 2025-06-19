#- Fully Quantum GAN for simulation for angle encoding -#

# Execution example
#python3 fullyq-sim-ang.py --n_qubits 4 --seed 5 --n_epoch 300 --print_progress 1

# INSTALATION INSTRUCTIONS:
# For linux 64-bit systems,
#uname -a

# Conda quick installation
#mkdir -p ~/miniconda3
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
#bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
#rm ~/miniconda3/miniconda.sh
#~/miniconda3/bin/conda init
#source ~/.bashrc


# Create enviroment with conda
#conda create -n myenv python=3.10
#conda activate myenv
#pip install qiskit==1.4.3 qiskit-machine-learning==0.8.2 'qiskit-machine-learning[sparse]' tensorflow[and-cuda]
# IMPORTANT: Make sure you are on 3.10
# May need to restart the kernel after instalation

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator

from qiskit_machine_learning.neural_networks import EstimatorQNN # Downgrade to qiskit 1.x so is compatible with qiskit-machine-learning 0.8.2
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = only ERROR
import tensorflow as tf
import numpy as np
import copy
import time
import argparse


# Parameter management for python scripts
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fully Quantum QGAN"
    )
    parser.add_argument("--n_qubits", required=True, type=int)
    parser.add_argument("--seed", required=False, type=int, default=np.random.randint(1, 1000))
    parser.add_argument("--print_progress", required=False, type=int, default=0)
    parser.add_argument("--n_epoch", required=False, type=int, default=300)
    parser.add_argument("--reset_data", required=False, type=int, default=0)
    parser.add_argument("--batch_size", required=False, type=int, default=1)
    parser.add_argument("--gpu_index", required=False, type=int, default="-1")
    args = parser.parse_args()

    N_QUBITS = args.n_qubits

    SEED = args.seed
    print_progress = args.print_progress

    max_epoch = args.n_epoch
    reset = args.reset_data
    batch_size = args.batch_size

    gpu_index = args.gpu_index


# Select GPU for tensorflow
if (gpu_index != -1):
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[gpu_index], 'GPU') # Set only one GPU as visible
    print("Device that is going to be used:", tf.config.list_logical_devices('GPU'))


# Build my own dataset of images: gradient images
def apply_curve(x, curve):
    if curve == 'linear':
        return x
    elif curve == 'quadratic':
        return x ** 2
    elif curve == 'sqrt':
        return np.sqrt(x)
    elif curve == 'log':
        return np.log1p(x * 9) / np.log(10)  # scale [0,1] into [0,1] log space
    elif curve == 'exp':
        return (np.exp(x * 3) - 1) / (np.exp(3) - 1)  # normalized exponential
    elif curve == 'sigmoid':
        return 1 / (1 + np.exp(-10 * (x - 0.5)))  # smooth S-curve
    elif curve == 'sin':
        return 0.5 * (1 - np.cos(np.pi * x))  # smooth start and end
    else:
        raise ValueError(f"Unknown curve type: {curve}")

def create_gradients(total_pixels, directions=None, curves=None, width=None, height=None):
    if directions is None:
        directions = [
            'top_left_to_bottom_right'
        ]
    if curves is None:
        curves = ['linear', 'quadratic', 'sqrt', 'log', 'exp', 'sigmoid', 'sin']

    if width is None or height is None:
        for h in range(int(np.sqrt(total_pixels)), 0, -1):
            if total_pixels % h == 0:
                width, height = total_pixels // h, h
                break
    elif width * height != total_pixels:
        raise ValueError("Provided width and height do not match total number of pixels.")

    max_val = 255
    gradients = []

    i, j = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Precompute normalized coordinate matrices for all directions
    norm_maps = {
        'left_to_right': np.tile(np.linspace(0, 1, width), (height, 1)),
        'right_to_left': np.tile(np.linspace(1, 0, width), (height, 1)),
        'top_to_bottom': np.tile(np.linspace(0, 1, height)[:, np.newaxis], (1, width)),
        'bottom_to_top': np.tile(np.linspace(1, 0, height)[:, np.newaxis], (1, width)),
        'top_left_to_bottom_right': (i + j) / (width + height - 2),
        'bottom_right_to_top_left': ((height - 1 - i) + (width - 1 - j)) / (width + height - 2),
        'top_right_to_bottom_left': (i + (width - 1 - j)) / (width + height - 2),
        'bottom_left_to_top_right': ((height - 1 - i) + j) / (width + height - 2)
    }

    for direction in directions:
        if direction not in norm_maps:
            raise ValueError(f"Unknown direction: {direction}")
        base_map = norm_maps[direction]

        for curve in curves:
            # Apply curve to normalized map
            curved_map = apply_curve(base_map, curve)
            gradients.append(curved_map)

    return gradients, (height, width)


# Function to decode quantum data back to classical data
def generate_sample(estimator, circuit):
    estimator = StatevectorEstimator()

    observables = []
    for i in range(N_QUBITS):
        text = "I" * i + "Z" + "I" * (N_QUBITS-i-1)
        observables.append(SparsePauliOp.from_list([(text, 1.0)]))

    pub = (circuit, observables)
    job = estimator.run([pub])
    result = job.result()[0]

    return result.data.evs


# Create real data sample circuit
def generate_real_circuit():
    disc_weights = ParameterVector('θ_r', N_QUBITS)
    qc = QuantumCircuit(N_QUBITS, name="Real data circuit")
    param_index = 0

    for q in range(N_QUBITS):
        qc.ry(disc_weights[param_index], q); param_index += 1
    
    return qc


# Create generator
def generate_generator():
    qc = RealAmplitudes(N_QUBITS,
                        reps=2, # Number of layers
                        parameter_prefix='θ_g',
                        name='Generator')
    return qc


# Create discriminator
def generate_discriminator():
    disc_weights = ParameterVector('θ_d', 3*(N_QUBITS+1))
    qc = QuantumCircuit(N_QUBITS, name="Discriminator")
    param_index = 0

    qc.barrier()

    for q in range(N_QUBITS):
        qc.h(q)
        qc.rx(disc_weights[param_index], q); param_index += 1
        qc.ry(disc_weights[param_index], q); param_index += 1
        qc.rz(disc_weights[param_index], q); param_index += 1

    for i in range(N_QUBITS - 1):
        qc.cx(i, N_QUBITS - 1)

    qc.rx(disc_weights[param_index], N_QUBITS-1); param_index += 1
    qc.ry(disc_weights[param_index], N_QUBITS-1); param_index += 1
    qc.rz(disc_weights[param_index], N_QUBITS-1); param_index += 1
    
    return qc


# Set up training quantum circuits
def generate_training_circuits(real_circuit, generator, discriminator):
    # Connect real data and discriminator
    real_disc_circuit = QuantumCircuit(N_QUBITS)
    real_disc_circuit.compose(real_circuit, inplace=True)
    real_disc_circuit.compose(discriminator, inplace=True)

    # Connect generator and discriminator
    gen_disc_circuit = QuantumCircuit(N_QUBITS)
    gen_disc_circuit.compose(generator, inplace=True)
    gen_disc_circuit.compose(discriminator, inplace=True)

    # Use EstimatorQNN to compile the circuit and handle gradient calculation
    estimator = StatevectorEstimator()

    gradient = ParamShiftEstimatorGradient(estimator=estimator)

    H1 = SparsePauliOp.from_list([("Z" + "I"*(N_QUBITS-1), 1.0)])

    # specify QNN to update generator parameters
    gen_qnn = EstimatorQNN(circuit=gen_disc_circuit,
                        input_params=gen_disc_circuit.parameters[:N_DPARAMS], # fixed parameters (discriminator parameters)
                        weight_params=gen_disc_circuit.parameters[N_DPARAMS:], # parameters to update (generator parameters)
                        estimator=estimator,
                        observables=[H1],
                        gradient=gradient,
                        default_precision=0.0
                        )

    # specify QNN to update discriminator parameters regarding to fake data
    disc_fake_qnn = EstimatorQNN(circuit=gen_disc_circuit,
                            input_params=gen_disc_circuit.parameters[N_DPARAMS:], # fixed parameters (generator parameters)
                            weight_params=gen_disc_circuit.parameters[:N_DPARAMS], # parameters to update (discriminator parameters)
                            estimator=estimator,
                            observables=[H1],
                            gradient=gradient,
                            default_precision=0.0
                            )

    # specify QNN to update discriminator parameters regarding to real data
    disc_real_qnn = EstimatorQNN(circuit=real_disc_circuit,
                            input_params=real_disc_circuit.parameters[N_DPARAMS:], # fixed parameters (real data parameters)
                            weight_params=real_disc_circuit.parameters[:N_DPARAMS], # parameters to update (discriminator parameters)
                            estimator=estimator,
                            observables=[H1],
                            gradient=gradient,
                            default_precision=0.0
                            )
    
    return estimator, gen_qnn, disc_fake_qnn, disc_real_qnn


# Initialize Adam optimizer from Keras (TensorFlow)
def generate_optimizers(reset, optimizers_data_folder, gen_params, disc_params):
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Create optimizer training state checkpoints
    ckpt = tf.train.Checkpoint(
        generator_vars=gen_params,
        discriminator_vars=disc_params,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, optimizers_data_folder, max_to_keep=3)
    if reset:
        ckpt_manager.save()
    else:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    return generator_optimizer, discriminator_optimizer, ckpt_manager


# Performance measurement function:
def calculate_performance(gen, real, shape, C1=0.05**2, C2=0.1**2):
    """
    Simplified SSIM using global mean and variance (no sliding window).
    
    Parameters:
    - img1_list, img2_list: Flattened images as lists or 1D arrays
    - shape: (height, width)
    - C1, C2: stability constants
    
    Returns:
    - scalar SSIM score
    """

    img1_list = (gen+1)/2
    img2_list = real/np.pi

    img1 = np.array(img1_list, dtype=np.float64).reshape(shape)
    img2 = np.array(img2_list, dtype=np.float64).reshape(shape)

    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim = numerator / denominator

    distance = 1 - ssim # Convert similarity to distance (0 best)
    
    return distance


def manage_files(data_folder_name="data", implementation_name="fullyq", training_data_file_name='training_data', parameter_data_file_name='parameters', optimizers_data_folder_name='optimizer'):
    data_folder = data_folder_name + '/' + implementation_name + '/' + "sim-ang" + '/' + 'q' + str(N_QUBITS) + '/' + 'seed' + str(SEED) + '/' 
    training_data_file = data_folder + training_data_file_name + '.txt'
    parameter_data_file = data_folder + parameter_data_file_name + '.txt'
    optimizers_data_folder = data_folder + optimizers_data_folder_name + '/'

    # Create data folder if it does not exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    return training_data_file, parameter_data_file, optimizers_data_folder


def initialize_parameters(reset, training_data_file, parameter_data_file):
    if reset == 1:
        current_epoch = 0
        gloss, dloss, perf = [], [], []

        np.random.seed(SEED)
        init_gen_params = np.random.uniform(low=-np.pi, high=np.pi, size=(N_GPARAMS,))
        init_disc_params = np.random.uniform(low=-np.pi, high=np.pi, size=(N_DPARAMS,))
        gen_params = tf.Variable(init_gen_params)
        disc_params = tf.Variable(init_disc_params)
        best_gen_params = tf.Variable(init_gen_params)

        # Reset data files
        with open(training_data_file, 'w') as file:
            pass
        with open(parameter_data_file, 'w') as file:
            file.write(str(gen_params.numpy().tolist()) + ";" + str(disc_params.numpy().tolist()) + ";" + str(gen_params.numpy().tolist()) + ";" + str(disc_params.numpy().tolist()) + ";" + str(best_gen_params.numpy().tolist()) + "\n")
            
    else:
        # Load training data
        try:
            with open(training_data_file) as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            print("Training data file not found. Resetting parameters.")
            return initialize_parameters(1, training_data_file, parameter_data_file)
        current_epoch = len(lines)
        gloss, dloss, perf = [], [], []
        for line in lines:
            line_data = line.split(";")
            if len(line_data) != 4:
                raise Exception("ERROR: Wrong data length in training_data.txt file in line:", line, ". Please, reset data.")
            gloss.append(np.float64(line_data[1]))
            dloss.append(np.float64(line_data[2]))
            perf.append(np.float64(line_data[3]))

        # Load parameters
        with open(parameter_data_file) as f: # Load parameters
            line = f.readline()
        line_data = line.split(";")
        if len(line_data) != 5:
            raise Exception("ERROR: Wrong number of parameters in parameters.txt file. Please, reset data.")
        init_gen_params = np.array(eval(line_data[0])).astype(float)
        init_disc_params = np.array(eval(line_data[1])).astype(float)
        gen_params = tf.Variable(np.array(eval(line_data[2])).astype(float))
        disc_params = tf.Variable(np.array(eval(line_data[3])).astype(float))
        best_gen_params = tf.Variable(np.array(eval(line_data[4])).astype(float))

    return current_epoch, gloss, dloss, perf, init_gen_params, init_disc_params, gen_params, disc_params, best_gen_params


# Training

#--- File management ---#
training_data_file, parameter_data_file, optimizers_data_folder = manage_files()

#--- Create quantum circuits ---#
real_circuit = generate_real_circuit()
generator = generate_generator()
discriminator = generate_discriminator()

N_DPARAMS = discriminator.num_parameters
N_GPARAMS = generator.num_parameters

#--- Initialize parameters ---#
current_epoch, gloss, dloss, perf, init_gen_params, init_disc_params, gen_params, disc_params, best_gen_params = initialize_parameters(reset, training_data_file, parameter_data_file)

#--- Create QNNs ---#
estimator, gen_qnn, disc_fake_qnn, disc_real_qnn = generate_training_circuits(real_circuit, generator, discriminator)

#--- Create and load optimizer states ---#
generator_optimizer, discriminator_optimizer, optimizers_ckpt_manager = generate_optimizers(reset, optimizers_data_folder, gen_params, disc_params)

#--- Load dataset ---#
X, dims = create_gradients(N_QUBITS)
X = [np.pi * real_sample / np.max(real_sample) for real_sample in X] # Rescale to [0, pi]


D_STEPS = 1
G_STEPS = 1
C_STEPS = 1
if print_progress:
    TABLE_HEADERS = "Epoch | Generator cost | Discriminator cost | Perf. | Best Perf. | Time |"
    print(TABLE_HEADERS)
file = open(training_data_file,'a')
start_time = time.time()

#--- Training loop ---#
try: # In case of interruption
    for epoch in range(current_epoch, max_epoch+1):

        #--- Quantum discriminator parameter updates ---#
        for disc_train_step in range(D_STEPS):
            # Calculate discriminator cost
            real_params = [X[np.random.randint(0, len(X))].flatten() for i in range(batch_size)][0]

            if (disc_train_step % D_STEPS == 0) and (epoch % C_STEPS == 0):
                value_dcost_fake = disc_fake_qnn.forward(gen_params, disc_params)[0,0]
                values_dcost_real = disc_real_qnn.forward(real_params, disc_params)
                value_dcost_real = values_dcost_real[0,0]
                # for i in range(1,batch_size):
                #     value_dcost_real += values_dcost_real[i,0]
                disc_loss = ((value_dcost_real - value_dcost_fake)-2)/4
                dloss.append(disc_loss)

            # Caltulate discriminator gradient
            grad_dcost_fake = disc_fake_qnn.backward(gen_params, disc_params)[1][0,0]
            grads_dcost_real = disc_real_qnn.backward(real_params, disc_params)[1]
            grad_dcost_real = grads_dcost_real[0,0]
            # for i in range(1,batch_size):
            #     grad_dcost_real += grads_dcost_real[i,0]
            grad_dcost = grad_dcost_real - grad_dcost_fake
            grad_dcost = tf.convert_to_tensor(grad_dcost)
            
            # Update discriminator parameters
            discriminator_optimizer.apply_gradients(zip([grad_dcost], [disc_params]))

        #--- Quantum generator parameter updates ---#
        for gen_train_step in range(G_STEPS):
            # Calculate generator cost
            if (gen_train_step % G_STEPS == 0) and (epoch % C_STEPS == 0):
                value_gcost = gen_qnn.forward(disc_params, gen_params)[0,0]
                gen_loss = (value_gcost-1)/2
                gloss.append(gen_loss)

            # Calculate generator gradient
            grad_gcost = gen_qnn.backward(disc_params, gen_params)[1][0,0]
            grad_gcost = tf.convert_to_tensor(grad_gcost)

            # Update generator parameters
            generator_optimizer.apply_gradients(zip([grad_gcost], [gen_params]))

        #--- Track performance and save best performing generator weights ---#
        gen_sample = generate_sample(estimator, generator.assign_parameters(gen_params.numpy()))
        
        real_sample = real_params
        
        current_perf = calculate_performance(gen_sample, real_sample, dims)
        perf.append(current_perf)
        if np.min(perf) == current_perf:
            best_gen_params = copy.deepcopy(gen_params) # New best

        #--- Save progress in file ---#
        file.write(str(epoch) + ";" + str(gloss[-1]) + ";" + str(dloss[-1]) + ";" + str(perf[-1]) + "\n")

        #--- Print progress ---#
        if print_progress and (epoch % 10 == 0):
            for header, val in zip(TABLE_HEADERS.split('|'),
                                (epoch, gloss[-1], dloss[-1], perf[-1], np.min(perf), (time.time() - start_time))):
                print(f"{val:.3g} ".rjust(len(header)), end="|")
            start_time = time.time()
            print()
            
#--- Save parameters and optimizer states data ---#
finally:
    file.close()
    file = open(parameter_data_file,'w')
    file.write(str(init_gen_params.tolist()) + ";" + str(init_disc_params.tolist()) + ";" + str(gen_params.numpy().tolist()) + ";" + str(disc_params.numpy().tolist()) + ";" + str(best_gen_params.numpy().tolist()) + "\n")
    file.close()

    optimizers_ckpt_manager.save()
    
print("Training complete:", training_data_file, "Results:", np.min(perf), "Improvement:", perf[0]-np.min(perf))