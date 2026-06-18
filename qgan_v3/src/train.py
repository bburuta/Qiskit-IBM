import time
import random
import numpy as np
import torch

from config_manager.config_manager import get_run_path
from datasets.images_dataset import get_images_dataset
from circuits import get_circuits, generate_circuits_sha
from models_torch import (
    f_loss,
    generate_torch_models,
    tie_eval_weights,
)
from torch_utils import (
    get_torch_device,
    get_torch_dtype,
    move_torch_models,
    get_torch_params,
    copy_gen_params,
)
from batch_torch import (
    get_torch_num_random_params,
    generate_torch_real_input,
    generate_torch_real_input_index,
    generate_torch_fake_disc_input,
    generate_torch_gen_input,
)
from evaluation_torch import get_evaluation_function, get_target_probs, batch_evaluation
from interrupter import Interrupter



#- Training data save -#

TRAINING_DATA_FILENAME = "training_data.pth"


# Get training data file path
def get_training_data_filename(run_path):
    return run_path / TRAINING_DATA_FILENAME


# Set random seed
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Get random state
def get_random_state():
    params = {
        'random_state': random.getstate(),
        'np_random_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        params['torch_cuda_rng_state'] = torch.cuda.get_rng_state_all()

    return params


# Set random state
def set_random_state(params):
    if 'random_state' in params:
        random.setstate(params['random_state'])

    np.random.set_state(params['np_random_state'])
    torch.set_rng_state(params['torch_rng_state'].cpu())

    if 'torch_cuda_rng_state' in params and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(params['torch_cuda_rng_state'])


# Initialize model parameters
def initialize_model_params(model, device, dtype):
    n_params = get_torch_params(model).numel()
    init_params = np.random.uniform(low=-np.pi, high=np.pi, size=(n_params,)) * 0.1
    init_tensor = torch.as_tensor(init_params, device=device, dtype=dtype)
    torch.nn.utils.vector_to_parameters(init_tensor, model.parameters())

    return init_params


# Create optimizer
def create_optimizers(model_g, model_d):
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=0.005)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.005)

    return optimizer_g, optimizer_d


# Save training data file
def save_training_data_file(filename, params):
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(params, filename)


# Get circuits SHA for the current config
def get_current_circuits_sha(config):
    generator_circuit, discriminator_circuit, randomizer_circuit, real_circuits = get_circuits(config)

    return generate_circuits_sha([generator_circuit, discriminator_circuit, randomizer_circuit, *real_circuits])


# Create training data
def create_training_data(seed, circuits_sha, model_g, model_d, eval_g, device, dtype):
    set_random_seed(seed)

    init_gen_params = initialize_model_params(model_g, device, dtype)
    init_disc_params = initialize_model_params(model_d, device, dtype)
    tie_eval_weights(eval_g, model_g)

    optimizer_g, optimizer_d = create_optimizers(model_g, model_d)

    params = {
        'circuits_sha': circuits_sha,
        'init_gen_params': init_gen_params,
        'init_disc_params': init_disc_params,
        'best_gen_params': init_gen_params,
        'gen_params': get_torch_params(model_g).cpu(),
        'disc_params': get_torch_params(model_d).cpu(),
        'current_epoch': 0,
        'metrics': {
            'gloss': {},
            'dloss': {},
            'eval': {},
            'times': {},
        },
        'model_g_state': model_g.state_dict(),
        'model_d_state': model_d.state_dict(),
        'eval_g_state': eval_g.state_dict(),
        'optimizer_g_state': optimizer_g.state_dict(),
        'optimizer_d_state': optimizer_d.state_dict(),
    }
    params.update(get_random_state())

    return params, optimizer_g, optimizer_d


# Load training data
def load_training_data(filename, circuits_sha, model_g, model_d, eval_g, device):
    params = torch.load(filename, weights_only=False, map_location=device)
    saved_circuits_sha = params.get('circuits_sha', params.get('saved_sha'))

    if saved_circuits_sha is not None and saved_circuits_sha != circuits_sha:
        print("Training data circuits_sha is different from current circuits_sha.")

    model_g.load_state_dict(params['model_g_state'])
    model_d.load_state_dict(params['model_d_state'])
    eval_g.load_state_dict(params['eval_g_state'])
    tie_eval_weights(eval_g, model_g)

    set_random_state(params)

    optimizer_g, optimizer_d = create_optimizers(model_g, model_d)
    optimizer_g.load_state_dict(params['optimizer_g_state'])
    optimizer_d.load_state_dict(params['optimizer_d_state'])

    return params, optimizer_g, optimizer_d


# Load or create training data
def load_or_create_training_data(filename, seed, circuits_sha, model_g, model_d, eval_g, device, dtype):
    if filename.exists():
        params, optimizer_g, optimizer_d = load_training_data(filename, circuits_sha, model_g, model_d, eval_g, device)
    else:
        params, optimizer_g, optimizer_d = create_training_data(seed, circuits_sha, model_g, model_d, eval_g, device, dtype)
        save_training_data_file(filename, params)
        print("Training data file created.")

    return params, optimizer_g, optimizer_d



#- Training inputs -#

# Get real discriminator input
def generate_real_disc_input(encoding, X, model_d, batch_size, device):
    if encoding == 'direct_circuit':
        real_input = None
    elif encoding == 'angle':
        real_input = generate_torch_real_input(X, batch_size, device)
    elif encoding == 'amplitude':
        real_input = generate_torch_real_input_index(batch_size, model_d.num_real_models, device)
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")

    return real_input



#- Forward and backward pass -#

# Discriminator pass
def disc_pass(encoding, model_g, model_d, optimizer_d, X, batch_size, num_random_params, randomness, device, dtype):
    optimizer_d.zero_grad()

    # Calculate discriminator gradient with real and generated data
    real_inputs = generate_real_disc_input(encoding, X, model_d, batch_size, device)
    fake_inputs = generate_torch_fake_disc_input(model_g, batch_size, num_random_params, randomness, device, dtype)
    real_output, fake_output = model_d(real_inputs, fake_inputs)
    real_loss = f_loss(real_output, torch.ones_like(real_output)) # 1-> Real guess (correct)
    fake_loss = f_loss(fake_output, -torch.ones_like(fake_output)) # -1-> Fake guess (correct)

    loss = real_loss + fake_loss
    loss.backward()

    optimizer_d.step()

    # Calculate discriminator cost
    disc_loss = (real_loss.item() + fake_loss.item() - 2) / 4

    return disc_loss


# Generator pass
def gen_pass(model_g, model_d, optimizer_g, batch_size, num_random_params, randomness, device, dtype):
    optimizer_g.zero_grad()

    # Calculate generator gradient
    gen_inputs = generate_torch_gen_input(model_d, batch_size, num_random_params, randomness, device, dtype)
    gen_output = model_g(gen_inputs)
    gen_loss = f_loss(gen_output, torch.ones_like(gen_output)) # 1-> Real guess (decieved)
    gen_loss.backward()

    optimizer_g.step()

    # Calculate generator cost
    gen_loss = (gen_loss.item() - 1) / 2
    
    return gen_loss



#- Training management -#

# Get loaded metric values
def get_training_state(params):
    current_epoch = params['current_epoch']
    gloss = params['metrics']['gloss']
    dloss = params['metrics']['dloss']
    eval_metrics = params['metrics']['eval']
    times = params['metrics']['times']
    best_gen_params = params['best_gen_params']
    min_eval = np.min(list(eval_metrics.values())) if eval_metrics else float('inf')

    return current_epoch, gloss, dloss, eval_metrics, times, best_gen_params, min_eval


# Save final training data
def save_final_training_data(filename, circuits_sha, model_g, model_d, eval_g, optimizer_g, optimizer_d, params, best_gen_params, epoch, gloss, dloss, eval_metrics, times):
    params = {
        'circuits_sha': circuits_sha,
        'init_gen_params': params['init_gen_params'],
        'init_disc_params': params['init_disc_params'],
        'best_gen_params': best_gen_params,
        'gen_params': get_torch_params(model_g).cpu(),
        'disc_params': get_torch_params(model_d).cpu(),
        'current_epoch': epoch + 1,
        'metrics': {
            'gloss': gloss,
            'dloss': dloss,
            'eval': eval_metrics,
            'times': times,
        },
        'model_g_state': model_g.state_dict(),
        'model_d_state': model_d.state_dict(),
        'eval_g_state': eval_g.state_dict(),
        'optimizer_g_state': optimizer_g.state_dict(),
        'optimizer_d_state': optimizer_d.state_dict(),
    }
    params.update(get_random_state())

    save_training_data_file(filename, params)


# Print training progress
def print_training_progress(epoch, gen_loss, disc_loss, current_eval, min_eval, elapsed_time):
    values = (epoch, gen_loss, disc_loss, current_eval, min_eval, elapsed_time)
    for val in values:
        print(f"{val:.3g} ".rjust(20), end="|")
    print()


# Print training summary
def print_training_summary(run_path, eval_metrics, times):
    eval_data = list(eval_metrics.values()) if eval_metrics else [0]
    print(
        "Training complete:",
        "\n   Data path:", run_path,
        "\n   Best eval:", np.min(eval_data),
        "\n   Total time:", sum(times.values())
    )



#- Training -#

# Train qGAN
def train(config, X=None):
    encoding = config['encoding']['type']
    randomness = config['encoding']['randomness']
    batch_size = config['encoding']['batch_size']
    eval_batch_size = config['encoding']['eval_batch_size']
    eval_method = config['encoding']['eval_method']

    device = get_torch_device(config['run']['device'])
    dtype = get_torch_dtype(config['backend']['simulator']['data_type'])
    seed = config['run']['seed']
    run_path = get_run_path(config)
    training_data_filename = get_training_data_filename(run_path)

    circuits_sha = get_current_circuits_sha(config)
    model_g, model_d, eval_g = generate_torch_models(config)
    move_torch_models(model_g, model_d, eval_g, device, dtype)

    num_random_params = get_torch_num_random_params(eval_g)
    params, optimizer_g, optimizer_d = load_or_create_training_data(training_data_filename, seed, circuits_sha, model_g, model_d, eval_g, device, dtype)
    current_epoch, gloss, dloss, eval_metrics, times, best_gen_params, min_eval = get_training_state(params)

    if X is None and config['dataset']['type'] == 'classical':
        X = get_images_dataset(config)
    if X is not None:
        X = torch.as_tensor(X, device=device, dtype=dtype)

    _, compute_targets = get_evaluation_function(eval_method)
    if compute_targets:
        _, _, _, real_circuits = get_circuits(config)
        target_probs = get_target_probs(
            config['dataset']['type'],
            config['encoding']['contrast'],
            device,
            dtype,
            real_circuits=real_circuits,
            X=X,
        )
    else:
        target_probs = None

    interrupter = Interrupter()

    if config['training']['print_every']:
        print("Epoch | Generator cost | Discriminator cost | Eval | Best eval | Time |")

    epoch = current_epoch - 1
    prev_times = sum(times.values())
    start_time = time.time()

    #--- Training loop ---#
    try:
        for epoch in range(current_epoch, config['training']['max_iterations']):
            disc_loss = float('nan')
            gen_loss = float('nan')

            #--- Quantum discriminator parameter updates ---#
            for disc_train_step in range(config['training']['disc_iterations']):
                disc_loss = disc_pass(encoding, model_g, model_d, optimizer_d, X, batch_size, num_random_params, randomness, device, dtype)
                dloss[epoch] = disc_loss

            #--- Quantum generator parameter updates ---#
            for gen_train_step in range(config['training']['gen_iterations']):
                gen_loss = gen_pass(model_g, model_d, optimizer_g, batch_size, num_random_params, randomness, device, dtype)
                gloss[epoch] = gen_loss

            #--- Track eval and save best performing generator weights ---#
            current_eval = batch_evaluation(eval_method, randomness, eval_g, eval_batch_size, num_random_params, device, dtype, target_probs=target_probs)
            eval_metrics[epoch] = current_eval
            if min_eval > current_eval:
                min_eval = current_eval
                best_gen_params = copy_gen_params(model_g)

            # Calculate time
            cur_time = time.time() - start_time
            times[epoch] = cur_time
            start_time = time.time()

            #--- Print progress ---#
            if config['training']['print_every'] and (epoch % config['training']['print_every'] == 0):
                now_times = sum(times.values())
                print_training_progress(epoch, gen_loss, disc_loss, current_eval, min_eval, now_times - prev_times)
                prev_times = now_times

            # In case of interruption
            if interrupter.kill_now:
                print("Interrupter: Graceful exit triggered. Breaking loop.")
                break
            
    #--- Save parameters and optimizer states data ---#
    finally:
        save_final_training_data(training_data_filename, circuits_sha, model_g, model_d, eval_g, optimizer_g, optimizer_d, params, best_gen_params, epoch, gloss, dloss, eval_metrics, times)
        print_training_summary(run_path, eval_metrics, times)

    return model_g, model_d, eval_g
