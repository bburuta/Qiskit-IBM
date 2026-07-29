import torch

from models_torch import AmpEvalBackend



#- Batch parallelization -#

# Create random input
def generate_torch_random_input(batch_size, num_params, randomness, device, dtype):
    return 2 * torch.pi * torch.rand(
        batch_size,
        num_params,
        device=device,
        dtype=dtype,
    ) * randomness


# Get real data indexes
def generate_torch_real_input_index(batch_size, num_samples, device):
    if batch_size > num_samples:
        data_indexes = torch.randint(low=0, high=num_samples, size=(batch_size,), device=device)
    else:
        data_indexes = torch.randperm(num_samples, device=device)[:batch_size]
    
    return data_indexes


# Get real data input
def generate_torch_real_input(X, batch_size, device):
    data_indexes = torch.randint(
        low=0,
        high=X.shape[0],
        size=(batch_size,),
        device=device,
    )

    return X[data_indexes].reshape(batch_size, -1)


# Generate fake input for discriminator
def generate_torch_fake_disc_input(model_g, batch_size, num_random_params, randomness, device, dtype):
    gen_params = torch.nn.utils.parameters_to_vector(model_g.parameters()).detach()

    gen_batch = gen_params.reshape(1, -1).expand(batch_size, -1)
    random_batch = generate_torch_random_input(batch_size, num_random_params, randomness, device, dtype)

    return torch.cat([gen_batch, random_batch], dim=1)


# Generate input for generator
def generate_torch_gen_input(model_d, batch_size, num_random_params, randomness, device, dtype):
    disc_params = torch.nn.utils.parameters_to_vector(model_d.parameters()).detach()

    disc_batch = disc_params.reshape(1, -1).expand(batch_size, -1)
    random_batch = generate_torch_random_input(batch_size, num_random_params, randomness, device, dtype)

    return torch.cat([disc_batch, random_batch], dim=1)


# Get number of random input parameters
def get_torch_num_random_params(eval_g):
    if isinstance(eval_g, AmpEvalBackend):
        num_random_params = len(eval_g.circuit_params) - eval_g.n_weights
    else:
        num_random_params = eval_g.neural_network.num_inputs

    return num_random_params
