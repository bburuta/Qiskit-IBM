import torch



#- Batch parallelization -#

# Create random input
def generate_random_input(batch_size, num_params, randomness, device, dtype):
    return 2 * torch.pi * torch.rand(
        batch_size,
        num_params,
        device=device,
        dtype=dtype,
    ) * randomness


# Get real data indexes
def generate_real_input_index(batch_size, num_samples, device):
    if batch_size > num_samples:
        data_indexes = torch.randint(low=0, high=num_samples, size=(batch_size,), device=device)
    else:
        data_indexes = torch.randperm(num_samples, device=device)[:batch_size]
    
    return data_indexes


# Get real data input
def generate_real_input(X, batch_size, device):
    data_indexes = torch.randint(
        low=0,
        high=X.shape[0],
        size=(batch_size,),
        device=device,
    )

    return X[data_indexes].reshape(batch_size, -1)


# Generate real discriminator input
def generate_real_disc_input(encoding, x_data, model_d, batch_size, device):
    if encoding == "angle":
        return generate_real_input(x_data, batch_size, device)
    if encoding in ["amplitude", "direct_circuit"]:
        return generate_real_input_index(batch_size, model_d.num_real_models, device)
    raise ValueError(f"Unknown encoding method: {encoding}")


# Generate input: random and model parameter batch
def generate_model_random_input(model, batch_size, num_random_params, randomness, device, dtype):
    model_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

    model_batch = model_params.reshape(1, -1).expand(batch_size, -1)
    random_batch = generate_random_input(batch_size, num_random_params, randomness, device, dtype)

    return torch.cat([model_batch, random_batch], dim=1)


# Get number of random input parameters
def get_num_random_params(eval_g):
    return eval_g.get_num_params()
