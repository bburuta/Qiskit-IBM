import torch



#- Torch configuration -#

# Get torch device
def get_device(device_type):
    if device_type == "GPU" and torch.cuda.is_available():
        print(f"GPUs available to PyTorch: {torch.cuda.device_count()}")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    return device


# Get torch data type
def get_dtype(data_type):
    if data_type == "double":
        dtype = torch.float64
    else:
        dtype = torch.float32
    
    return dtype


# Move torch models to device and dtype
def move_models(model_g, model_d, eval_g, device, dtype):
    model_g.to(device=device, dtype=dtype)
    model_d.to(device=device, dtype=dtype)
    eval_g.to(device=device, dtype=dtype)



#- Torch parameters -#

# Get torch model params
def get_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()
