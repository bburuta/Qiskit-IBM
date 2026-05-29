import yaml
import argparse
import torch

from backend import create_backends
from dataset import load_dataset_file


# Load config file
def load_config_file(filename):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)

    return config



if __name__ == "__main__":

    #- Parameter management for python scripts -#
    parser = argparse.ArgumentParser(
        description="Fully Quantum GAN"
    )
    parser.add_argument("-c", "--config_path", required=True, type=str)
    args = parser.parse_args()

    config_path = args.config_path



    #- Load configuration file -#
    config = load_config_file(config_path)


    #- Create backends -#
    session, train_backend, train_estimator, train_pm, eval_backend, eval_estimator, eval_pm = create_backends(config, config_path)
    print(train_backend) # Print backend properties


    #- Torch configuration -#
    # Select torch device
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" # before torch import to select specific devices
    if config['implementation_options']['device'] == "GPU" and torch.cuda.is_available():
        print(f"GPUs available to PyTorch: {torch.cuda.device_count()}")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if config['backend_options']['sim_backend_options']['data_type'] == "double":
        dtype = torch.float64
    else:
        dtype = torch.float32


    #- Load dataset -#
    X = torch.as_tensor(load_dataset_file(config, config_path + "dataset.npy"), device=device, dtype=dtype)


    #- Embedding -#
    