from pathlib import Path


#- File names -#

CONFIG_FILENAME = "config.yaml"
TRAINING_DATA_FILENAME = "training_data.pth"
CIRCUITS_FILENAME = "circuits.qpy"
RUN_BACKEND_FILENAME = "backend.pkl"
DATASET_FILE_EXTENSION = ".npz"
REAL_BACKEND_FILE_EXTENSION = ".pkl"



#- Root paths -#

# Get qgan_v4 project root
def get_qgan_v4_root():
    return Path(__file__).resolve().parents[3]


# Get repository root
def get_repository_root():
    return get_qgan_v4_root().parent



#- Shared storage paths -#

# Get datasets directory
def get_datasets_path():
    return get_qgan_v4_root() / "datasets"


# Get prepared datasets directory
def get_prepared_datasets_path():
    return get_datasets_path() / "prepared"


# Get cached backend info directory
def get_backends_path():
    return get_qgan_v4_root() / "backends"



#- Run storage paths -#

# Resolve absolute, repository-relative, or qgan_v4-relative run data path
def resolve_data_path(data_path):
    path = Path(data_path)
    if path.is_absolute():
        return path

    qgan_v4_root = get_qgan_v4_root()
    if path.parts and path.parts[0] == qgan_v4_root.name:
        return get_repository_root() / path

    return qgan_v4_root / path


# Get run directory
def get_run_path(config):
    return resolve_data_path(config["run"]["data_path"]) / config["run"]["id"]


# Get run config file path
def get_config_filename(config):
    return get_run_path(config) / CONFIG_FILENAME


# Get training data file path
def get_training_data_filename(config):
    return get_run_path(config) / TRAINING_DATA_FILENAME


# Get circuits file path
def get_circuits_filename(config):
    return get_run_path(config) / CIRCUITS_FILENAME


# Get run backend file path
def get_run_backend_filename(config):
    return get_run_path(config) / RUN_BACKEND_FILENAME


#- Shared file paths -#

# Get prepared dataset file path
def get_prepared_dataset_filename(config):
    dataset_id = config["dataset"]["id"]
    return get_prepared_datasets_path() / f"{dataset_id}{DATASET_FILE_EXTENSION}"


# Get real backend info file path
def get_real_backend_filename(config):
    backend_id = config["backend"]["real"]["id"]
    return get_backends_path() / f"{backend_id}{REAL_BACKEND_FILE_EXTENSION}"
