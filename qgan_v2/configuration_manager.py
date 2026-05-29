# %% [markdown]
# Configurations

# %%
import yaml
import copy
import os
import datetime as dt
import numpy as np
from itertools import product

# %%
#- Configuration manager -#

# Generate run id
def generate_run_id(imp_config):
    run_id = (
        f"{imp_config['implementation']}-"
        f"q{imp_config['n_qubits']}-"
        f"{imp_config['execution_type']}-"
        f"{imp_config['device']}-"
        f"{imp_config['gradient_method']}-"
    )
    if imp_config['implementation'] in ['qgan_TorchConnector_ang', 'qgan_TorchConnector_amp']:
        run_id += f"random{imp_config['random_input']}-"
    run_id += f"seed{imp_config['seed']}"

    return run_id

# Get data path
def get_data_path(config):
    # Generate run id if None
    if config['data_management']['run_id'] is None:
        config['data_management']['run_id'] = generate_run_id(config['implementation_options'])
    
    config_path = config['data_management']['data_path'] + config['data_management']['run_id'] + "/"
    return config_path


# Save config file
def create_config_file(config, overwrite=False):
    config_path = get_data_path(config)
    filename = config_path + "config.yaml"

    fileExists = os.path.exists(filename)
    if fileExists and not overwrite:
        print("Configuration file already exists. Path: " + config_path)
        return filename
    
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    
    with open(filename, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    action = "created" if not fileExists else "overwritten"
    print(f"Configuration file {action}. Path: {config_path}")
    return filename



# Load config file
def load_config_file(data_path):
    filename = data_path + "config.yaml"
    
    if not os.path.exists(filename):
        create_config_file(filename)

    with open(filename, "r") as file:
        config = yaml.safe_load(file)

    return config

# %%
#- Configuration manager example -#

# Configuration dict
config = {
    # Implementation options
    'implementation_options': {
        'implementation': "qgan_TorchConnector", # 'qgan_TorchConnector', 'qgan_TorchConnector_ang' or 'qgan_TorchConnector_amp'
        'execution_type': "noiseless", # 'noiseless', 'noisy' or 'real'
        'n_qubits': 4,
        #'gen_circuit': , TODO 1, 2, 3, maybe join gen-disc so they are leveled
        #'disc_circuit': ,
        'device': "CPU", # 'GPU' or 'CPU'
        'gradient_method': "SPSA", # qiskit_algorithms.gradients For now: 'PSR', 'SPSA' and 'REG'
        'seed': 0,
    },

    # Training parameters
    'training_parameters': {
        'max_iterations': 1000,
        'gen_iterations': 1,
        'disc_iterations': 1,
        'print_progress_iterations': 1,
    },

    # Data management
    'data_management': {
        'data_path': "data/", # Path where all files are stored
        'run_id': None, # Execution id, just an identifier. If none, automatically assigned: TODO
        'reset_training_data': False, # Reset training data
        'create_circuits': False, # Create circuits or load from file
        'reset_backend': False, # Get current backend state or load from file
    },

    # Backend options
    'backend_options': {
        'training_precision': 0.03125, # Training precision. 0.03125 ~ 1024 shots
        'data_type': "double", # 'single' or 'double'

        # Real backend options
        'real_backend_options': {
            'name': "ibm_basquecountry", # Real backend name
            'channel': "ibm_quantum_platform", # Real backend channel
            "real_estimator_options": {
                "resilience_level": 1, # Mitigacion basica de errores.
                "dynamical_decoupling": {"enable": True,}, # Inserta pulsos/secuencias entre puertas para reducir decoherencia mientras los qubits están inactivos.
            },
        },

        # Simulation backend options
        'sim_backend_options': {
            'noiseless_sim_method': 'statevector',
            'noisy_sim_method': 'density_matrix',
        },

    },

    # Evaluation options
    'eval_options': {
        'eval_execution_type': 'noiseless', # 'noiseless' or 'noisy'.
        'eval_precision': 0.0,
        #'eval_function': , TODO
    },

    # Device options
    'gpu_device_options': {
        'cuStateVec_enable': True,   # NVIDIA library optimization
        'batched_shots_gpu': True,   # Parallelize batch on GPU
        'blocking_enable': False,     # Disable chunking; simulation fits in VRAM 
        #'target_gpus':[0,1], # TODO import torch after? In .py read how many after printing aviables?

        'runtime_parameter_bind_enable': True, # tells Aer to keep the circuit parameterized and bind the numeric values at execution time TODO prueba pa saber las mejores options
    },

    # Embedding options
    'embedding_options': {
        'reset_dataset': False, 
        'batch_size': 4, # How many samples' gradients are going to be calculated in a step
        'random_input': True, # Add randomness in the input when generating a sample
        'randomness': 0.1 # Multiplier for random values to be less random
    },
}

# Notes
'''
method:
    For noiseless execution:
        - statevector
        - matrix_product_state: more qubits, low entanglement
    For noisy execution:
        - density_matrix: uses 4^n_qubits memory, can simulate noise
        - statevector: uses 2^n_qubits memory, less acurate noise simulation
    - stabilizer: only for Clifford circuits (only H, CNOT, and S gates)
    - tensor_network: for large circuits (when reaching memory limits, only GPU)

gradient:
    For noiseless execution:
    - REG: exact, crece fuerte con n_qubits, suave con n_params
    - PSR: exact, 2*n_params executions
    - SPSA: aproximado, 2 executions
    For noisy execution (shot based):
    - REG: not supported
    - PSR: exact + noise
    - SPSA: aproximado + noise, se puede aumentar shots para reducir ruido
'''

'''
# More options
backend_for_info = AerSimulator()
print("AerSimulator backend configuration options:")
for option in backend_for_info.options:
    print(" -", option)

'''


# Create config file
create_config_file(config, overwrite=True)


# Load config file
config = load_config_file(get_data_path(config))


print(config)

# %%
#- Create all parameter-combination configuration files -#

# Change these defaults to choose the shared configuration values.
DEFAULT_CONFIG_VALUES = {
    'implementation_options': {
        'n_qubits': 3,
        'seed': 0,
    },
    'training_parameters': {
        'max_iterations': 2,
        'gen_iterations': 1,
        'disc_iterations': 1,
        'print_progress_iterations': 1,
    },
    'data_management': {
        'data_path': "data/test/",
        'run_id': None,
        'reset_training_data': True,
        'create_circuits': True,
        'reset_backend': True,
    },
    'backend_options': {
        'training_precision': 0.5,
        'data_type': "double",
        'real_backend_options': {
            'name': "ibm_basquecountry",
            'channel': "ibm_quantum_platform",
            'reset_backend': False,
            'real_estimator_options': {
                'resilience_level': 1,
                'dynamical_decoupling': {'enable': True},
            },
        },
        'sim_backend_options': {
            'noiseless_sim_method': 'statevector',
            'noisy_sim_method': 'statevector', # 'density_matrix' too expensive TODO
        },
    },
    'eval_options': {
        'eval_execution_type': 'noiseless',
        'eval_precision': 0.0,
    },
    'gpu_device_options': {
        'cuStateVec_enable': True,
        'batched_shots_gpu': True,
        'blocking_enable': False,
        'runtime_parameter_bind_enable': True,
    },
    'embedding_options': {
        'reset_dataset': True,
        'batch_size': 2,
        'randomness': 1,
    },
}

PARAMETER_OPTIONS = {
    'implementation': ("qgan_TorchConnector", "qgan_TorchConnector_ang", "qgan_TorchConnector_amp"),
    'execution_type': ("noiseless", "noisy"), # "real" TODO
    'device': ("CPU","GPU"), # TODO gpu
    'gradient_method': ("SPSA",), # "REG" solo noiseless, 'PSR' muy lento en noisy TODO
    'random_input': (True, False),
}

RANDOM_INPUT_IMPLEMENTATIONS = ("qgan_TorchConnector_ang", "qgan_TorchConnector_amp")
OVERWRITE_CONFIG_FILES = False


def build_config_from_options(implementation, execution_type, device, gradient_method, random_input=None, n_qubits=None):
    conf = copy.deepcopy(DEFAULT_CONFIG_VALUES)
    implementation_options = conf['implementation_options']
    implementation_options.update({
        'implementation': implementation,
        'execution_type': execution_type,
        'device': device,
        'gradient_method': gradient_method,
    })
    if n_qubits is not None:
        implementation_options['n_qubits'] = n_qubits
    if implementation in RANDOM_INPUT_IMPLEMENTATIONS:
        implementation_options['random_input'] = random_input
        conf['embedding_options']['random_input'] = random_input
    return conf


created_parameter_combination_files = []
base_options = product(
    PARAMETER_OPTIONS['implementation'],
    PARAMETER_OPTIONS['execution_type'],
    PARAMETER_OPTIONS['device'],
    PARAMETER_OPTIONS['gradient_method'],
)

for implementation, execution_type, device, gradient_method in base_options:
    random_inputs = PARAMETER_OPTIONS['random_input'] if implementation in RANDOM_INPUT_IMPLEMENTATIONS else (None,)
    for random_input in random_inputs:
        conf = build_config_from_options(
            implementation=implementation,
            execution_type=execution_type,
            device=device,
            gradient_method=gradient_method,
            random_input=random_input,
        )
        create_config_file(conf, overwrite=OVERWRITE_CONFIG_FILES)
        created_parameter_combination_files.append(get_data_path(conf) + "config.yaml")

print(f"{len(created_parameter_combination_files)} parameter-combination configuration files created.")


# %%
#- Create battery of experiments' configuration files -#

# Change these defaults to choose the shared configuration values.
DEFAULT_CONFIG_VALUES = {
    'implementation_options': {
        'seed': 0,
        'gradient_method': "SPSA",
    },
    'training_parameters': {
        'max_iterations': 1000,
        'gen_iterations': 1,
        'disc_iterations': 1,
        'print_progress_iterations': 10,
    },
    'data_management': {
        'data_path': "data/",
        'run_id': None,
        'reset_training_data': False,
        'create_circuits': False,
        'reset_backend': False,
    },
    'backend_options': {
        'training_precision': 0.03125,
        'data_type': "double",
        'real_backend_options': {
            'name': "ibm_basquecountry",
            'channel': "ibm_quantum_platform",
            'reset_backend': False,
            'real_estimator_options': {
                'resilience_level': 1,
                'dynamical_decoupling': {'enable': True},
            },
        },
        'sim_backend_options': {
            'noiseless_sim_method': 'statevector',
            'noisy_sim_method': 'density_matrix',
        },
    },
    'eval_options': {
        'eval_execution_type': 'noiseless',
        'eval_precision': 0.0,
    },
    'gpu_device_options': {
        'cuStateVec_enable': True,
        'batched_shots_gpu': True,
        'blocking_enable': False,
        'runtime_parameter_bind_enable': True,
    },
    'embedding_options': {
        'reset_dataset': False,
        'batch_size': 4,
        'randomness': 1,
        'random_input': True,
    },
}

PARAMETER_OPTIONS = {
    'n_qubits': (4, 8, 16),
    'implementation': ("qgan_TorchConnector", "qgan_TorchConnector_ang", "qgan_TorchConnector_amp"),
    'execution_type': ("noiseless", "noisy"), # "real" TODO
    'device': ("CPU", "GPU"),
    'random_input': (True, False),
}

RANDOM_INPUT_IMPLEMENTATIONS = ("qgan_TorchConnector_ang", "qgan_TorchConnector_amp")
OVERWRITE_CONFIG_FILES = False


def build_config_from_options(implementation, execution_type, device, n_qubits, random_input=None):
    conf = copy.deepcopy(DEFAULT_CONFIG_VALUES)
    implementation_options = conf['implementation_options']
    implementation_options.update({
        'implementation': implementation,
        'execution_type': execution_type,
        'n_qubits': n_qubits,
        'device': device,
    })
    if implementation in RANDOM_INPUT_IMPLEMENTATIONS:
        implementation_options['random_input'] = random_input
        conf['embedding_options']['random_input'] = random_input
    return conf


created_parameter_combination_files = []
base_options = product(
    PARAMETER_OPTIONS['implementation'],
    PARAMETER_OPTIONS['execution_type'],
    PARAMETER_OPTIONS['device'],
    PARAMETER_OPTIONS['n_qubits'],
)

for implementation, execution_type, device, n_qubits in base_options:
    random_inputs = PARAMETER_OPTIONS['random_input'] if implementation in RANDOM_INPUT_IMPLEMENTATIONS else (None,)
    for random_input in random_inputs:
        conf = build_config_from_options(
            implementation=implementation,
            execution_type=execution_type,
            device=device,
            n_qubits=n_qubits,
            random_input=random_input,
        )
        create_config_file(conf, overwrite=OVERWRITE_CONFIG_FILES)
        created_parameter_combination_files.append(get_data_path(conf) + "config.yaml")

print(f"{len(created_parameter_combination_files)} parameter-combination configuration files created.")


# %%
#- Execute configuration files -#

from pathlib import Path
import subprocess
import sys


CONFIG_ROOT_PATH = "data/test"
IMPLEMENTATION_SCRIPTS = {
    "qgan_TorchConnector": Path("qgan_TorchConnector.py"),
    "qgan_TorchConnector_ang": Path("qgan_TorchConnector_ang.py"),
    "qgan_TorchConnector_amp": Path("qgan_TorchConnector_amp.py"),
}
STOP_ON_ERROR = True
DRY_RUN = False # to print commands without executing


def execute_configs(data_path):
    data_path = Path(data_path)

    executed_configs = []
    for run_dir in sorted(path for path in data_path.iterdir() if path.is_dir()):
        config_path = run_dir / "config.yaml"

        with open(config_path, "r") as file:
            config = yaml.safe_load(file) or {}

        implementation = config.get('implementation_options', {}).get('implementation')
        script_path = IMPLEMENTATION_SCRIPTS.get(implementation)
        if script_path is None:
            raise ValueError(f"Unknown implementation '{implementation}' in {config_path}")

        command = [sys.executable, str(script_path), "-c", str(config_path)]
        print("Executing:", " ".join(command))

        if not DRY_RUN:
            result = subprocess.run(command)
            if result.returncode != 0 and STOP_ON_ERROR:
                raise RuntimeError(f"Command failed with return code {result.returncode}: {command}")

        executed_configs.append(str(config_path))

    print(f"{len(executed_configs)} configuration files executed.")
    return executed_configs


executed_configs = execute_configs(CONFIG_ROOT_PATH)



