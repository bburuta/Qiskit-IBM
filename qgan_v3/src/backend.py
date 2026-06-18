import numpy as np
import os
import pickle
import datetime as dt
from pathlib import Path

from qiskit_ibm_runtime import EstimatorV2 as EstimatorV2_rh, QiskitRuntimeService, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as EstimatorV2_sim, SamplerV2 as SamplerV2_sim
from qiskit_aer.noise import NoiseModel

from config_manager.config_manager import get_run_path



#- Create backend -#

# Get shots from precision
def get_shots(precision):
    if np.isclose(precision, 0, atol=1e-8):
        return None
    else:
        return int(1/(precision**2))
    

# Get precision from shots
def get_precision(shots):
    if shots is None or shots <= 0:
        return 0.0
    return 1 / np.sqrt(shots)


# Get train backend options dict
def get_train_sim_options(config):
    sim_backend_options = config['backend']['simulator']
    execution_type = config['experiment']['execution_type']
    precision = config['backend']['precision']
    
    sim_options = {
        'method': sim_backend_options['noisy_method'] if execution_type in ['noisy', 'real'] else sim_backend_options['noiseless_method'],
        'precision': sim_backend_options['data_type'],
        'shots': get_shots(precision),
        'seed_simulator': config['run']['seed'],
    }

    if config['run']['device'] == "GPU":
        sim_options['device'] = 'GPU'
        sim_options.update(sim_backend_options['gpu'])

    return sim_options


# Get evaluation backend options dict
def get_eval_sim_options(config):
    sim_backend_options = config['backend']['simulator']

    sim_options = {
        'method': sim_backend_options['noiseless_method'],
        'precision': sim_backend_options['data_type'],
        'shots': None,
        'seed_simulator': config['run']['seed'],
    }

    if config['run']['device'] == "GPU":
        sim_options['device'] = 'GPU'
        sim_options.update(sim_backend_options['gpu'])

    return sim_options


# # Save account
# QiskitRuntimeService.save_account(
#     token="",
#     instance="crn:v1:bluemix:public:quantum-computing:eu-de:a/cb804b30dfcb48b890393bfd6e41e9c2:4cb40c64-a531-4c13-b39c-e04c31185259::",
#     set_as_default = True,
#     overwrite=True
# )


# Create real backend
def create_real_backend(real_backend_options):
    service = QiskitRuntimeService(channel=real_backend_options['channel'])
    backend = service.backend(real_backend_options['name']) #backend = service.least_busy(min_num_qubits=30)

    return service, backend


# Create noiseless backend
def create_noiseless_backend(sim_options):
    backend = AerSimulator(**sim_options)
    return backend


# Create noisy backend
def create_noisy_backend(sim_options, real_backend_info):
    backend = AerSimulator(
        noise_model=real_backend_info["noise_model"],
        coupling_map=real_backend_info["coupling_map"],
        basis_gates=real_backend_info["basis_gates"],
        **sim_options
    )
    backend._target = real_backend_info["target"]

    return backend


# Get real backend info dict
def get_real_backend_info(real_backend):
    return {
        "name": real_backend.name,
        "target": real_backend.target,
        "configuration": real_backend.configuration(),
        "properties": real_backend.properties(),
        "basis_gates": real_backend.operation_names,
        "coupling_map": real_backend.coupling_map,
        "noise_model": NoiseModel.from_backend(real_backend)
    }


# Create real backend session and estimator
def create_real_estimator(backend, real_estimator_options):
    # Create session
    session = Session(backend=backend)

    # Create estimator for real hardware
    estimator = EstimatorV2_rh(mode=session, options=real_estimator_options)

    return session, estimator


# Create estimator for simulation
def create_sim_estimator(backend):
    estimator = EstimatorV2_sim(
        options = {
            "default_precision": get_precision(backend.options['shots']),
            "backend_options": backend.options,
        })
    
    return estimator


# # Create sampler for simulation
# def create_sim_sampler(backend):
#     sampler = SamplerV2_sim(
#         default_shots = eval_options['shots'],
#         options = {"backend_options": eval_backend.options}
#     )
#     # # Noiseless sampler (not BaseV2, cannot be used for TorchConnector)
#     # sampler = Sampler_sim(
#     #     backend_options = backend.options,
#     #     run_options = {
#     #         'shots': None,
#     #     },
#     #     skip_transpilation=True
#     # )


# Create preset pass manager
def create_pm(backend, seed):
    pm = generate_preset_pass_manager(
        optimization_level=3, 
        backend=backend, 
        seed_transpiler=seed
    )

    return pm


#- Backend save -#

BACKEND_FILENAME = 'backend.pkl'

# Shared real backend info
QGAN_ROOT = Path(__file__).resolve().parents[1]
BACKENDS_PATH = QGAN_ROOT / 'backends'
REAL_BACKEND_FILE_EXTENSION = '.pkl'


# Get backend file path
def get_backend_filename(config):
    if config['run']['id'] is None:
        raise ValueError("run.id must be defined to save backend file")

    return get_run_path(config) / BACKEND_FILENAME


# Get real backend file path
def get_real_backend_filename(config):
    backend_id = config['backend']['real']['id']

    return BACKENDS_PATH / f"{backend_id}{REAL_BACKEND_FILE_EXTENSION}"


# Save backend file
def create_backend_file(backend_dict, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(backend_dict, f)

    print("Backend file created.")


# Load backend file (just for simulation)
def load_backend_file(filename):
    with open(filename, "rb") as f:
            backend_dict = pickle.load(f)

    return backend_dict


# Load or create real backend info
def load_or_create_real_backend_info(config, real_backend=None):
    real_backend_options = config['backend']['real']
    filename = get_real_backend_filename(config)

    if real_backend_options['reset_info'] or not os.path.exists(filename):
        if real_backend is None:
            _, real_backend = create_real_backend(real_backend_options)
        
        real_backend_info = get_real_backend_info(real_backend)
        create_backend_file(real_backend_info, filename)

    return load_backend_file(filename)



#- Backend management -#

# Load backend options and create backends
def create_backends(config, save_real_backend_info=True, save_backend_file=False):
    # Save backend options file
    if save_backend_file:
        filename = get_backend_filename(config)
        
        # Reset backend file
        if config['backend']['reset'] or not os.path.exists(filename):
            backend_dict = {
                'timestamp': dt.datetime.now(dt.timezone.utc),
                'train_backend_dict': get_train_sim_options(config),
                'eval_backend_dict': get_eval_sim_options(config),
            }
            
            create_backend_file(backend_dict, filename)

        # Load backend file
        backend_dict = load_backend_file(filename)
    
    else:
        backend_dict = {
            'timestamp': dt.datetime.now(dt.timezone.utc),
            'train_backend_dict': get_train_sim_options(config),
            'eval_backend_dict': get_eval_sim_options(config),
        }

    # Evaluation is always ideal noiseless simulation
    backend_dict['eval_backend_dict'] = get_eval_sim_options(config)


    # Create train backend
    execution_type = config['experiment']['execution_type']
    real_backend_options = config['backend']['real']
    real_estimator_options = real_backend_options['estimator']

    # Create backend
    if execution_type == "real":
        _, train_backend = create_real_backend(real_backend_options)
        session, train_estimator = create_real_estimator(train_backend, real_estimator_options)

        if save_real_backend_info:
            real_backend_info = load_or_create_real_backend_info(config, train_backend)

    elif execution_type == "noisy":
        # Save real backend info
        if save_real_backend_info:
            real_backend_info = load_or_create_real_backend_info(config)
        else:
            _, real_backend = create_real_backend(real_backend_options)
            real_backend_info = get_real_backend_info(real_backend)

        sim_options = backend_dict['train_backend_dict']

        train_backend = create_noisy_backend(sim_options, real_backend_info)
        session = None
        train_estimator = create_sim_estimator(train_backend)

    elif execution_type == "noiseless":
        sim_options = backend_dict['train_backend_dict']

        train_backend = create_noiseless_backend(sim_options)
        session = None
        train_estimator = create_sim_estimator(train_backend)
    else:
        raise ValueError(f"Unsupported execution_type: {execution_type}")

    # Transpilation method
    train_pm = create_pm(train_backend, config['run']['seed'])


    # Create eval backend
    sim_options = backend_dict['eval_backend_dict']
    eval_backend = create_noiseless_backend(sim_options)
    eval_estimator = create_sim_estimator(eval_backend) # TODO eval backend only noiseless, amp with torch module without SamplerQNN
    eval_pm = create_pm(eval_backend, config['run']['seed'])


    return session, train_backend, train_estimator, train_pm, eval_backend, eval_estimator, eval_pm



#- Main -#

if __name__ == "__main__":
    # Real backend options example
    real_backend_options = {
        'id': "ibm_basquecountry",
        'name': "ibm_basquecountry",
        'channel': "ibm_quantum_platform",
        'reset_info': False,
        'estimator': {
            'resilience_level': 1,
            'dynamical_decoupling': {
                'enable': True
            }
        }
    }

    # Get real backend info
    _, train_backend = create_real_backend(real_backend_options)
    real_backend_info = get_real_backend_info(train_backend)
    
    # Create real backend info file
    config = {'backend': {'real': real_backend_options}}
    real_backend_filename = get_real_backend_filename(config)
    if not os.path.exists(real_backend_filename):
        create_backend_file(real_backend_info, real_backend_filename)

    print(real_backend_info)
