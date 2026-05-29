import numpy as np
import os
import datetime as dt
import pickle

from qiskit_ibm_runtime import EstimatorV2 as EstimatorV2_rh, QiskitRuntimeService, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as EstimatorV2_sim, SamplerV2 as SamplerV2_sim

from qiskit_ibm_runtime import EstimatorV2 as EstimatorV2_rh, QiskitRuntimeService, Session



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


# Get backend options dict
def get_sim_options(config, execution_type, precision):
    sim_backend_options = config['backend_options']['sim_backend_options']
    
    sim_options = {
        'method': sim_backend_options['noisy_sim_method'] if execution_type in ['noisy', 'real'] else sim_backend_options['noiseless_sim_method'],
        'precision': sim_backend_options['data_type'],
        'shots': get_shots(precision),
        'seed_simulator': config['implementation_options']['seed'],
    }

    if config['implementation_options']['device'] == "GPU":
        sim_options['device'] = 'GPU'
        sim_options.update(sim_backend_options['gpu_device_options'])

    return sim_options


# Get backend simulation dict
def get_sim_backend_dict(config, execution_type, precision):
    sim_options = get_sim_options(config, execution_type, precision)

    if execution_type == "noisy" and config['backend_options']['reset_backend']:
        # Get real backend info
        real_backend_options = config['backend_options']['real_backend_options']
        service = QiskitRuntimeService(channel=real_backend_options['channel'])
        real_backend = service.backend(real_backend_options['name']) #backend = service.least_busy(min_num_qubits=30)

        backend = AerSimulator.from_backend(real_backend, **sim_options) # Get current backend state

    else:
        backend = AerSimulator(**sim_options)

    backend_dict = {
        'configuration': backend.configuration(),
        'properties': backend.properties(),
        'target': backend.target,
        'options': backend.options,
        'noise_model': backend.options.noise_model
    }
    
    return backend_dict


# Save backend file
def create_backend_file(config, filename):
    backend_dict = {
        'timestamp': dt.datetime.now(dt.timezone.utc),
        'train_backend_dict': get_sim_backend_dict(
            config,
            config['implementation_options']['execution_type'], 
            config['backend_options']['train_backend_options']['train_precision']
            ),
        'eval_backend_dict': get_sim_backend_dict(
            config, 
            config['backend_options']['eval_backend_options']['eval_execution_type'], 
            config['backend_options']['eval_backend_options']['eval_precision']
            )
    }

    with open(filename, "wb") as f:
        pickle.dump(backend_dict, f)

    print("Backend file created.")


# Load backend file (just for simulation)
def load_backend_file(config, filename):
    if config['backend_options']['reset_backend'] or not os.path.exists(filename):
        create_backend_file(config, filename)

    with open(filename, "rb") as f:
            backend_dict = pickle.load(f)

    return backend_dict



# Create backend for real hardware
def create_real_backend(real_backend_options):
    # Get real backend
    service = QiskitRuntimeService(channel=real_backend_options['channel'])
    backend = service.backend(real_backend_options['name']) #backend = service.least_busy(min_num_qubits=30)

    # Create session
    session = Session(backend=backend)

    # Create estimator
    estimator = EstimatorV2_rh(mode=session, options=real_backend_options['real_estimator_options'])

    return service, backend, session, estimator


# Create backend for simulation
def create_sim_backend(backend_dict):
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
            "default_precision": get_precision(backend_dict['options']['shots']),
            "backend_options": backend.options,
        })
    
    return backend, estimator


# Load backend options and create backends
def create_backends(config, config_path):
    backend_dict = load_backend_file(config, config_path + "backend.pkl")

    if config['implementation_options']['execution_type'] == "real":
        _, train_backend, session, train_estimator = create_real_backend(config['backend_options']['real_backend_options'])
    else:
        train_backend, train_estimator = create_sim_backend(backend_dict['train_backend_dict'])
        session = None

    eval_backend, eval_estimator = create_sim_backend(backend_dict['eval_backend_dict'])

    # Transpilation method
    train_pm = generate_preset_pass_manager(
        optimization_level=3,
        backend=train_backend,
        seed_transpiler=config['implementation_options']['seed']
    )

    eval_pm = generate_preset_pass_manager(
        optimization_level=3, 
        backend=eval_backend, 
        seed_transpiler=config['implementation_options']['seed'])
    
    return session, train_backend, train_estimator, train_pm, eval_backend, eval_estimator, eval_pm
# TODO en amp eval_estimator no, sampler_estimator
