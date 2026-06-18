import random
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from qgan_v4.training.torch_utils import get_params
from qgan_v4.storage.paths import get_training_data_filename


@dataclass
class TrainingMetrics:
    gloss: dict[int, float] = field(default_factory=dict)
    dloss: dict[int, float] = field(default_factory=dict)
    eval: dict[int, float] = field(default_factory=dict)
    times: dict[int, float] = field(default_factory=dict)

    def best_eval(self):
        values = list(self.eval.values())
        return min(values) if values else float("inf")


@dataclass
class TrainingState:
    config: dict[str, Any]
    current_epoch: int
    metrics: TrainingMetrics
    best_gen_params: Any = None
    init_gen_params: Any = None
    model_g_state: Any = None
    model_d_state: Any = None
    optimizer_g_state: Any = None
    optimizer_d_state: Any = None
    random_state: Any = None
    np_random_state: Any = None
    torch_rng_state: Any = None
    torch_cuda_rng_state: Any = None


#- Random state -#

# Set all training random seeds
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Get current random state
def get_random_state():
    state = TrainingState(
        config={},
        current_epoch=0,
        metrics=TrainingMetrics(),
        random_state=random.getstate(),
        np_random_state=np.random.get_state(),
        torch_rng_state=torch.get_rng_state(),
    )
    if torch.cuda.is_available():
        state.torch_cuda_rng_state = torch.cuda.get_rng_state_all()

    return state


# Restore saved random state
def set_random_state(state):
    random.setstate(state.random_state)
    np.random.set_state(state.np_random_state)
    torch.set_rng_state(state.torch_rng_state.cpu())

    if state.torch_cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda_rng_state)


# Update random state
def copy_random_state(source, target):
    target.random_state = source.random_state
    target.np_random_state = source.np_random_state
    target.torch_rng_state = source.torch_rng_state
    target.torch_cuda_rng_state = source.torch_cuda_rng_state
    return target


#- Model initialization -#

# Initialize model parameters
def initialize_model_params(model):
    params = get_params(model)
    init_params = np.random.uniform(low=-np.pi, high=np.pi, size=(params.numel(),)) * 0.1 # Near 0 params for smooth gradients at start
    init_tensor = torch.as_tensor(init_params, device=params.device, dtype=params.dtype)
    torch.nn.utils.vector_to_parameters(init_tensor, model.parameters())
    return init_params


# Save current generator parameters as the best parameters
def save_best_gen_params(state):
    state.best_gen_params = get_params(state.model_g).cpu().numpy().copy()
    return state.best_gen_params


# Create generator and discriminator optimizers
def create_optimizers(model_g, model_d):
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=0.005)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.005)
    return optimizer_g, optimizer_d


#- Training data save/load -#

# Save training data file
def save_training_data_file(filename, state):
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)


# Load training data file
def load_training_data_file(filename):
    return torch.load(filename, weights_only=False, map_location="cpu")


# Find changed config values
def find_config_changes(previous_config, current_config, path=None):
    path = path or []

    if isinstance(previous_config, dict) and isinstance(current_config, dict):
        changes = []
        for key in sorted(previous_config.keys() | current_config.keys()):
            changes.extend(
                find_config_changes(
                    previous_config.get(key),
                    current_config.get(key),
                    path + [key],
                )
            )
        return changes

    if previous_config != current_config:
        return [(".".join(path), previous_config, current_config)]

    return []


# Warn if saved training config differs from current config
def warn_config_changes(previous_config, current_config):
    changes = find_config_changes(previous_config, current_config)
    if not changes:
        return

    details = "\n".join(
        f"- {path}: previous={previous!r}, current={current!r}"
        for path, previous, current in changes
    )
    warnings.warn(
        "Training data config differs from current config:\n" + details,
        stacklevel=2,
    )


# Create initial training data
def create_training_data(seed, config, model_g, model_d):
    set_random_seed(seed)

    init_gen_params = initialize_model_params(model_g)
    initialize_model_params(model_d)

    optimizer_g, optimizer_d = create_optimizers(model_g, model_d)
    training_state = TrainingState(
        config=config,
        current_epoch=0,
        metrics=TrainingMetrics(),
        best_gen_params=init_gen_params,
        init_gen_params=init_gen_params,
        model_g_state=model_g.state_dict(),
        model_d_state=model_d.state_dict(),
        optimizer_g_state=optimizer_g.state_dict(),
        optimizer_d_state=optimizer_d.state_dict(),
    )
    copy_random_state(get_random_state(), training_state)

    return training_state, optimizer_g, optimizer_d


# Load training data
def load_training_data(filename, config, model_g, model_d):
    training_state = load_training_data_file(filename)
    warn_config_changes(training_state.config, config)
    training_state.config = config

    model_g.load_state_dict(training_state.model_g_state)
    model_d.load_state_dict(training_state.model_d_state)

    set_random_state(training_state)

    optimizer_g, optimizer_d = create_optimizers(model_g, model_d)
    optimizer_g.load_state_dict(training_state.optimizer_g_state)
    optimizer_d.load_state_dict(training_state.optimizer_d_state)

    return training_state, optimizer_g, optimizer_d


# Load existing training data or create fresh data
def load_or_create_training_data(filename, seed, config, model_g, model_d):
    if filename.exists() and not config["training"].get("reset_data", False):
        return load_training_data(filename, config, model_g, model_d)

    training_state, optimizer_g, optimizer_d = create_training_data(seed, config, model_g, model_d)
    save_training_data_file(filename, training_state)
    print("Training data file created.")
    return training_state, optimizer_g, optimizer_d


# Save final training data
def save_checkpoint(state, epoch):
    training_state = TrainingState(
        config=state.config,
        current_epoch=epoch + 1,
        metrics=state.metrics,
        best_gen_params=state.best_gen_params,
        init_gen_params=state.init_gen_params,
        model_g_state=state.model_g.state_dict(),
        model_d_state=state.model_d.state_dict(),
        optimizer_g_state=state.optimizer_g.state_dict(),
        optimizer_d_state=state.optimizer_d.state_dict(),
    )
    copy_random_state(get_random_state(), training_state)
    save_training_data_file(get_training_data_filename(state.config), training_state)
