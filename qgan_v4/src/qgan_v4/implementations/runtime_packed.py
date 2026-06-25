from dataclasses import dataclass
from typing import Any

import torch

from qgan_v4.circuits.factory import get_circuits
from qgan_v4.datasets.images import get_images_dataset
from qgan_v4.evaluation.torch_metrics import batch_evaluation, get_evaluation_function, get_target_probs
from qgan_v4.implementations.base import QGANImplementation
from qgan_v4.models.packed_circuits import combine_gradients
from qgan_v4.models.runtime_packed import generate_models, to_numpy
from qgan_v4.storage.paths import get_training_data_filename
from qgan_v4.training.batch_torch import (
    generate_random_input,
    generate_real_disc_input,
    get_num_random_params,
)
from qgan_v4.training.data import TrainingState, load_or_create_training_data
from qgan_v4.training.torch_utils import get_dtype


#- Training state -#

# Runtime packed training state
@dataclass
class RuntimePackedState(TrainingState):
    model_g: Any = None
    model_d: Any = None
    eval_g: Any = None
    optimizer_g: Any = None
    optimizer_d: Any = None
    session: Any = None
    x_data: Any = None
    target_probs: Any = None
    num_random_params: int = 0
    device: Any = None
    dtype: Any = None


#- Training implementation -#

# Runtime primitive implementation with manual Torch gradient assignment
class RuntimePackedImplementation(QGANImplementation):
    name = "runtime_packed"

    # Create models, optimizers and training data
    def setup(self, config):
        return setup_runtime_packed(config)

    # Wasserstein-style signed output loss
    def loss(self, outputs, labels):
        return (-outputs * labels).mean()

    # Combine primitive gradients with loss gradients and assign Torch gradients
    def assign_gradient(self, weight, *terms):
        gradient = sum(
            combine_gradients(circuit_gradients, to_numpy(loss_gradients))
            for circuit_gradients, loss_gradients in terms
        )
        weight.grad = torch.as_tensor(gradient, dtype=weight.dtype)

    # Run one discriminator optimizer step
    def train_discriminator_step(self, state):
        config = state.config
        state.optimizer_d.zero_grad()

        real_inputs = generate_real_disc_input(
            config["encoding"]["type"],
            state.x_data,
            state.model_d,
            config["encoding"]["batch_size"],
            state.device,
        )
        random_inputs = generate_random_input(
            config["encoding"]["batch_size"],
            state.num_random_params,
            config["encoding"]["randomness"],
            state.device,
            state.dtype,
        )
        real_output, fake_output, real_gradients, fake_gradients = state.model_d(
            real_inputs,
            random_inputs,
            state.model_g.weight_values,
        )
        real_labels = torch.ones_like(real_output)
        fake_labels = -torch.ones_like(fake_output)
        real_loss = self.loss(real_output, real_labels)
        fake_loss = self.loss(fake_output, fake_labels)
        real_loss_gradients, fake_loss_gradients = torch.autograd.grad(
            real_loss + fake_loss,
            (real_output, fake_output),
        )

        self.assign_gradient(
            state.model_d.weight,
            (real_gradients, real_loss_gradients),
            (fake_gradients, fake_loss_gradients),
        )
        state.optimizer_d.step()

        return (real_loss.item() + fake_loss.item() - 2) / 4

    # Run one generator optimizer step
    def train_generator_step(self, state):
        config = state.config
        state.optimizer_g.zero_grad()

        random_inputs = generate_random_input(
            config["encoding"]["batch_size"],
            state.num_random_params,
            config["encoding"]["randomness"],
            state.device,
            state.dtype,
        )
        output, circuit_gradients = state.model_g(
            random_inputs,
            state.model_d.weight_values,
        )
        labels = torch.ones_like(output)
        loss = self.loss(output, labels)
        loss_gradients = torch.autograd.grad(loss, output)[0]

        self.assign_gradient(state.model_g.weight, (circuit_gradients, loss_gradients))
        state.optimizer_g.step()

        return (loss.item() - 1) / 2

    # Evaluate the generator with the shared noiseless evaluation model
    def evaluate(self, state):
        config = state.config
        return batch_evaluation(
            config["encoding"]["eval_method"],
            config["encoding"]["randomness"],
            state.eval_g,
            config["encoding"]["eval_batch_size"],
            state.num_random_params,
            state.device,
            state.dtype,
            target_probs=state.target_probs,
            eval_weights=state.model_g.weight,
        )

    # Close the Runtime session when one was created
    def close(self, state):
        if state.session is not None:
            state.session.close()


#- Setup -#

# Create the complete runtime_packed training state
def setup_runtime_packed(config):
    device = torch.device("cpu")
    dtype = get_dtype(config["backend"]["simulator"]["data_type"])
    circuit_bundle = get_circuits(config)
    model_g, model_d, eval_g, session = generate_models(config, circuit_bundle)

    training_state, optimizer_g, optimizer_d = load_or_create_training_data(
        get_training_data_filename(config),
        config["run"]["seed"],
        config,
        model_g,
        model_d,
    )

    x_data = None
    if config["dataset"]["type"] == "classical":
        x_data = torch.as_tensor(get_images_dataset(config), dtype=dtype)

    _, compute_targets = get_evaluation_function(config["encoding"]["eval_method"])
    target_probs = None
    if compute_targets:
        target_probs = get_target_probs(
            config["dataset"]["type"],
            config["encoding"]["contrast"],
            device,
            dtype,
            real_circuits=circuit_bundle[3],
            X=x_data,
        )

    return RuntimePackedState(
        config=config,
        current_epoch=training_state.current_epoch,
        metrics=training_state.metrics,
        best_gen_params=training_state.best_gen_params,
        init_gen_params=training_state.init_gen_params,
        model_g=model_g,
        model_d=model_d,
        eval_g=eval_g,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        session=session,
        x_data=x_data,
        target_probs=target_probs,
        num_random_params=get_num_random_params(eval_g),
        device=device,
        dtype=dtype,
    )
