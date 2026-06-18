from dataclasses import dataclass
from typing import Any

import torch

from qgan_v4.circuits.factory import get_circuits
from qgan_v4.datasets.images import get_images_dataset
from qgan_v4.evaluation.torch_metrics import batch_evaluation, get_evaluation_function, get_target_probs
from qgan_v4.implementations.base import QGANImplementation
from qgan_v4.models.qml_torch2 import generate_models
from qgan_v4.storage.paths import get_training_data_filename
from qgan_v4.training.batch_torch import (
    generate_model_random_input,
    generate_real_disc_input,
    get_num_random_params,
)
from qgan_v4.training.data import TrainingState, load_or_create_training_data
from qgan_v4.training.torch_utils import get_device, get_dtype, move_models


@dataclass
class QMLTorchState(TrainingState):
    model_g: Any = None
    model_d: Any = None
    eval_g: Any = None
    optimizer_g: Any = None
    optimizer_d: Any = None
    x_data: Any = None
    target_probs: Any = None
    num_random_params: int = 0
    device: Any = None
    dtype: Any = None


class QMLTorchImplementation(QGANImplementation):
    """Qiskit Machine Learning + Torch implementation."""

    name = "qml_torch"

    def setup(self, config):
        eval_method = config["encoding"]["eval_method"]

        device = get_device(config["run"]["device"])
        dtype = get_dtype(config["backend"]["simulator"]["data_type"])
        training_data_file = get_training_data_filename(config)

        circuit_bundle = get_circuits(config)
        model_g, model_d, eval_g = generate_models(config, circuit_bundle)
        move_models(model_g, model_d, eval_g, device, dtype)

        num_random_params = get_num_random_params(eval_g)
        training_state, optimizer_g, optimizer_d = load_or_create_training_data(
            training_data_file,
            config["run"]["seed"],
            config,
            model_g,
            model_d,
        )

        x_data = None
        if config["dataset"]["type"] == "classical":
            x_data = get_images_dataset(config)
        if x_data is not None:
            x_data = torch.as_tensor(x_data, device=device, dtype=dtype)

        _, compute_targets = get_evaluation_function(eval_method)
        if compute_targets:
            target_probs = get_target_probs(
                config["dataset"]["type"],
                config["encoding"]["contrast"],
                device,
                dtype,
                real_circuits=circuit_bundle[3],
                X=x_data,
            )
        else:
            target_probs = None

        return QMLTorchState(
            config=config,
            current_epoch=training_state.current_epoch,
            metrics=training_state.metrics,
            best_gen_params=training_state.best_gen_params,
            model_g=model_g,
            model_d=model_d,
            eval_g=eval_g,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            init_gen_params=training_state.init_gen_params,
            x_data=x_data,
            target_probs=target_probs,
            num_random_params=num_random_params,
            device=device,
            dtype=dtype,
        )
    
    def loss(self, x, label):
        return (-x * label).mean()

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
        fake_inputs = generate_model_random_input(
            state.model_g,
            config["encoding"]["batch_size"],
            state.num_random_params,
            config["encoding"]["randomness"],
            state.device,
            state.dtype,
        )
        real_output, fake_output = state.model_d(real_inputs, fake_inputs)
        real_loss = self.loss(real_output, torch.ones_like(real_output))
        fake_loss = self.loss(fake_output, -torch.ones_like(fake_output))

        loss = real_loss + fake_loss
        loss.backward()
        state.optimizer_d.step()

        return (real_loss.item() + fake_loss.item() - 2) / 4

    def train_generator_step(self, state):
        config = state.config
        state.optimizer_g.zero_grad()

        gen_inputs = generate_model_random_input(
            state.model_d,
            config["encoding"]["batch_size"],
            state.num_random_params,
            config["encoding"]["randomness"],
            state.device,
            state.dtype,
        )
        gen_output = state.model_g(gen_inputs)
        gen_loss = self.loss(gen_output, torch.ones_like(gen_output))
        gen_loss.backward()
        state.optimizer_g.step()

        return (gen_loss.item() - 1) / 2

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
