import numpy as np
import torch

from qgan_v4.execution.backend import create_backends
from qgan_v4.models.packed_circuits import (
    get_gradient_method,
    packed_gradients,
    packed_values,
    parameter_values,
    prepare_angle_disc_job,
    prepare_angle_real_job,
    prepare_direct_disc_job,
    prepare_gen_job,
    prepare_fixed_real_job,
)
from qgan_v4.models.qml_torch2 import generate_eval_g, generate_eval_model
from qgan_v4.models.qnn import compose_circuits, get_observables


# Convert Torch inputs to NumPy values for Qiskit primitives
def to_numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


# Create NumPy values and a CPU Torch parameter sharing the same memory
def new_weight(size):
    values = np.zeros(size, dtype=float)
    return values, torch.nn.Parameter(torch.from_numpy(values))


#- Torch models -#

# Generator model backed by one packed primitive job
class Generator(torch.nn.Module):
    # Store the packed job and create generator weights
    def __init__(self, estimator, gradient, gen_job):
        super().__init__()
        self.estimator = estimator
        self.gradient = gradient
        self.gen_job = gen_job
        self.weight_values, self.weight = new_weight(len(gen_job["gen_params"]))

    # Return generator outputs and circuit gradients
    def forward(self, random_inputs, disc_values):
        parameter_rows = parameter_values(
            self.gen_job,
            disc_values=to_numpy(disc_values),
            gen_values=self.weight_values,
            input_values=to_numpy(random_inputs),
        )
        values = packed_values(self.estimator, self.gen_job, parameter_rows)
        circuit_gradients = packed_gradients(
            self.gradient,
            self.gen_job,
            parameter_rows,
            "gen",
        )
        output = torch.as_tensor(values, dtype=self.weight.dtype).requires_grad_()
        return output, circuit_gradients


# Discriminator model that executes real and fake jobs separately
class SeparateDiscriminator(torch.nn.Module):
    # Store separate discriminator jobs and create shared weights
    def __init__(
        self,
        encoding,
        estimator,
        gradient,
        pass_manager,
        fake_disc_job,
        real_disc_job,
        discriminator,
        real_circuits,
    ):
        super().__init__()
        self.encoding = encoding
        self.estimator = estimator
        self.gradient = gradient
        self.pass_manager = pass_manager
        self.fake_disc_job = fake_disc_job
        self.real_disc_job = real_disc_job
        self.discriminator = discriminator
        self.real_circuits = real_circuits
        self.weight_values, self.weight = new_weight(len(fake_disc_job["disc_params"]))

    # Number of fixed real circuits available
    @property
    def num_real_models(self):
        return len(self.real_circuits)

    # Execute the angle or fixed real discriminator job
    def real(self, real_inputs):
        if self.encoding == "angle":
            real_disc_job = self.real_disc_job
            parameter_rows = parameter_values(
                real_disc_job,
                disc_values=self.weight_values,
                input_values=real_inputs,
            )
        else:
            real_disc_job = prepare_fixed_real_job(
                self.real_circuits,
                real_inputs.astype(int),
                self.discriminator,
                self.pass_manager,
            )
            parameter_rows = parameter_values(
                real_disc_job,
                disc_values=self.weight_values,
            )

        values = packed_values(self.estimator, real_disc_job, parameter_rows)
        circuit_gradients = packed_gradients(
            self.gradient,
            real_disc_job,
            parameter_rows,
            "disc",
        )
        return values, circuit_gradients

    # Execute the shared generator and fake discriminator job
    def fake(self, random_inputs, gen_values):
        parameter_rows = parameter_values(
            self.fake_disc_job,
            disc_values=self.weight_values,
            gen_values=gen_values,
            input_values=random_inputs,
        )
        values = packed_values(self.estimator, self.fake_disc_job, parameter_rows)
        circuit_gradients = packed_gradients(
            self.gradient,
            self.fake_disc_job,
            parameter_rows,
            "disc",
        )
        return values, circuit_gradients

    # Return separate real and fake outputs and gradients
    def forward(self, real_input, random_inputs, gen_values):
        real_values, real_gradients = self.real(to_numpy(real_input))
        fake_values, fake_gradients = self.fake(
            to_numpy(random_inputs),
            to_numpy(gen_values),
        )
        real_output = torch.as_tensor(real_values, dtype=self.weight.dtype).requires_grad_()
        fake_output = torch.as_tensor(fake_values, dtype=self.weight.dtype).requires_grad_()
        return real_output, fake_output, real_gradients, fake_gradients


# Joined discriminator model for direct-circuit encoding
class DirectJoinedDiscriminator(torch.nn.Module):
    # Store one joined job per fixed real circuit
    def __init__(self, estimator, gradient, disc_jobs):
        super().__init__()
        self.estimator = estimator
        self.gradient = gradient
        self.disc_jobs = disc_jobs
        self.weight_values, self.weight = new_weight(len(disc_jobs[0]["disc_params"]))

    # Number of joined real circuit jobs available
    @property
    def num_real_models(self):
        return len(self.disc_jobs)

    # Execute one real and fake discriminator circuit together
    def forward(self, real_input, random_inputs, gen_values):
        real_index = int(to_numpy(real_input).reshape(-1)[0])
        disc_job = self.disc_jobs[real_index]
        parameter_rows = parameter_values(
            disc_job,
            disc_values=self.weight_values,
            gen_values=to_numpy(gen_values),
            input_values=to_numpy(random_inputs),
        )
        values = packed_values(self.estimator, disc_job, parameter_rows)
        circuit_gradients = packed_gradients(
            self.gradient,
            disc_job,
            parameter_rows,
            "disc",
        )
        real_output = torch.as_tensor(values[:1], dtype=self.weight.dtype).requires_grad_()
        fake_output = torch.as_tensor(values[1:], dtype=self.weight.dtype).requires_grad_()
        return real_output, fake_output, circuit_gradients[:1], circuit_gradients[1:]


# Joined discriminator model for angle encoding
class AngleJoinedDiscriminator(torch.nn.Module):
    # Store one half-real half-fake packed discriminator job
    def __init__(self, estimator, gradient, disc_job):
        super().__init__()
        self.estimator = estimator
        self.gradient = gradient
        self.disc_job = disc_job
        self.half_batch = disc_job["batch_size"] // 2
        self.weight_values, self.weight = new_weight(len(disc_job["disc_params"]))

    # Execute joined angle branches and split their outputs and gradients
    def forward(self, real_inputs, random_inputs, gen_values):
        input_values = [
            *to_numpy(real_inputs)[:self.half_batch],
            *to_numpy(random_inputs)[:self.half_batch],
        ]
        parameter_rows = parameter_values(
            self.disc_job,
            disc_values=self.weight_values,
            gen_values=to_numpy(gen_values),
            input_values=input_values,
        )
        values = packed_values(self.estimator, self.disc_job, parameter_rows)
        circuit_gradients = packed_gradients(
            self.gradient,
            self.disc_job,
            parameter_rows,
            "disc",
        )
        real_output = torch.as_tensor(
            values[:self.half_batch],
            dtype=self.weight.dtype,
        ).requires_grad_()
        fake_output = torch.as_tensor(
            values[self.half_batch:],
            dtype=self.weight.dtype,
        ).requires_grad_()
        return (
            real_output,
            fake_output,
            circuit_gradients[:self.half_batch],
            circuit_gradients[self.half_batch:],
        )


#- Model creation -#

# Create the separate discriminator implementation for one encoding
def create_separate_discriminator(
    encoding,
    estimator,
    gradient,
    pass_manager,
    gen_job,
    discriminator,
    real_circuits,
    batch_size,
):
    real_disc_job = None
    if encoding == "angle":
        real_disc_job = prepare_angle_real_job(
            real_circuits[0],
            discriminator,
            batch_size,
            pass_manager,
        )

    return SeparateDiscriminator(
        encoding,
        estimator,
        gradient,
        pass_manager,
        gen_job,
        real_disc_job,
        discriminator,
        real_circuits,
    )


# Create the joined discriminator implementation for one encoding
def create_joined_discriminator(
    encoding,
    estimator,
    gradient,
    pass_manager,
    randomizer,
    generator,
    discriminator,
    real_circuits,
    batch_size,
):
    if encoding == "direct_circuit":
        disc_jobs = [
            prepare_direct_disc_job(
                randomizer,
                generator,
                discriminator,
                real_circuit,
                pass_manager,
            )
            for real_circuit in real_circuits
        ]
        return DirectJoinedDiscriminator(estimator, gradient, disc_jobs)

    if encoding == "angle":
        disc_job = prepare_angle_disc_job(
            randomizer,
            generator,
            discriminator,
            real_circuits[0],
            batch_size,
            pass_manager,
        )
        return AngleJoinedDiscriminator(estimator, gradient, disc_job)

    raise ValueError("Joined discriminator packing does not support amplitude encoding.")


# Create runtime_packed training and evaluation models
def generate_models(config, circuit_bundle):
    execution_type = config["experiment"]["execution_type"]
    if execution_type not in {"noisy", "fake_real", "real"}:
        raise ValueError("runtime_packed is only for execution_type: noisy, fake_real, real.")

    encoding = config["encoding"]["type"]
    session, _, estimator, pass_manager, eval_backend, eval_estimator, eval_pm = create_backends(config)
    generator, discriminator, randomizer, real_circuits = circuit_bundle
    batch_size = config["encoding"]["batch_size"]
    gradient = get_gradient_method(config, estimator)
    discriminator_packing = config["implementation"]["discriminator_packing"]

    gen_job = prepare_gen_job(
        randomizer,
        generator,
        discriminator,
        batch_size,
        pass_manager,
    )
    model_g = Generator(estimator, gradient, gen_job)

    if discriminator_packing == "separate":
        model_d = create_separate_discriminator(
            encoding,
            estimator,
            gradient,
            pass_manager,
            gen_job,
            discriminator,
            real_circuits,
            batch_size,
        )
    else:
        model_d = create_joined_discriminator(
            encoding,
            estimator,
            gradient,
            pass_manager,
            randomizer,
            generator,
            discriminator,
            real_circuits,
            batch_size,
        )

    ran_gen = compose_circuits(randomizer, generator)
    _, obs_eval = get_observables(ran_gen.num_qubits)
    eval_model = generate_eval_model(
        encoding,
        ran_gen,
        obs_eval,
        eval_estimator,
        eval_pm,
    )
    eval_g = generate_eval_g(encoding, eval_model, eval_backend)

    return model_g, model_d, eval_g, session
