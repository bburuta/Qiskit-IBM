import torch
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from torch.func import functional_call
from qiskit_machine_learning.connectors import TorchConnector

from qgan_v4.execution.backend import create_backends
from qgan_v4.models.qnn import (
    get_composed_circuits,
    get_gradient_method,
    get_observables,
    split_params_by_prefix,
    transpile_train_circuits,
    prepare_eval_circuit_ang,
    prepare_eval_circuit_amp,
    transpile_eval_circuits,
    generate_train_qnns,
    generate_eval_qnn_ang,
)



#- Real discriminator -#

# Base class for real discriminator branches run with external weights
class RealDiscriminator(torch.nn.Module, ABC):
    # Build a torch model and remember its real weight parameter name
    def _make_model(self, qnn):
        model = TorchConnector(qnn)
        weight_name = next(iter(model.named_parameters(recurse=False)))[0]
        return model, weight_name

    # Number of real discriminator circuits available
    @property
    @abstractmethod
    def num_real_models(self):
        pass

    # Return how many inputs the real branch expects
    @abstractmethod
    def get_num_params(self):
        pass

    # Run the real branch using discriminator weights from the fake branch
    @abstractmethod
    def forward(self, real_input, weights):
        pass


# Real discriminator branch for angle encoding
class RealDiscriminatorAngle(RealDiscriminator):
    # Store one real discriminator model
    def __init__(self, disc_real_qnn):
        super().__init__()
        model, self.weight_name = self._make_model(disc_real_qnn)

        # Keep this model unregistered so only the fake branch owns weights
        object.__setattr__(self, "model", model)

    # Angle mode has only one real discriminator model
    @property
    def num_real_models(self):
        return 1

    # Return the number of circuit inputs expected by this model
    def get_num_params(self):
        return self.model.neural_network.num_inputs

    # Run the model with the shared discriminator weights
    def forward(self, real_input, weights):
        return functional_call(
            self.model,
            {self.weight_name: weights},
            (real_input,),
        )


# Real discriminator branch for amplitude and direct-circuit encodings
class RealDiscriminatorAmplitude(RealDiscriminator):
    # Store one real discriminator model per prepared real circuit
    def __init__(self, disc_real_qnns, max_workers=1):
        super().__init__()
        model_specs = [self._make_model(disc_real_qnn) for disc_real_qnn in disc_real_qnns]
        self.models = [model for model, _ in model_specs]
        self.weight_names = [weight_name for _, weight_name in model_specs]
        self.max_workers = max_workers

    # Return how many prepared real circuits can be selected
    @property
    def num_real_models(self):
        return len(self.models)

    # Return the number of circuit inputs expected by each model
    def get_num_params(self):
        return self.models[0].neural_network.num_inputs

    # Return the real parameter name for the selected model
    def get_weight_name(self, index):
        return self.weight_names[index]

    # Run one selected model with the shared discriminator weights
    def call_model(self, index, weights):
        return functional_call(
            self.models[index],
            {self.get_weight_name(index): weights},
            (),
        )

    # Run the selected real models and join their scalar outputs
    def forward(self, real_input, weights):
        indexes = real_input.detach().cpu().reshape(-1) if torch.is_tensor(real_input) else real_input

        if self.max_workers == 1:
            outputs = [
                self.call_model(int(index), weights)
                for index in indexes
            ]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                outputs = list(executor.map(lambda index: self.call_model(int(index), weights), indexes))

        return torch.cat([output.reshape(-1) for output in outputs], dim=0)



# Real and fake discriminator -#

# Discriminator class that joins real circuits discriminator with fake circuit discriminator
class JoinedDiscriminator(torch.nn.Module):
    # Build real and fake discriminator branches
    def __init__(self, disc_real_qnns, disc_fake_qnn, encoding="angle", max_workers=1):
        super().__init__()
        self.model_df = TorchConnector(disc_fake_qnn)
        
        if encoding in ["amplitude", "direct_circuit"]:
            self.model_dr = RealDiscriminatorAmplitude(disc_real_qnns, max_workers=max_workers)
        elif encoding == "angle":
            self.model_dr = RealDiscriminatorAngle(disc_real_qnns[0])
        else:
            raise ValueError(f"Unknown encoding method: {encoding}")

    # Expose the single trainable discriminator weight vector
    @property
    def weight(self):
        return self.model_df.weight

    # Number of real discriminator circuits available
    @property
    def num_real_models(self):
        return self.model_dr.num_real_models

    # Save only one discriminator weight tensor
    def state_dict(self):
        return {"weight": self.weight.detach()}

    # Restore the fake branch weights used by both discriminator branches
    def load_state_dict(self, state_dict):
        return self.model_df.load_state_dict({"weight": state_dict["weight"]}, strict=False)

    # Return real and fake discriminator outputs for one training batch
    def forward(self, real_input, fake_input):
        real_output = self.model_dr(real_input, self.weight)
        fake_output = self.model_df(fake_input)
        return real_output, fake_output



#- Amplitude evaluation backend -#

# Amplitude evaluation with backend.run and SaveProbabilities
class AmpEvalBackend(torch.nn.Module):
    # Store the transpiled evaluation circuit and its parameter order
    def __init__(self, gen_eval_circuit, eval_backend, bind_before_run=False):
        super().__init__()
        self.gen_eval_circuit = gen_eval_circuit
        self.eval_backend = eval_backend
        self.bind_before_run = bind_before_run
        _, gen_params, input_params = split_params_by_prefix(list(gen_eval_circuit.parameters))
        self.circuit_params = gen_params + input_params
        self.n_weights = len(gen_params)

    # Return how many random inputs the evaluation circuit expects
    def get_num_params(self):
        return len(self.circuit_params) - self.n_weights

    # Run exact backend evaluation and return probabilities as a tensor
    def forward(self, random_input, weights):
        # Get batch tensor
        batch_size = random_input.shape[0]
        gen_batch = weights.reshape(1, -1).expand(batch_size, -1)
        input_batch = torch.cat([gen_batch, random_input], dim=1)

        # Get batch list
        input_batch = input_batch.detach().cpu()
        if self.bind_before_run:
            bound_circuits = [
                self.gen_eval_circuit.assign_parameters(
                    {
                        self.circuit_params[i]: input_batch[j][i].item()
                        for i in range(len(self.circuit_params))
                    },
                    inplace=False,
                )
                for j in range(batch_size)
            ]
            result = self.eval_backend.run(bound_circuits).result()
            probs = torch.stack([
                torch.as_tensor(
                    result.data(j)['probabilities'],
                    dtype=weights.dtype,
                    device=weights.device,
                )
                for j in range(batch_size)
            ])

        else:
            parameter_binds = {
                self.circuit_params[i]: [input_batch[j][i].item() for j in range(batch_size)]
                for i in range(len(self.circuit_params))
            }

            # Get exact probability distribution
            job = self.eval_backend.run([self.gen_eval_circuit], parameter_binds=[parameter_binds])
            result = job.result()
            probs = result.data(0)['probabilities']
            probs = torch.as_tensor(probs, dtype=weights.dtype, device=weights.device)

            if probs.dim() == 1:
                probs = probs.reshape(1, -1)

        return probs



#- Torch model creation -#

# Create evaluation generator torch model
def generate_eval_g(encoding, gen_eval_model, eval_backend, config):
    if encoding == 'angle':
        eval_g = RealDiscriminatorAngle(gen_eval_model)
    elif encoding in ['direct_circuit', 'amplitude']:
        bind_before_run = (config["experiment"]["execution_type"] == "noiseless" and config["backend"]["simulator"]["device"] == "GPU")
        eval_g = AmpEvalBackend(gen_eval_model, eval_backend, bind_before_run=bind_before_run)
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")

    eval_g.eval()

    return eval_g


# Create evaluation generator model depending on encoding
def generate_eval_model(encoding, ran_gen_circuit, obs_eval, eval_estimator, eval_pm):
    if encoding == 'angle':
        gen_eval_circuit = prepare_eval_circuit_ang(ran_gen_circuit.copy())
        gen_eval_transpiled, obs_gen_eval = transpile_eval_circuits(gen_eval_circuit, obs_eval, eval_pm)
        return generate_eval_qnn_ang(gen_eval_transpiled, obs_gen_eval, eval_estimator)

    elif encoding in ['direct_circuit', 'amplitude']:
        gen_eval_circuit = prepare_eval_circuit_amp(ran_gen_circuit.copy())
        return eval_pm.run(gen_eval_circuit)

    else:
        raise ValueError(f"Unknown encoding method: {encoding}")


# Create circuits, QNNs and torch models
def generate_models(config, circuit_bundle):
    # Read the config values needed to build the torch/QNN models
    encoding = config['encoding']['type']
    max_workers = config['encoding']['max_parallel_threads']
    n_qubits = config['experiment']['n_qubits']
    seed = config['run']['seed']

    # Create Qiskit backends, circuits, observables and gradients
    _, _, train_estimator, train_pm, eval_backend, eval_estimator, eval_pm = create_backends(config)
    generator_circuit, discriminator_circuit, randomizer_circuit, real_circuits = circuit_bundle
    ran_gen_circuit, gen_disc_circuit, real_disc_circuits = get_composed_circuits(generator_circuit, discriminator_circuit, randomizer_circuit, real_circuits)

    obs_train, obs_eval = get_observables(n_qubits)
    gradient = get_gradient_method(
        config['experiment']['gradient_method'],
        train_estimator,
        seed,
    )

    # Convert the circuits into train/eval QNNs and torch modules
    real_disc_transpiled, gen_disc_transpiled, obs_real_disc, obs_gen_disc = transpile_train_circuits(gen_disc_circuit, real_disc_circuits, obs_train, train_pm)
    gen_qnn, disc_fake_qnn, disc_real_qnns = generate_train_qnns(real_disc_transpiled, gen_disc_transpiled, obs_real_disc, obs_gen_disc, gradient, train_estimator)
    gen_eval_model = generate_eval_model(encoding, ran_gen_circuit, obs_eval, eval_estimator, eval_pm)

    model_g = TorchConnector(gen_qnn)
    model_g.train()

    model_d = JoinedDiscriminator(
        disc_real_qnns,
        disc_fake_qnn,
        encoding=encoding,
        max_workers=max_workers,
    )
    model_d.train()

    eval_g = generate_eval_g(encoding, gen_eval_model, eval_backend, config)

    return model_g, model_d, eval_g
