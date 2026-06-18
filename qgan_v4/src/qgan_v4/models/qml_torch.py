import torch
from concurrent.futures import ThreadPoolExecutor

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


# Evaluation connector that reuses the current generator weights
class EvalTorchConnector(TorchConnector):
    # Tie this evaluation connector to the generator weight parameter
    def set_weight_source(self, model):
        if 'weight' in self._parameters:
            self._parameters.pop('weight')
        if '_weights' in self._parameters:
            self._parameters.pop('_weights')
        self._parameters['weight'] = model.weight
        self._parameters['_weights'] = model.weight

    # Return how many random inputs the evaluation QNN expects
    def get_num_random_params(self):
        return self.neural_network.num_inputs


# Real discriminator class that joins multiple real circuits with discriminator
class RealDiscriminatorEnsemble(torch.nn.Module):
    # Build one TorchConnector per real discriminator QNN
    def __init__(self, disc_real_qnns, encoding="angle", max_workers=1):
        super().__init__()
        self.models = torch.nn.ModuleList([TorchConnector(disc_real_qnn) for disc_real_qnn in disc_real_qnns])
        self.encoding = encoding
        self.max_workers = max_workers

    # Tie all real discriminator circuits to the fake discriminator weights
    def tie_weights(self, weights):
        for model in self.models:
            if 'weight' in model._parameters:
                model._parameters.pop('weight')
            if '_weights' in model._parameters:
                model._parameters.pop('_weights')
            model._parameters['weight'] = weights
            model._parameters['_weights'] = weights

    # Run the real discriminator branch for the selected encoding
    def forward(self, real_input=None):
        if self.encoding == "direct_circuit":
            return self.models[0]()

        elif self.encoding == "angle":
            return self.models[0](real_input)

        elif self.encoding == "amplitude":
            indexes = real_input.detach().cpu().reshape(-1) if torch.is_tensor(real_input) else real_input
            selected_models = [self.models[int(index)] for index in indexes]

            if self.max_workers == 1:
                outputs = [model() for model in selected_models]
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    outputs = list(executor.map(lambda model: model(), selected_models))

            return torch.cat([output.reshape(-1) for output in outputs], dim=0)

        else:
            raise ValueError(f"Unknown encoding method: {self.encoding}")


# Discriminator class that joins real circuits discriminator with fake circuit discriminator
class JoinedDiscriminator(torch.nn.Module):
    # Build real and fake discriminator branches and tie their weights
    def __init__(self, disc_real_qnns, disc_fake_qnn, encoding="angle", max_workers=1):
        super().__init__()
        self.model_dr = RealDiscriminatorEnsemble(
            disc_real_qnns,
            encoding=encoding,
            max_workers=max_workers,
        )
        self.model_df = TorchConnector(disc_fake_qnn)
        self.tie_weights()

    # Number of real circuits available for amplitude batches
    @property
    def num_real_models(self):
        return len(self.model_dr.models)

    # Share one discriminator parameter vector between real and fake branches
    def tie_weights(self):
        self.model_dr.tie_weights(self.model_df.weight)

    # Expose only the fake branch parameters to the optimizer
    def parameters(self, recurse=True):
        return self.model_df.parameters(recurse=recurse)

    # Expose only the fake branch named parameters to avoid duplicates
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self.model_df.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    # Save a single discriminator weight tensor
    def state_dict(self, *args, **kwargs):
        keep_vars = kwargs.get("keep_vars", False)
        return {
            "weight": self.model_df.weight if keep_vars else self.model_df.weight.detach(),
        }

    # Restore discriminator weights and reconnect the shared real branches
    def load_state_dict(self, state_dict, strict=True):
        result = self.model_df.load_state_dict({"weight": state_dict["weight"]}, strict=False)
        self.tie_weights()
        return result

    # Re-tie weights after device or dtype changes
    def _apply(self, fn):
        super()._apply(fn)
        self.tie_weights()
        return self

    # Return real and fake discriminator outputs for one training batch
    def forward(self, real_input=None, fake_input=None):
        real_output = self.model_dr(real_input)
        fake_output = self.model_df(fake_input)
        return real_output, fake_output


# Amplitude evaluation with backend.run and SaveProbabilities
class AmpEvalBackend(torch.nn.Module):
    # Store the transpiled evaluation circuit and its parameter order
    def __init__(self, gen_eval_circuit, eval_backend):
        super().__init__()
        self.gen_eval_circuit = gen_eval_circuit
        self.eval_backend = eval_backend
        _, gen_params, input_params = split_params_by_prefix(list(gen_eval_circuit.parameters))
        self.circuit_params = gen_params + input_params
        self.n_weights = len(gen_params)

        # Placeholder weight is tied to model_g immediately after creation.
        self.weight = torch.nn.Parameter(torch.zeros(self.n_weights))
        self._parameters['_weights'] = self.weight

    # Tie exact evaluation to the current generator weight parameter
    def set_weight_source(self, model):
        if 'weight' in self._parameters:
            self._parameters.pop('weight')
        if '_weights' in self._parameters:
            self._parameters.pop('_weights')
        self._parameters['weight'] = model.weight
        self._parameters['_weights'] = model.weight

    # Return the number of random circuit parameters after generator weights
    def get_num_random_params(self):
        return len(self.circuit_params) - self.n_weights

    # Run exact backend evaluation and return probabilities as a torch tensor
    def forward(self, random_input):
        batch_size = random_input.shape[0]
        gen_batch = self.weight.reshape(1, -1).expand(batch_size, -1)
        input_batch = torch.cat([gen_batch, random_input], dim=1).detach().cpu()

        parameter_binds = {
            self.circuit_params[i]: [input_batch[j][i].item() for j in range(batch_size)]
            for i in range(len(self.circuit_params))
        }

        job = self.eval_backend.run([self.gen_eval_circuit], parameter_binds=[parameter_binds])
        result = job.result()
        probs = result.data(0)['probabilities']
        probs = torch.as_tensor(probs, dtype=self.weight.dtype, device=self.weight.device)

        if probs.dim() == 1:
            probs = probs.reshape(1, -1)

        return probs


#- Torch model creation -#

# Create evaluation generator torch model for the selected encoding
def generate_eval_g(encoding, gen_eval_model, model_g, eval_backend):
    if encoding == 'angle':
        eval_g = EvalTorchConnector(gen_eval_model)
    elif encoding in ['direct_circuit', 'amplitude']:
        eval_g = AmpEvalBackend(gen_eval_model, eval_backend)
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")

    eval_g.set_weight_source(model_g)
    eval_g.eval()

    return eval_g


# Create the raw evaluation QNN or transpiled exact-evaluation circuit
def generate_eval_model(encoding, ran_gen_circuit, obs_eval, gradient, eval_estimator, eval_backend, eval_pm):
    if encoding == 'angle':
        gen_eval_circuit = prepare_eval_circuit_ang(ran_gen_circuit.copy())
        gen_eval_transpiled, obs_gen_eval = transpile_eval_circuits(gen_eval_circuit, obs_eval, eval_pm)
        return generate_eval_qnn_ang(gen_eval_transpiled, obs_gen_eval, gradient, eval_estimator)

    elif encoding in ['direct_circuit', 'amplitude']:
        gen_eval_circuit = prepare_eval_circuit_amp(ran_gen_circuit.copy())
        return eval_pm.run(gen_eval_circuit)

    else:
        raise ValueError(f"Unknown encoding method: {encoding}")


# Create circuits, QNNs and torch connector models
def generate_models(config, circuit_bundle):
    # Read the pieces of config needed to build the torch/QNN models
    encoding = config['encoding']['type']
    max_workers = config['encoding']['batch_size']
    n_qubits = config['experiment']['n_qubits']
    seed = config['run']['seed']

    # Create Qiskit backends, circuits, observables and gradients
    _, _, train_estimator, train_pm, eval_backend, eval_estimator, eval_pm = create_backends(config)
    generator_circuit, discriminator_circuit, randomizer_circuit, real_circuits = circuit_bundle
    ran_gen_circuit, gen_disc_circuit, real_disc_circuits = get_composed_circuits(generator_circuit, discriminator_circuit, randomizer_circuit, real_circuits)

    obs_train, obs_eval = get_observables(n_qubits)
    gradient = get_gradient_method(config['experiment']['gradient_method'], train_estimator, seed)

    # Convert the circuits into train/eval QNNs and torch modules
    real_disc_transpiled, gen_disc_transpiled, obs_real_disc, obs_gen_disc = transpile_train_circuits(gen_disc_circuit, real_disc_circuits, obs_train, train_pm)
    gen_qnn, disc_fake_qnn, disc_real_qnns = generate_train_qnns(real_disc_transpiled, gen_disc_transpiled, obs_real_disc, obs_gen_disc, gradient, train_estimator)
    gen_eval_model = generate_eval_model(encoding, ran_gen_circuit, obs_eval, gradient, eval_estimator, eval_backend, eval_pm)

    model_g = TorchConnector(gen_qnn)
    model_g.train()

    model_d = JoinedDiscriminator(
        disc_real_qnns,
        disc_fake_qnn,
        encoding=encoding,
        max_workers=max_workers,
    )
    model_d.train()

    eval_g = generate_eval_g(encoding, gen_eval_model, model_g, eval_backend)

    return model_g, model_d, eval_g
