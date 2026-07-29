import torch
from concurrent.futures import ThreadPoolExecutor

from qiskit_machine_learning.connectors import TorchConnector

from backend import create_backends
from circuits import get_circuits
from qnn import (
    get_composed_circuits,
    get_gradient_method,
    get_observables,
    transpile_train_circuits,
    prepare_eval_circuit_ang,
    prepare_eval_circuit_amp,
    transpile_eval_circuits,
    generate_train_qnns,
    generate_eval_qnn_ang,
)



#- Torch modules -#

#f_loss = torch.nn.MSELoss(reduction="sum")
class FLoss(torch.nn.Module):
    def __init__(self):
        super(FLoss, self).__init__()

    def forward(self, x, label):
        loss = -x * label
        return loss.mean() # 'mean' for batches
    
f_loss = FLoss()


# Real discriminator class that joins multiple real circuits with discriminator
class RealDiscriminatorEnsemble(torch.nn.Module):
    def __init__(self, disc_real_qnns, encoding="angle", max_workers=1):
        super().__init__()
        self.models = torch.nn.ModuleList([TorchConnector(disc_real_qnn) for disc_real_qnn in disc_real_qnns])
        self.encoding = encoding
        self.max_workers = max_workers

    def tie_weights(self, weights):
        for model in self.models:
            if 'weight' in model._parameters:
                model._parameters.pop('weight')
            if '_weights' in model._parameters:
                model._parameters.pop('_weights')
            model._parameters['weight'] = weights
            model._parameters['_weights'] = weights

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
    def __init__(self, disc_real_qnns, disc_fake_qnn, initial_weights=None, encoding="angle", max_workers=1):
        super().__init__()
        self.model_dr = RealDiscriminatorEnsemble(
            disc_real_qnns,
            encoding=encoding,
            max_workers=max_workers,
        )
        self.model_df = TorchConnector(disc_fake_qnn, initial_weights=initial_weights)
        self.tie_weights()

    @property
    def num_real_models(self):
        return len(self.model_dr.models)

    def tie_weights(self):
        self.model_dr.tie_weights(self.model_df.weight)

    def parameters(self, recurse=True):
        return self.model_df.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self.model_df.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def state_dict(self, *args, **kwargs):
        return {
            'model_df_state': self.model_df.state_dict(*args, **kwargs),
        }

    def load_state_dict(self, state_dict, strict=True):
        result = self.model_df.load_state_dict(state_dict['model_df_state'], strict=strict)
        self.tie_weights()
        return result

    def _apply(self, fn):
        super()._apply(fn)
        self.tie_weights()
        return self

    def forward(self, real_input=None, fake_input=None):
        real_output = self.model_dr(real_input)
        fake_output = self.model_df(fake_input)
        return real_output, fake_output


# Amplitude evaluation with backend.run and SaveProbabilities
class AmpEvalBackend(torch.nn.Module):
    def __init__(self, gen_eval_circuit, eval_backend, initial_weights=None):
        super().__init__()
        self.gen_eval_circuit = gen_eval_circuit
        self.eval_backend = eval_backend
        self.circuit_params = list(gen_eval_circuit.parameters)
        self.n_weights = sum(p.name.startswith("θ_g") for p in self.circuit_params)

        # Weight initialization to later be tied to model_g weights
        if initial_weights is None:
            initial_weights = torch.zeros(self.n_weights)

        self.weight = torch.nn.Parameter(torch.as_tensor(initial_weights).clone().detach())
        self._parameters['_weights'] = self.weight

    def forward(self, random_input):
        # Get batch tensor
        batch_size = random_input.shape[0]
        gen_batch = self.weight.reshape(1, -1).expand(batch_size, -1)
        input_batch = torch.cat([gen_batch, random_input], dim=1)

        # Get batch list
        input_batch = input_batch.detach().cpu()
        parameter_binds = {
            self.circuit_params[i]: [input_batch[j][i].item() for j in range(batch_size)]
            for i in range(len(self.circuit_params))
        }

        # Get exact probability distribution
        job = self.eval_backend.run([self.gen_eval_circuit], parameter_binds=[parameter_binds])
        result = job.result()
        probs = result.data(0)['probabilities']
        probs = torch.as_tensor(probs, dtype=self.weight.dtype, device=self.weight.device)

        if probs.dim() == 1:
            probs = probs.reshape(1, -1)

        return probs


# Tie evaluation generator weights to training generator weights
def tie_eval_weights(eval_g, model_g):
    if 'weight' in eval_g._parameters:
        eval_g._parameters.pop('weight')
    if '_weights' in eval_g._parameters:
        eval_g._parameters.pop('_weights')
    eval_g._parameters['weight'] = model_g.weight
    eval_g._parameters['_weights'] = model_g.weight



#- Torch model creation -#

# Create generator torch connector model
def generate_torch_model_g(gen_qnn, init_gen_params=None):
    model_g = TorchConnector(gen_qnn, initial_weights=init_gen_params)
    model_g.train()

    return model_g


# Create discriminator torch connector model
def generate_torch_model_d(encoding, disc_fake_qnn, disc_real_qnns, init_disc_params=None, max_workers=1):
    model_d = JoinedDiscriminator(
        disc_real_qnns,
        disc_fake_qnn,
        initial_weights=init_disc_params,
        encoding=encoding,
        max_workers=max_workers,
    )
    model_d.train()

    return model_d


# Create evaluation generator torch connector model with eval backend (exact probability distribution)
def generate_torch_eval_g_amp_exact(gen_eval_transpiled, eval_backend, init_gen_params=None):
    eval_g = AmpEvalBackend(gen_eval_transpiled, eval_backend, initial_weights=init_gen_params)
    return eval_g


# Create evaluation generator torch connector model
def generate_torch_eval_g(encoding, gen_eval_model, model_g, eval_backend, init_gen_params=None):
    if encoding == 'angle':
        gen_eval_qnn = gen_eval_model
        eval_g = TorchConnector(gen_eval_qnn, initial_weights=init_gen_params)
    elif encoding in ['direct_circuit', 'amplitude']:
        gen_eval_transpiled = gen_eval_model
        eval_g = generate_torch_eval_g_amp_exact(gen_eval_transpiled, eval_backend, init_gen_params=init_gen_params)
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")

    tie_eval_weights(eval_g, model_g)
    eval_g.eval()

    return eval_g


# Create evaluation generator model depending on encoding
def generate_eval_model(encoding, ran_gen_circuit, obs_eval, gradient, eval_estimator, eval_backend, eval_pm):
    if encoding == 'angle':
        gen_eval_circuit = prepare_eval_circuit_ang(ran_gen_circuit.copy())
        gen_eval_transpiled, obs_gen_eval = transpile_eval_circuits(gen_eval_circuit, obs_eval, eval_pm)
        gen_eval_model = generate_eval_qnn_ang(gen_eval_transpiled, obs_gen_eval, gradient, eval_estimator)
    
    elif encoding in ['direct_circuit', 'amplitude']:
        gen_eval_circuit = prepare_eval_circuit_amp(ran_gen_circuit.copy())
        gen_eval_model = eval_pm.run(gen_eval_circuit)
    
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")

    return gen_eval_model


# Create torch connector models
def create_torch_models(encoding, max_workers, gen_qnn, disc_fake_qnn, disc_real_qnns, gen_eval_model, eval_backend, init_gen_params=None, init_disc_params=None):
    model_g = generate_torch_model_g(gen_qnn, init_gen_params=init_gen_params)
    model_d = generate_torch_model_d(
        encoding,
        disc_fake_qnn,
        disc_real_qnns,
        init_disc_params=init_disc_params,
        max_workers=max_workers,
    )
    eval_g = generate_torch_eval_g(
        encoding,
        gen_eval_model,
        model_g,
        eval_backend,
        init_gen_params=init_gen_params,
    )

    return model_g, model_d, eval_g


# Create circuits, QNNs and torch connector models
def generate_torch_models(config):
    encoding = config['encoding']['type']
    max_workers = config['encoding']['batch_size']
    n_qubits = config['experiment']['n_qubits']
    seed = config['run']['seed']

    _, _, train_estimator, train_pm, eval_backend, eval_estimator, eval_pm = create_backends(config)
    generator_circuit, discriminator_circuit, randomizer_circuit, real_circuits = get_circuits(config)
    ran_gen_circuit, gen_disc_circuit, real_disc_circuits = get_composed_circuits(generator_circuit, discriminator_circuit, randomizer_circuit, real_circuits)

    obs_train, obs_eval = get_observables(n_qubits)
    gradient = get_gradient_method(config['experiment']['gradient_method'], train_estimator, seed)

    real_disc_transpiled, gen_disc_transpiled, obs_real_disc, obs_gen_disc = transpile_train_circuits(gen_disc_circuit, real_disc_circuits, obs_train, train_pm)
    gen_qnn, disc_fake_qnn, disc_real_qnns = generate_train_qnns(real_disc_transpiled, gen_disc_transpiled, obs_real_disc, obs_gen_disc, gradient, train_estimator)
    gen_eval_model = generate_eval_model(encoding, ran_gen_circuit, obs_eval, gradient, eval_estimator, eval_backend, eval_pm)

    return create_torch_models(encoding, max_workers, gen_qnn, disc_fake_qnn, disc_real_qnns, gen_eval_model, eval_backend)
