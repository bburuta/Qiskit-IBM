# Configuration

Battery files live in `qgan/configs/batteries/`. They define default config values and one or more variable sweeps.

Minimal shape:

```yaml
default_config_values:
  implementation:
    name: qml_torch
    discriminator_packing: separate
  run:
    id: null
    label: null
    data_path: qgan/data/test

variable_config_values_list:
  qml_torch_test:
    experiment.implementation: [base, ang, amp]
    experiment.gradient_method: [PSR, SPSA]
    run.label: [qml_torch_test]
    run.seed: [0]
```

Dotted keys such as `experiment.n_qubits` are preferred because simple keys can be ambiguous.

Important fields:

- `implementation.name`: `qml_torch`, `runtime_packed`, or `manual_estimator`.
- `implementation.discriminator_packing`: `separate` executes real and fake discriminator circuits independently; `joined` combines them into one packed circuit.
- `experiment.implementation`: qGAN preset, usually `base`, `ang`, or `amp`.
- `experiment.execution_type`: `noiseless`, `noisy`, `fake_real`, or `real`.
- `experiment.gradient_method`: `PSR`, `SPSA`, or `REG`.
- `training.init_scale`: initial trainable parameter scale applied to samples from `[-pi, pi]`.
- `training.learning_rate`: shared Adam learning rate for the generator and discriminator optimizers.
- `run.data_path`: output directory for generated configs and checkpoints.
- `run.label`: optional suffix appended to generated run IDs.
- `run.device`: PyTorch device, `CPU` or `GPU`.
- `backend.simulator.device`: Aer simulator device, `CPU` or `GPU`.
- `backend.transpilation`: compiler optimization, layout, and routing settings.

Use `qml_torch` for `noiseless` or `REG`. Use `runtime_packed` for packed primitive execution against noisy, fake-runtime, or real Runtime backends.

## Outputs

Battery configs are written to:

```text
<run.data_path>/<run.id>/config.yaml
```

Training writes or updates:

```text
<run.data_path>/<run.id>/training_data.pth
```

The default smoke-test battery writes under `qgan/data/test/`.

Run IDs are generated from the qGAN preset, implementation adapter, runtime-packed discriminator packing, qubits, execution type, gradient method, Aer device, randomness, and seed, for example:

```text
ang-qml_torch-q3-noiseless-PSR-aerGPU-rand0-seed0
```

Set `run.label` to append a short group label to the final run ID. The label is appended as `:<label>` whether `run.id` was generated from `null` or written manually:

```yaml
run:
  id: null
  label: noisy_dm_hw
```

```text
ang-qml_torch-q3-noisy-PSR-aerCPU-rand0-seed0:noisy_dm_hw
```

## Presets

`experiment.implementation` controls dataset and encoding defaults:

- `base`: direct quantum circuit dataset with `direct_circuit` encoding.
- `ang`: generated gradient image dataset with angle encoding.
- `amp`: generated gradient image dataset with amplitude encoding.

Prepared generated-gradient datasets are stored under `qgan/datasets/prepared/`. Missing generated-gradient datasets are created automatically when `dataset.reset` is true or the expected `.npz` file does not exist.

## Runtime And Backend Notes

Recommended hardware transpilation settings:

```yaml
backend:
  transpilation:
    optimization_level: 3
    layout_method: sabre
    routing_method: sabre
```

`runtime_packed` supports the `base`, `ang`, and `amp` presets with `noisy`, `fake_real`, or `real` execution and `PSR` or `SPSA` gradients. Joined discriminator packing supports `base` with `batch_size: 1` and `ang` with an even batch size. Use separate packing for `amp`.

`runtime_packed` requires `run.device: CPU` because its trainable parameters share memory with NumPy. Set `backend.simulator.device: GPU` to run supported Aer simulation on GPU independently.

Execution modes:

- `noiseless`: local ideal Aer simulation.
- `noisy`: local Aer simulation from cached real-backend calibration data.
- `fake_real`: local Runtime execution using `qiskit_ibm_runtime.fake_provider.FakeSherbrooke`; this checks hardware-style transpilation and execution without submitting IBM Quantum jobs.
- `real`: IBM Runtime execution on `backend.real.name`.

Noisy local simulation supports two mapping modes:

```yaml
backend:
  simulator:
    noisy_backend_mapping: hardware
    # noisy_backend_mapping: noise_model
```

Use `noise_model` when you want calibrated local noise without mapping small circuits onto the full real-backend topology.
