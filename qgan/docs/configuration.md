# Configuration

Battery files live in `qgan/configs/batteries/test/` for smoke-test batteries and `qgan/configs/batteries/train/` for training batteries. The folder inventory is in `qgan/configs/batteries/README.md`. They define default config values and one or more variable sweeps.

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
- `training.checkpoint_every`: completed-epoch interval for checkpoint saves; `0` disables periodic saves and writes only at the end.
- `encoding.random_circuit`: randomizer circuit mode used when `encoding.randomness` is nonzero. For `direct_circuit` and `amplitude`: `0` none, `1` RY gates, `2` EfficientSU2 with two repetitions. Mode `3` is kept in code as a disabled fixed-statevector prototype. For `angle`: `0` none, any nonzero value uses the original angle-style RY randomizer.
- `encoding.randomness`: normalized random input strength in `[0, 1]`. For parameterized randomizers, it scales samples from `[0, 2*pi]`. `0` disables the randomizer circuit regardless of `encoding.random_circuit`.
- `run.data_path`: output directory for generated configs and checkpoints.
- `run.label`: optional suffix appended to generated run IDs.
- `run.device`: PyTorch device, `CPU` or `GPU`.
- `backend.reset`: recreate the cached per-run backend options file.
- `backend.save_backend_file`: save and reuse the per-run backend options file.
- `backend.real.info_storage`: where real-backend calibration/topology data is cached: `shared` stores it in `qgan/backends/`; `run` stores it beside each generated `config.yaml`.
- `backend.real.reset_info`: recreate cached real-backend calibration/topology data.
- `backend.real.confirm_runtime_execution`: ask for confirmation before `real` or `fake_real` Runtime execution.
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

Backend options are cached per run:

```text
<run.data_path>/<run.id>/backend.pkl
```

Real-backend calibration/topology data is cached according to `backend.real.info_storage`:

```text
shared -> qgan/backends/<backend.real.id>.pkl
run    -> <run.data_path>/<run.id>/real_backend.pkl
```

Battery-generated configs set `backend.real.reset_info: false` for `shared` storage so runs do not repeatedly overwrite the same shared file. Use `--reset-rb` to refresh the shared backend file once before a battery run. `run` storage keeps `backend.real.reset_info` and refreshes independently inside each run when it is true.

For `fake_real`, `run` storage saves the local `FakeSherbrooke` backend info in the run folder.

The default smoke-test battery writes under `qgan/data/test/`.

Run IDs are generated from the qGAN preset, implementation adapter, runtime-packed discriminator packing, qubits, execution type, gradient method, Aer device, randomness scale, and seed, for example:

```text
ang-qml_torch-q3-noiseless-PSR-aerGPU-rand0.1-seed0
```

Set `run.label` to append a short group label to the final run ID. The label is appended as `:<label>` whether `run.id` was generated from `null` or written manually:

```yaml
run:
  id: null
  label: noisy_dm_hw
```

```text
ang-qml_torch-q3-noisy-PSR-aerCPU-rand0.1-seed0:noisy_dm_hw
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

When `backend.real.confirm_runtime_execution: true`, the run asks before `real` or `fake_real` Runtime execution.

Noisy local simulation supports two mapping modes:

```yaml
backend:
  simulator:
    noisy_backend_mapping: hardware
    # noisy_backend_mapping: noise_model
```

Use `noise_model` when you want calibrated local noise without mapping small circuits onto the full real-backend topology.
