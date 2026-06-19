# qGAN v4

`qgan_v4` is a package-style rewrite of the qGAN experiments. It separates the generic GAN loop from implementation-specific quantum execution so new approaches can be added without growing one large training script.

## Status

Working:

- `qml_torch`: Qiskit Machine Learning + Torch implementation owned by `qgan_v4`.
- Battery config generation.
- Smoke-test training from generated configs.
- Checkpoint and metric persistence in `training_data.pth`.
- Basic visualization helpers for completed runs.

Scaffolded only:

- `angle_runtime_packed`: target for packed IBM Runtime angle experiments.
- `manual_estimator`: target for Qiskit primitive or gradient implementations without `qiskit-machine-learning`.

These scaffolded implementations are registered so configs can be shaped now, but training them raises `NotImplementedError`.

## Installation

### Environment Installation

#### Python `venv`

From the repository root, create and activate a Python 3.10 virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Remove the local `venv` environment when it is no longer needed:

```bash
deactivate
rm -rf .venv
```

#### Conda

From the repository root, create and activate a Python 3.10 Conda environment:

```bash
conda create -n qgan_v4 python=3.10
conda activate qgan_v4
```

Remove the Conda environment when it is no longer needed:

```bash
conda deactivate
conda env remove -n qgan_v4
```

### Requirement Installation

Install the qGAN v4 Python requirements inside the active environment:

```bash
python -m pip install --upgrade pip
pip install -r qgan_v4/requirements.txt
```

For notebooks:

```bash
python -m ipykernel install --user --name qgan_v4
```

Optional GPU simulator support depends on your Linux/CUDA setup:

```bash
pip install qiskit-aer-gpu
```

Run commands from the repository root with `qgan_v4/src` on `PYTHONPATH`:

```bash
PYTHONPATH=qgan_v4/src python3 -m qgan_v4.main --help
```

## Quick Start

Run the smoke-test battery:

```bash
PYTHONPATH=qgan_v4/src python3 -m qgan_v4.main -p qgan_v4/configs/batteries/test.yaml
```

Start from fresh training data:

```bash
PYTHONPATH=qgan_v4/src python3 -m qgan_v4.main -p qgan_v4/configs/batteries/test.yaml --reset_data
```

Add `--overwrite` to regenerate existing config files. Add `--reset_rb` or `--reset_real_backend_info` to refresh real backend info. Add `--stop_on_error` to stop a battery run at the first failed config.

## Outputs

Battery configs are written to:

```text
<run.data_path>/<run.id>/config.yaml
```

Training writes or updates:

```text
<run.data_path>/<run.id>/training_data.pth
```

The default smoke-test battery writes under `qgan_v4/data/test/`.

Run IDs are generated from the qGAN preset, implementation adapter, qubits, execution type, gradient method, device, and seed, for example:

```text
ang-qml_torch-q3-noiseless-PSR-CPU-seed0
```

## Configs

Battery files live in:

```text
qgan_v4/configs/batteries/
```

Current battery files:

- `test.yaml`: small smoke-test combinations.
- `train.yaml`: larger training sweep template.

Battery files use the v3-style structure:

```yaml
default_config_values:
  implementation:
    name: qml_torch
  run:
    data_path: qgan_v4/data/test

variable_config_values_list:
  qml_torch_test:
    experiment.implementation: [base, ang, amp]
    experiment.gradient_method: [PSR, SPSA]
    run.seed: [0]
```

Dotted keys such as `experiment.n_qubits` are preferred because simple keys can be ambiguous.

Important config fields:

- `implementation.name`: execution adapter. Use `qml_torch` for runnable training.
- `experiment.implementation`: qGAN preset: `base`, `ang`, or `amp`.
- `experiment.execution_type`: `noiseless`, `noisy`, `fake_real`, or `real`.
- `experiment.gradient_method`: `PSR`, `SPSA`, or `REG`.
- `run.device`: `CPU` or `GPU`.

Execution modes:

- `noiseless`: local ideal Aer simulation.
- `noisy`: local Aer simulation from cached real-backend calibration data.
- `fake_real`: local Aer simulation using `qiskit_ibm_runtime.fake_provider.FakeSherbrooke`; this checks hardware-style transpilation and noisy execution without submitting IBM Runtime jobs.
- `real`: IBM Runtime execution on `backend.real.name`.

Noisy local simulation supports two mapping modes:

```yaml
backend:
  simulator:
    noisy_backend_mapping: hardware     # full backend coupling map/target, closest to hardware transpilation
    # noisy_backend_mapping: noise_model # faster; use backend-derived NoiseModel without full coupling map/target
```

Use `noise_model` when you want calibrated local noise without mapping small circuits onto the full real-backend topology.

For a hardware-shaped dry run, set:

```yaml
experiment:
  execution_type: fake_real
backend:
  real:
    id: null
```

For actual hardware execution, set the backend name:

```yaml
experiment:
  execution_type: real
backend:
  real:
    name: ibm_basquecountry
    channel: ibm_quantum_platform
```

`fake_real` always uses `FakeSherbrooke`; there is no config option for any other fake backend. It does not prove queue submission or account access. It proves that the run can be transpiled and evaluated against a Sherbrooke-like backend model locally.

## Presets

`experiment.implementation` controls dataset and encoding defaults:

- `base`: direct quantum circuit dataset with `direct_circuit` encoding.
- `ang`: generated gradient image dataset with angle encoding.
- `amp`: generated gradient image dataset with amplitude encoding.

Prepared generated-gradient datasets are stored under `qgan_v4/datasets/prepared/`. Missing generated-gradient datasets are created automatically when `dataset.reset` is true or the expected `.npz` file does not exist.

## Visualization

After training a run, visualize it from Python:

```python
from qgan_v4.visualization import run_visualization

run_visualization(
    "qgan_v4/data/test/ang-qml_torch-q3-noiseless-PSR-CPU-seed0/config.yaml",
    {
        "draw_circuits": False,
        "draw_probs": True,
        "draw_images": True,
        "draw_results": True,
    },
)
```

Visualization reads configs and checkpoints. It does not train missing runs.

## Layout

```text
qgan_v4/
  configs/batteries/       # sweep definitions
  datasets/prepared/       # reusable prepared datasets
  data/                    # generated configs and checkpoints
  notebooks/               # tutorial notebook
  src/qgan_v4/
    main.py                # CLI
    train.py               # generic training loop
    config/                # loading, validation, battery expansion
    storage/               # filesystem layout helpers
    runs/                  # training state and checkpoint helpers
    implementations/       # concrete qGAN adapters
    execution/             # backend and Runtime helpers
    models/                # Torch, QML, and manual model code
    circuits/              # circuit factories and parameters
    datasets/              # dataset creation/loading
    evaluation/            # metrics
    visualization/         # plotting helpers
```
