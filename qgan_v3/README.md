# qGAN v3

Fully quantum GAN experiments using Qiskit, Qiskit Machine Learning, Qiskit Aer, and PyTorch.

## Project Structure

- `src/`: training code, circuit creation, config management, evaluation, and backend setup.
- `src/datasets/`: Python dataset generation, loading, and quantum dataset helpers.
- `src/config_manager/`: default config creation files and allowed config values.
- `data/`: generated run configs, training checkpoints, metrics, and run outputs.
- `datasets/`: raw and prepared dataset folders.
- `notebooks/`: experiment and runner notebooks.
- `others/`: older scripts and reference implementations.

## Installation

Create a Python 3.10 environment with `conda`:

```bash
conda create -n qgan python=3.10
conda activate qgan
pip install -r requirements.txt
```

Create a Python 3.10 environment with `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For notebooks:

```bash
python -m ipykernel install --user --name qgan
```

Optional GPU simulator support depends on your Linux/CUDA setup:

```bash
pip install qiskit-aer-gpu
```

## Configs

Config files support three implementations:

- `base`: quantum dataset with direct circuit encoding.
- `ang`: generated classical gradients with angle encoding.
- `amp`: generated classical gradients with amplitude encoding.

## Train

Run training from a generated config:

```bash
python qgan_v3/src/main.py qgan_v3/data/test/base-q3-noiseless-PSR-CPU-seed0/config.yaml
```

Training data is saved in the run folder as `training_data.pth`, so interrupted runs can resume from the same config.

## Notes

- Use `CPU` configs unless PyTorch, Qiskit Aer, CUDA, and `qiskit-aer-gpu` are correctly installed.
- Real hardware and noisy simulation may require a configured IBM Quantum account.
- Generated run data goes under the configured `run.data_path`, usually `qgan_v3/data/` or `qgan_v3/data/test/`.
