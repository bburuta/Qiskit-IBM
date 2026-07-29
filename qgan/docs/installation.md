# Installation

Create an environment from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e qgan
```

For notebooks:

```bash
pip install -e "qgan[notebooks]"
python -m ipykernel install --user --name qgan_v2
```

For visualization helpers and circuit drawings:

```bash
pip install -e "qgan[visualization]"
```

For notebooks that also use visualization:

```bash
pip install -e "qgan[notebooks,visualization]"
python -m ipykernel install --user --name qgan_v2
```

Optional GPU simulator support depends on the local CUDA/Linux setup:

```bash
pip install -e "qgan[gpu]"
```

For notebooks, visualization, and GPU simulator support together:

```bash
pip install -e "qgan[notebooks,visualization,gpu]"
python -m ipykernel install --user --name qgan_v2
```

GPU support is not included in `qgan[notebooks,visualization]` because `qiskit-aer-gpu-cu11` depends on the local CUDA/Linux setup and should be installed only on compatible machines.

The package installs the `qgan-v2` command and exposes the Python package `qgan_v2`.
