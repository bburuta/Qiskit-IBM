# qGAN

`qgan` is the active package-style rewrite of the qGAN experiments. It separates the generic GAN loop from implementation-specific quantum execution so new approaches can be added without growing one large training script. The import package is `qgan_v2`.

## Status

Working:

- `qml_torch`: Qiskit Machine Learning + Torch implementation owned by `qgan_v2`.
- `runtime_packed`: primitive-backed implementation that packs same-template training batches into one circuit and can run against local Aer, fake-runtime, or real IBM Runtime execution.
- Battery config generation.
- Smoke-test training from generated configs.
- Checkpoint and metric persistence in `training_data.pth`.
- Basic visualization helpers for completed runs.

Scaffolded only:

- `manual_estimator`: target for Qiskit primitive or gradient implementations without `qiskit-machine-learning`.

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e qgan
```

For notebooks, install the notebook extra:

```bash
pip install -e "qgan[notebooks,visualization]"
python -m ipykernel install --user --name qgan_v2
```

## Quick Start

```bash
qgan-v2 --help
qgan-v2 -p qgan/configs/batteries/test.yaml
```

## Documentation

More detailed documentation lives in `qgan/docs/`:

- [Installation](docs/installation.md)
- [Quick Start](docs/quickstart.md)
- [Configuration](docs/configuration.md)
- [Project Layout](docs/project-layout.md)
- [Other Versions](docs/other-versions.md)

## License

Code is licensed under the MIT License. Thesis text, explanatory documentation,
and original figures are licensed under CC BY 4.0 unless otherwise noted.
Datasets, generated outputs, IBM Quantum backend information, and third-party
materials may have separate terms. See [LICENSE](LICENSE).
