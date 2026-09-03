# Project Layout

```text
qgan/
  pyproject.toml           # package metadata
  README.md                # project overview
  docs/                    # project documentation
  configs/batteries/test/  # smoke-test sweep definitions
  configs/batteries/train/ # training sweep definitions
  datasets/prepared/       # reusable prepared datasets
  data/                    # generated configs and checkpoints
  notebooks/               # tutorial notebook
  tests/                   # lightweight package tests
  src/qgan_v2/             # installable Python package
  other_versions/          # archived older experiments
```

The active package is `qgan_v2`. The folder `qgan` is the project root and contains configs, docs, datasets, notebooks, tests, and package metadata.

Generated training outputs should live under `qgan/data/`. Prepared reusable datasets live under `qgan/datasets/prepared/`.
