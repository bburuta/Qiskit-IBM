# Project Layout

```text
Qiskit-IBM/
  qgan/
    pyproject.toml           # package metadata
    README.md                # project overview
    docs/                    # project documentation
    configs/batteries/test/  # smoke-test sweep definitions
    configs/batteries/train/ # training sweep definitions
    datasets/prepared/       # reusable prepared datasets
    data/                    # generated configs and checkpoints
    notebooks/               # tutorial and results-analysis notebooks
    figures/                 # optional exported analysis figures
    tests/                   # package and analysis tests
    src/qgan_v2/             # installable Python package
    other_versions/          # archived older experiments
  slurm/                     # batch files and cluster monitoring helpers
```

The active package is `qgan_v2`. The folder `qgan` is the project root and contains configs, docs, datasets, notebooks, tests, and package metadata.

Generated training outputs should live under `qgan/data/`. Prepared reusable datasets live under `qgan/datasets/prepared/`.

The active analysis code is under `qgan/src/qgan_v2/analysis/`; reusable
single-run visualization is under `qgan/src/qgan_v2/visualization/`. See
[Results Analysis and Visualization](results-analysis.md) for their public
entry points. The top-level `slurm/` directory is cluster-specific rather than
part of the installed Python package; see its [helper guide](../../slurm/README.md).
