# Battery Configs

Battery YAMLs are grouped by intent:

- `test/`: short smoke tests, Aer option checks, checkpoint timing checks, and fake/real hardware sanity checks.
- `train/`: full training batteries used for convergence and timing studies. Local simulator batteries are split by CPU/GPU device, and real-hardware studies live in separate `*_rh.yaml` files.

Each YAML file starts with a short description of the experiments it runs. The
shared schema, config fields, generated output paths, and execution-mode details
are documented in the [configuration docs](../../docs/configuration.md).
