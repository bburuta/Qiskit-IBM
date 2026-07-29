# Quick Start

Show CLI help:

```bash
qgan-v2 --help
```

Run the default smoke-test battery:

```bash
qgan-v2 -p qgan/configs/batteries/test.yaml
```

Start from fresh training data:

```bash
qgan-v2 -p qgan/configs/batteries/test.yaml --reset_data
```

Useful flags:

- `--overwrite`: regenerate existing config files.
- `--reset_rb` or `--reset_real_backend_info`: refresh cached real backend information.
- `--stop_on_error`: stop a battery run at the first failed config.

Generated configs and checkpoints are written under the configured `run.data_path`, usually `qgan/data/test/` for smoke tests.

If a battery run fails, the short error is printed in the terminal and the full traceback is saved beside that run's config:

```text
<run.data_path>/<run.id>/error_traceback.txt
```

To run without installing the package:

```bash
PYTHONPATH=qgan/src python3 -m qgan_v2.main -p qgan/configs/batteries/test.yaml
```

After training a run, visualize it from Python:

```bash
pip install -e "qgan[visualization]"
```

```python
from qgan_v2.visualization import run_visualization

run_visualization(
    "qgan/data/test/ang-qml_torch-q3-noiseless-PSR-aerCPU-seed0/config.yaml",
    {
        "draw_circuits": False,
        "draw_hardware_layout": False,
        "draw_probs": True,
        "draw_images": True,
        "draw_results": True,
    },
)
```

Visualization reads existing configs and checkpoints. It does not train missing runs.
