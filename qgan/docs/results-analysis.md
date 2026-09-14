# Results Analysis and Visualization

The results tools read completed or failed run directories without starting new
training. Install the optional dependencies before using the notebook or Python
API:

```bash
pip install -e "qgan[results]"
```

The main entry point is
[`notebooks/results_analysis.ipynb`](../notebooks/results_analysis.ipynb). It can
be opened from either the repository root or the `qgan/` directory; its setup
cell locates the package source automatically.

## Data discovery

`ResultsAnalysis.load("qgan")` reads these directories:

```text
qgan/data/train/        # convergence, hardware, and validation runs
qgan/data/train/times/  # five-epoch timing runs
```

Each run directory is expected to contain `config.yaml` and, for a run with a
checkpoint, `training_data.pth`. A missing checkpoint or an
`error_traceback.txt` is retained as feasibility evidence rather than silently
discarded.

```python
from qgan_v2.analysis import ResultsAnalysis

analysis = ResultsAnalysis.load(
    "qgan",
    export_figures=False,
    gpu_model_label="RTX6000",
)
print(analysis.results.summary())
```

Set `export_figures=True` to save PDF and 300-dpi PNG copies under
`qgan/figures/results_analysis/`. Pass `figure_dir=...` to use another output
directory. Figures are always displayed inline.

## Analysis workflow

The notebook is the executable record of the full workflow. Its sections cover:

- matched device timing, epoch-time distributions, and experimental
  feasibility;
- preset learning dynamics and initial/last/best generated outputs;
- noiseless, noisy, and real-QPU comparisons;
- gradient-method and input-randomness comparisons;
- qubit scaling by preset, execution type, gradient method, and randomness;
- fake-real implementation and discriminator-packing validation.

The corresponding high-level methods live on `ResultsAnalysis`, so individual
sections can also be run from Python. For example:

```python
for preset in ("base", "ang", "amp"):
    analysis.matched_timing_comparison(preset=preset)
analysis.timing_distributions(compare_by="randomness")
analysis.timing_distributions(preset=("base", "ang", "amp"))
analysis.timing_distributions(compare_by="randomness", layout=(2, 2))

runs = analysis.preset_scaling(
    filters={
        "implementation": "qml_torch",
        "execution_type": "noiseless",
        "gradient_method": "PSR",
        "randomness": 0,
    },
    metric_transform="standardize",
)
```

Filters match normalized run metadata. Common fields are `preset`,
`implementation`, `execution_type`, `gradient_method`, `n_qubits`,
`randomness`, `seed`, and `eval_method`. A tuple, list, or set selects any of
the supplied values.

The lower-level `qgan_v2.analysis` API also exposes result loading and
selection, per-run summaries, grouped tables, paired deltas, convergence and
final-metric plots, runtime plots, and feasibility plots. Use these functions
when a comparison is not represented by a `ResultsAnalysis` method.

The timing workflow deliberately uses several complementary views. The matched
dumbbell plot compares identical workloads across environments. The
distribution panels show each run's median as a prominent circle and retain
its individual epoch times as faint shadow circles. Their `preset`,
`execution_type`, `n_qubits`, `gradient_method`, and `randomness` arguments each
accept either one value or a sequence. Passing a
sequence compares the selected devices in one panel per value, such as
`randomness=(0, 1)`, while keeping the other settings controlled. Passing
`compare_by="randomness"` instead makes CPU/1, CPU/4, GPU, and Real-QPU panels,
with rand0/rand1 categories on each panel's x-axis. Every eligible configuration
is pooled unless the other scientific arguments are supplied as filters. The
same comparison mode supports `preset`, `execution_type`, `n_qubits`, and
`gradient_method`. Panels use a compact horizontal `1 × N` layout by default,
so changing the requested environments or controlled configurations adjusts
the number of columns without introducing another row. Pass
`layout=(rows, columns)`, such as `layout=(2, 2)`, to override it; the grid must
have at least as many cells as generated panels. Within each device panel,
comparison categories with no recorded timing values are omitted together with
their unused space. Panel widths are proportional to the number of displayed
categories, keeping each category slot the same physical width across panels.
Entirely empty panels are not rendered. `runtime_scaling()`,
`training_cost_table()`, and `training_cost_figure()` remain available as
optional API methods, but the notebook omits them from its main narrative: the
first repeats timing evidence at a coarser level, while the cost views linearly
project short timing runs rather than reporting measured full-training costs.

## Score transformations

Convergence plots accept `transform="none"`, `"normalize"`, or
`"standardize"`. The transform is applied independently to each run before
seed aggregation:

- `none` preserves the recorded evaluation scale;
- `normalize` applies per-run min-max scaling to `[0, 1]`;
- `standardize` applies a per-run z-score.

Constant finite series map to zero. The checkpoint data is never mutated.
Transformed curves are useful for comparing trajectory shapes across metric
scales, but they no longer show absolute model quality. For that reason, the
workflow's best-score summaries continue to use raw evaluation values.

The analysis treats each run/seed as an independent replicate. Convergence
curves show the median and interquartile range across runs, with individual
traces retained faintly. Epochs are not treated as independent samples. See the
[results analysis plan](results-analysis-plan.md) for the exact comparison
choices, limitations, and interpretation rules used by the current notebook.

## Generated-output visualization

Use `run_visualization` for the full diagnostic view of one run:

```python
from qgan_v2.visualization import run_visualization

run_visualization(
    "qgan/data/test/ang-qml_torch-q3-noiseless-PSR-aerCPU-rand0.1-seed0/config.yaml",
    {
        "draw_circuits": False,
        "draw_hardware_layout": False,
        "draw_probs": True,
        "draw_images": True,
        "draw_results": True,
    },
)
```

For publication-style qualitative comparisons, use the focused helpers:

```python
from qgan_v2.visualization import (
    plot_generated_output,
    plot_generated_output_sequence,
)

fig, manifest = plot_generated_output_sequence(
    "path/to/run/config.yaml",
    parameter_sets=("initial", "last", "best"),
    random_seed=0,
    save_path="figures/run_sequence.png",
)

fig, manifest = plot_generated_output(
    "path/to/run/config.yaml",
    parameter_set="best",
    random_seed=10,
    num_outputs=4,
)
```

The sequence uses one random input for every parameter snapshot, making the
panels directly comparable. With an integer `random_seed`, multiple outputs use
consecutive seeds; `None` draws fresh random inputs. Each function returns the
Matplotlib figure and a manifest containing the run, selection, seed, and
available evaluation metadata.

Qualitative outputs are reconstructed with an ideal statevector even when the
checkpoint came from noisy or real execution. The original execution type is
kept in the manifest/title, but the generated image is not a new noisy or QPU
measurement.
