# Results Analysis Plan

This plan describes the analysis currently implemented in `notebooks/results_analysis.ipynb`. It treats the executed notebook as the source of truth and distinguishes the focused comparisons in the current analysis from the broader comparisons proposed in the previous plan.

## Analysis principles

- Main quality conclusions use completed 1000-epoch noiseless or noisy simulator runs.
- CPU and GPU copies of the same scientific simulator run are deduplicated for quality analysis. Hardware environment remains explicit in timing comparisons.
- KL-divergence results from the `base` preset are kept separate from the image-gradient evaluation used by `ang` and `amp`.
- Learning curves show individual runs faintly and summarize the seeds with the median and interquartile range.
- Best-score plots use box plots with individual seed points. Best score is interpreted together with the epoch at which it occurred.
- Short real-hardware runs and single-seed fake-real runs are treated as behavioral or implementation case studies, not statistical evidence.

## 1. Timing and Experimental Feasibility

The chapter starts with measured timing and feasibility so that the reader understands the resource constraints behind the completed experimental battery before interpreting model quality.

### 1.1 Matched device runtime

The timing comparison uses completed five-epoch timing runs. Median time per epoch is the main statistic because it is less sensitive to initialization overhead than the mean.

The main matched-runtime figure is rendered separately for `base`, `ang`, and
`amp`. Each figure retains paired noiseless and noisy panels. Its x-axis groups
runs by qubit count and gradient method (for example, q4 SPSA or q8 REG), while
its logarithmic y-axis reports median seconds per epoch. Vertical guides span
the available CPU/1, CPU/4, and GPU measurements within each run group. All
environment markers share the exact run-group x position, and direct labels
report speed-up relative to CPU/1.
Real-QPU measurements are added to the noisy panel where available, currently
the q4 SPSA and q4 PSR groups for `base`. Each panel shows only run groups with
observed data. Single-environment measurements remain visible, including noisy
q16 GPU timings, but have no speed-up label when CPU/1 is unavailable.

The matching spans:

- Encoding preset: `base`, `ang`, or `amp`, with one preset per figure.
- Execution type: noiseless and noisy simulation where available.
- Gradient method: SPSA, PSR, and REG.
- Qubit count: q4, q8, and q16 where available.
- Execution environment: one CPU per task, four CPUs per task, GPU execution labelled as RTX6000, and real hardware where available.

All panels use a logarithmic time axis. CPU/1, CPU/4, GPU, and Real QPU are
retained as distinct execution environments.

### 1.2 Epoch-time distributions

Configurable box-and-strip panels expose every recorded epoch time while
comparing CPU/1, CPU/4, and GPU measurements. Their arguments select encoding
preset, execution type, qubit count, gradient method, and randomness; any
argument can be a sequence to compare multiple values. The notebook shows the
pooled `rand0`/`rand1` comparison via `compare_by="randomness"`, using one panel
for each of CPU/1, CPU/4, GPU, and Real QPU and placing randomness on the x-axis.
The panels use a compact horizontal `1 × N` layout for any panel count by
default; `layout=(rows, columns)` provides an explicit grid override. It includes
all eligible presets, execution types, qubit counts, and gradient methods.
Direct scientific arguments remain available as optional filters, and omitting
`compare_by` retains controlled configuration panels. Empty comparison
categories and their slots are omitted independently from each device panel.
Panel widths scale with their displayed category counts, so the remaining
categories keep the same physical width across panels; panels with no timing
values are not shown. A prominent circle shows the median epoch time of every
run, with the run's individual epoch times shown as faint shadow circles behind
it. Epochs are treated as repeated measurements within one timing run, not as
independent replicates.

### 1.3 Experimental feasibility and limitations

The feasibility analysis expands the selected convergence and timing battery configurations and checks whether the corresponding training checkpoint exists and reaches the requested epoch budget.

The figures show:

- A completion matrix by qubit count and execution type.
- Counts of incomplete requested runs by limiting factor.

The reported limitations include time-expensive noisy PSR, unavailable real-hardware execution time, q16 noisy CPU memory limits, q16 noisy GPU allocation limits, and q16 amplitude-encoding transpilation limits. These are experimental-design constraints and are not interpreted as model-quality results.

Replication is stated once, compactly, in this subsection. The timing battery
uses seed 0 once per configuration and records five repeated epoch timings; its
epoch points are not independent replicates. The main convergence groups
contain three completed runs (seeds 0, 1, and 2). Exceptions report their actual
count with the corresponding result. The hardware comparison contains two
short `qml_torch` case studies, and fake-real implementation validation
contains one seed per implementation/packing configuration.

## 2. Encoding Presets

This section establishes the behavior of the three encoding/data presets and connects their quantitative results to generated outputs.

### 2.1 Learning dynamics and best results

The focused comparison fixes:

- `qml_torch` implementation.
- Noiseless simulation.
- PSR gradients.
- Four qubits.
- `rand0`.
- Seeds 0, 1, and 2.

Each preset is shown in a separate column so that KL divergence and image-gradient scores are never pooled. Generator loss, discriminator loss, and evaluation score are plotted over training. Individual seed traces remain visible beneath the median and interquartile range.

The associated performance figure reports the best evaluation score and epoch of the best evaluation, with individual seed points overlaid. A grouped table reports the number of completed runs, included seeds, and aggregate best-score statistics.

### 2.2 Initial, last, best, and target outputs

For each preset, the notebook renders the generator output at initialization, the final checkpoint, and the best checkpoint beside the target. The current executed selection is noiseless q4 PSR, `rand0`, training seed 0, with qualitative sampling seed 0.

These figures demonstrate how training changes the generated output and whether the last checkpoint regresses relative to the best checkpoint. The manifest printed with the figures links every example to its run identifier and evaluation score.

## 3. Execution-Type Comparison

### 3.1 Noiseless versus noisy simulation

The replicated comparison fixes `qml_torch`, SPSA, q4, `rand0`, and seeds 0–2. Each preset remains in its own evaluation-metric panel.

The figures report:

- Evaluation learning dynamics for noiseless and noisy simulation.
- Best evaluation score by execution type.
- Epoch of best evaluation by execution type.

### 3.2 Noiseless, noisy, and real hardware

The real-hardware analysis is a case study for the available q4 `base`/`qml_torch` runs. Simulator curves are truncated to the observed hardware epoch budget for the corresponding gradient method, avoiding a comparison between short hardware runs and full 1000-epoch simulator trajectories.

The notebook shows matched evaluation dynamics, best evaluation, epoch of best evaluation, and a table of real-run identifiers, completed epochs, and median time per epoch. The results are interpreted as behavioral and feasibility evidence rather than a replicated estimate of QPU performance.

## 4. Gradient Method Comparison

The main gradient comparison fixes `base`, `qml_torch`, noiseless execution, q4, `rand0`, and KL evaluation. SPSA, PSR, and REG are compared through:

- Evaluation learning dynamics.
- Best evaluation score with seed points.
- Epoch of best evaluation.

Gradient runtime is not repeated in this section because it is included in the timing analysis. The scaling section provides a separate view of gradient behavior across qubit counts for the `base` preset.

## 5. Randomness Effect

### 5.1 Complete five-level sweep

The notebook identifies configurations containing a complete grid of randomness values `0`, `0.1`, `0.25`, `0.5`, and `1` for seeds 0, 1, and 2. `RANDOMNESS_PRESET` selects `base`, `ang`, or `amp`; `RANDOMNESS_GRADIENT_METHOD` selects PSR, REG, or SPSA; and `RANDOMNESS_SWEEP_INDEX` selects the q4 or q8 sweep. Thus, each preset has six selectable complete sweeps: three gradient methods at two qubit counts. There are 18 complete sweeps across all three presets. The currently rendered selection is `amp`, PSR, q4.

The figures show:

- Evaluation learning dynamics for all five randomness levels.
- Best evaluation score.
- Epoch of best evaluation.
- Evaluation-step volatility, defined as the median absolute change between consecutive evaluation scores.

### 5.2 Fresh samples from a selected randomized run

The executed example uses the best parameters from the completed `amp`, noiseless q4 PSR, randomness 0.25, training-seed-0 run. Four fresh outputs are displayed beside the target. The manifest records the run, parameter set, input-seed policy, score, and evaluation epoch.

### 5.3 Wider paired rand0/rand1 evidence

The wider analysis constructs matched `rand0` and `rand1` pairs for `base`, `qml_torch`, noiseless KL runs. It plots paired changes in:

- Best evaluation score.
- Epoch of best evaluation.
- Evaluation-step volatility.

The pairing prevents unrelated configurations from being treated as randomness comparisons.

## 6. Scaling Analysis

Scaling is separated into four focused comparisons. Every subsection shows representative evaluation dynamics and best-score/epoch-of-best scaling across the available qubit counts.

`SCALING_SCORE_TRANSFORM` controls the convergence-curve display. `none` shows raw evaluation scores, `normalize` applies per-run min-max scaling to `[0, 1]`, and `standardize` applies a per-run z-score. Transformations make trajectory shapes easier to compare when raw metric scales differ, but they remove absolute-quality information. Therefore, the best-score summary panels always retain raw evaluation values.

### 6.1 Scaling by preset

Noiseless `qml_torch`, PSR, and `rand0` are fixed. The `base` and `ang` presets use available q4, q8, and q16 results; `amp` is limited to q4 and q8.

### 6.2 Scaling by execution type

The comparison fixes `ang`, `qml_torch`, SPSA, and `rand0`. Noiseless and noisy execution are faceted separately, and no noisy q16 conclusion is inferred.

### 6.3 Scaling by gradient method

The comparison fixes `base`, `qml_torch`, noiseless execution, and `rand0`. PSR, REG, and SPSA are faceted separately across their available qubit counts.

### 6.4 Scaling by randomness

The comparison fixes `ang`, `qml_torch`, noiseless execution, and PSR. Its first figure facets all five randomness levels and directly compares q4 with q8 in every panel. A second focused dynamics figure compares `rand0` with `rand1` across the available qubit counts, followed by raw best-score and epoch-of-best summaries.

## 7. Implementation and Packing Validation

The final section validates the alternative implementation paths using fake-real q4, SPSA, `rand0`, seed-0 runs for all three presets.

The evaluation curves compare:

- `qml_torch`.
- `runtime_packed/separate`.
- `runtime_packed/joined` where the packing mode is valid.

The accompanying table records run identifier, preset, implementation/packing mode, completed epochs, best evaluation, and epoch of best evaluation. Because this is a single-seed fake-real experiment, it supports implementation and behavioral validation only; it is not used for statistical quality or hardware-performance claims.

## Executive summary

The current results analysis contains 387 completed, device-deduplicated simulator runs, 226 completed five-epoch timing runs, two `qml_torch` real-hardware case-study runs, and eight usable implementation-validation runs. Its narrative begins with timing and feasibility, then examines preset behavior, execution noise and hardware, gradient choice, input randomness, qubit scaling, and implementation/packing validation.

The strongest statistical evidence comes from completed simulator runs with individual seed traces and median/IQR summaries. The notebook keeps incompatible evaluation metrics separate and treats limited QPU and fake-real evidence conservatively. Its central performance statistic is currently the best evaluation score, supported by the epoch at which that score occurred and, for randomness, trajectory volatility.

## Differences and omissions relative to the previous plan

The following items from the previous plan are not part of the current implemented scope or remain unreported.

### Changed scope or organization

- Timing and feasibility now come first; the previous plan began with learning dynamics and final quality.
- Runtime-scaling and projected 1000-epoch-cost figures are omitted from the main narrative because the matched timing and epoch distributions already expose the measured evidence without adding an extrapolated summary.
- There is no single comprehensive final-performance section. Best-score summaries are distributed across preset, execution, gradient, randomness, and scaling sections.
- The main gradient comparison is a focused `base` q4 `rand0` analysis rather than the full preset × qubit × randomness matrix proposed previously.
- The main noiseless/noisy comparison is q4. Qubit-dependent execution behavior is studied separately using the `ang` preset.
- Implementation packing is framed only as training-path validation. The notebook deliberately avoids a strong quality or timing claim from single-seed fake-real runs.
- Qualitative examples are explicitly selected through notebook parameters rather than automatically selected as best, worst, or median runs.

### Metrics and figures still missing from the previous plan

- Median evaluation over the last 50 or 100 epochs as a prominently reported result. This is now considered an optional late-training robustness check rather than required evidence of convergence.
- Simultaneous rendering of all six complete five-level randomness sweeps for one preset. Section 5.1 instead provides explicit preset, gradient-method, and q4/q8 selectors, while Section 6.4 renders the complete five-level q4/q8 comparison for its selected gradient method.
- Full gradient comparisons for `base`, `ang`, and `amp` at q4/q8, relevant q16 cases, and both `rand0` and `rand1`.
- Matched noisy q4/q8 comparisons for all presets; noisy scaling is currently focused on `ang`.
- Time per circuit evaluation, primitive call, or submission. The current checkpoints do not contain the required `primitive_calls`, `circuit_evaluations`, or `submissions` counters.
- A packing-overhead figure comparing joined and separate primitive/submission cost.
- Objective qualitative selection of the best median-final-window run, worst completed run, and median run.
- Representative generated outputs for both noisy and noiseless simulation and for real hardware.

### Current notebook consistency notes

- Figures are shown inline, but `EXPORT_FIGURES` is currently `False`. PDF and 300-dpi PNG files will not be written until export is enabled.
