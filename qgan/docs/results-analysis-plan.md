# Results Analysis Plan

This plan describes the analysis currently implemented in `notebooks/results_analysis.ipynb`. It treats the executed notebook as the source of truth and distinguishes the focused comparisons in the current analysis from the broader comparisons proposed in the previous plan.

## Analysis principles

- Main quality conclusions use completed 1000-epoch noiseless or noisy simulator runs.
- CPU and GPU copies of the same scientific simulator run are deduplicated for quality analysis. Hardware environment remains explicit in timing comparisons.
- KL-divergence results from the `base` preset are kept separate from the image-gradient evaluation used by `ang` and `amp`.
- Learning curves show individual runs faintly and summarize the seeds with the median and interquartile range.
- Best-score plots use box plots with individual seed points. Best score is interpreted together with the epoch at which it occurred.
- Short real-hardware runs and single-seed fake-real runs are treated as behavioral or implementation case studies, not statistical evidence.

## 1. Time, Cost, and Feasibility

The chapter starts with computational cost and feasibility so that the reader understands the resource constraints behind the completed experimental battery.

### 1.1 Time analysis

The timing comparison uses completed five-epoch timing runs. Median time per epoch is the main statistic because it is less sensitive to initialization overhead than the mean.

The main figure compares median epoch time across:

- Encoding preset: `base`, `ang`, and `amp`.
- Execution type: noiseless and noisy simulation where available.
- Gradient method: SPSA, PSR, and REG.
- Input randomness: principally `rand0` and `rand1`.
- Execution environment: one CPU per task, four CPUs per task, and GPU execution labelled as RTX6000.

The comparisons use a fixed reference configuration for factors not varied in each panel: `ang`, `qml_torch`, noiseless execution, SPSA, four qubits, and `rand0`.

### 1.2 Experimental limitations and feasibility

The feasibility analysis expands the selected convergence and timing battery configurations and checks whether the corresponding training checkpoint exists and reaches the requested epoch budget.

The figures show:

- A completion matrix by qubit count and execution type.
- Counts of incomplete requested runs by limiting factor.

The reported limitations include time-expensive noisy PSR, unavailable real-hardware execution time, q16 noisy CPU memory limits, q16 noisy GPU allocation limits, and q16 amplitude-encoding transpilation limits. These are experimental-design constraints and are not interpreted as model-quality results.

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

For each preset, the notebook renders the generator output at initialization, the final checkpoint, and the best checkpoint beside the target. The current executed selection is noiseless q4 PSR, `rand0`, training seed 1, with qualitative sampling seed 0.

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

Gradient runtime is not repeated in this section because it is included in the timing analysis. The scaling section provides a separate view of gradient behavior across qubit counts for the `ang` preset.

## 5. Randomness Effect

### 5.1 Complete five-level sweep

The notebook identifies configurations containing a complete grid of randomness values `0`, `0.1`, `0.25`, `0.5`, and `1` for seeds 0, 1, and 2. The currently rendered sweep is the first matched `base` sweep: noiseless q4 PSR with KL evaluation.

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

### 6.1 Scaling by preset

Noiseless `qml_torch`, SPSA, and `rand0` are fixed. The `base` and `ang` presets use available q4, q8, and q16 results; `amp` is limited to q4 and q8.

### 6.2 Scaling by execution type

The comparison fixes `ang`, `qml_torch`, SPSA, and `rand0`. Noiseless and noisy execution are faceted separately, and no noisy q16 conclusion is inferred.

### 6.3 Scaling by gradient method

The comparison fixes `ang`, `qml_torch`, noiseless execution, and `rand0`. PSR, REG, and SPSA are faceted separately across their available qubit counts.

### 6.4 Scaling by randomness

The comparison fixes `ang`, `qml_torch`, noiseless execution, and SPSA. `rand0` and `rand1` are faceted separately across the available qubit counts.

## 7. Implementation and Packing Validation

The final section validates the alternative implementation paths using fake-real q4, SPSA, `rand0`, seed-0 runs for all three presets.

The evaluation curves compare:

- `qml_torch`.
- `runtime_packed/separate`.
- `runtime_packed/joined` where the packing mode is valid.

The accompanying table records run identifier, preset, implementation/packing mode, completed epochs, best evaluation, and epoch of best evaluation. Because this is a single-seed fake-real experiment, it supports implementation and behavioral validation only; it is not used for statistical quality or hardware-performance claims.

## Executive summary

The current results analysis contains 374 completed, device-deduplicated simulator runs, 224 completed five-epoch timing runs, three usable real-hardware runs, and eight usable implementation-validation runs. Its narrative begins with cost and feasibility, then examines preset behavior, execution noise and hardware, gradient choice, input randomness, qubit scaling, and implementation/packing validation.

The strongest statistical evidence comes from completed simulator runs with individual seed traces and median/IQR summaries. The notebook keeps incompatible evaluation metrics separate and treats limited QPU and fake-real evidence conservatively. Its central performance statistic is currently the best evaluation score, supported by the epoch at which that score occurred and, for randomness, trajectory volatility.

## Differences and omissions relative to the previous plan

The following items from the previous plan are not part of the current implemented scope or remain unreported.

### Changed scope or organization

- Timing and feasibility now come first; the previous plan began with learning dynamics and final quality.
- There is no single comprehensive final-performance section. Best-score summaries are distributed across preset, execution, gradient, randomness, and scaling sections.
- The main gradient comparison is a focused `base` q4 `rand0` analysis rather than the full preset × qubit × randomness matrix proposed previously.
- The main noiseless/noisy comparison is q4. Qubit-dependent execution behavior is studied separately using the `ang` preset.
- Implementation packing is framed only as training-path validation. The notebook deliberately avoids a strong quality or timing claim from single-seed fake-real runs.
- Qualitative examples are explicitly selected through notebook parameters rather than automatically selected as best, worst, or median runs.

### Metrics and figures still missing from the previous plan

- Final evaluation score plots.
- Median evaluation over the last 50 or 100 epochs as a plotted and prominently reported result.
- Completed-run counts for every grouped comparison, rather than only selected tables.
- Final-window evaluation versus randomness.
- Rendering or summarizing all six detected complete five-level randomness sweeps; the notebook currently displays only one selected sweep.
- Full gradient comparisons for `base`, `ang`, and `amp` at q4/q8, relevant q16 cases, and both `rand0` and `rand1`.
- Matched noisy q4/q8 comparisons for all presets; noisy scaling is currently focused on `ang`.
- Total training-time figures and timing versus qubit count.
- Time per circuit evaluation, primitive call, or submission. The current checkpoints do not contain the required `primitive_calls`, `circuit_evaluations`, or `submissions` counters.
- A packing-overhead figure comparing joined and separate primitive/submission cost.
- Objective qualitative selection of the best median-final-window run, worst completed run, and median run.
- Representative generated outputs for both noisy and noiseless simulation and for real hardware.
- Inclusion or explicit exclusion rationale for the third usable real-hardware run, `base-runtime_packed-separate-q4-real-SPSA-aerCPU-rand0-seed0`; the current hardware comparison filters to `qml_torch` and displays two runs.

### Current notebook consistency notes

- The qualitative-output prose says the default training seed is 0, while the executed code sets `OUTPUT_SEED = 1`. The prose or code should be changed so they agree.
- The notebook identifies three usable real-hardware runs in its opening summary but calls the hardware subsection a comparison of the two available real-QPU runs. This is correct only after stating that the subsection is restricted to `qml_torch`.
- Figures are shown inline, but `EXPORT_FIGURES` is currently `False`. PDF and 300-dpi PNG files will not be written until export is enabled.
