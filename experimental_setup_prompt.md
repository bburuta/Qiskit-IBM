# Experimental Setup factual prompt

Use the following factual information to produce the Experimental Setup section for the Fully Quantum Generative Adversarial Network thesis. The experiment definitions and selections correspond to the code executed in `qgan/notebooks/results_analysis.ipynb`.

## Experiment data used by the results notebook

The results notebook loads two collections:

1. `qgan/data/train`
   - Convergence experiments.
   - Real-hardware case studies.
   - Fake-real implementation-validation runs.
2. `qgan/data/train/times`
   - Dedicated five-epoch timing experiments.

The notebook currently identifies:

- 717 raw convergence or case-study configurations.
- 387 completed, device-deduplicated simulator runs.
- 226 completed five-epoch timing runs.
- 2 usable physical-hardware runs.
- 8 usable fake-real implementation-validation runs.
- 235 completed simulator runs evaluated with the image-gradient metric.
- 152 completed simulator runs evaluated with KL divergence.

A simulator run enters the main quality analysis only if:

- Its execution type is noiseless or noisy.
- It contains at least 1000 evaluation epochs.
- The largest recorded epoch is at least 999.
- Its checkpoint loads successfully.

CPU and GPU executions of the same scientific configuration are treated as duplicate quality experiments. One representative checkpoint, preferably the CPU checkpoint, is retained. The compute device is not discarded in the timing analysis.

The scientific identity of a run includes:

- Encoding preset.
- Implementation.
- Packing strategy where applicable.
- Execution type.
- Gradient method.
- Number of qubits.
- Randomizer circuit.
- Randomness level.
- Training and evaluation batch sizes.
- Evaluation metric.
- Learning rate.
- Training budget.
- Generator and discriminator update counts.
- Estimator precision.
- Simulation methods.
- Noisy-backend mapping.

When randomness is zero, the configured randomizer circuit does not distinguish scientific runs because the randomizer is disabled.

## QGAN architecture

The main implementation is `qml_torch`, built with Qiskit Machine Learning's `EstimatorQNN` and `TorchConnector`.

Three QNN roles are created during training:

- Generator QNN: differentiates the generator parameters in the randomizer-generator-discriminator circuit while keeping discriminator and random-input parameters fixed.
- Fake-data discriminator QNN: differentiates discriminator parameters using generated data while keeping generator and random-input parameters fixed.
- Real-data discriminator QNN or ensemble: differentiates discriminator parameters using real-data preparation circuits.

The training observable is the Pauli-Z operator on the final logical qubit. Generator and discriminator training outputs are therefore expectation values nominally in `[-1,1]`.

### Generator circuit

The generator uses Qiskit's `RealAmplitudes` ansatz:

- Number of repetitions: 3.
- Entanglement: reverse linear.
- Number of trainable parameters: `4n` for `n` logical qubits.
- Four-qubit generator: 16 parameters.
- Eight-qubit generator: 32 parameters.
- Sixteen-qubit generator: 64 parameters.

### Discriminator circuit

The discriminator begins with Qiskit's `EfficientSU2` ansatz:

- Number of repetitions: 1.
- Entanglement: reverse linear.
- Default RY and RZ rotation structure.

It is extended by:

- A CNOT from every other qubit to the final qubit, applied in reverse qubit order.
- One additional trainable RY rotation on the final qubit.
- One additional trainable RZ rotation on the final qubit.

The discriminator contains `4n+2` trainable parameters:

- Four-qubit discriminator: 18 parameters.
- Eight-qubit discriminator: 34 parameters.
- Sixteen-qubit discriminator: 66 parameters.

The total generator-plus-discriminator parameter counts are:

- Four qubits: 34.
- Eight qubits: 66.
- Sixteen qubits: 130.

### Circuit transpilation

Every execution backend uses a Qiskit preset pass manager with:

- Optimization level 3.
- SABRE layout selection.
- SABRE routing.
- Transpiler seed equal to the run seed.

For noisy and hardware execution, transpilation uses the target, basis gates, and coupling constraints of the corresponding physical backend or saved backend snapshot.

## Data and encoding presets

Three presets are present: `base`, `ang`, and `amp`.

### Base preset

The base preset learns a fixed quantum probability distribution.

Dataset:

- Type: quantum.
- Source: `specific_distribution`.
- Number of real-data preparation circuits: 1.

The target circuit:

- Starts in `|0...0>`.
- Applies Hadamard gates to qubits 0 through `n-2`.
- Applies a CNOT with qubit `n-2` as control and qubit `n-1` as target.
- Produces a uniform distribution over `2^(n-1)` supported basis states, with the final two qubits correlated.

Encoding and batches:

- The target circuit is used directly.
- No classical image encoding is performed.
- Training batch size: 1.
- Evaluation batch size: 1.
- Evaluation metric: KL divergence.

### Angle preset

The angle preset learns the spatial structure of procedurally generated grayscale gradient images.

Dataset:

- Type: classical.
- Source: `generated_gradients`.
- Number of images: 7.
- Gradient direction: top-left to bottom-right.
- Pixel values lie in `[0,1]`.

The seven gradient curves are:

- Linear.
- Quadratic.
- Square root.
- Logarithmic.
- Exponential.
- Sigmoid.
- Sinusoidal.

The number of pixels equals the number of logical qubits:

- Four qubits: 4 pixels arranged as 2x2.
- Eight qubits: 8 pixels arranged as 2x4.
- Sixteen qubits: 16 pixels arranged as 4x4.

Encoding and batches:

- Each pixel value is used directly as the angle of one RY rotation.
- Pixel values are not multiplied by pi in the current implementation.
- One logical qubit represents one pixel.
- Training batch size: 4.
- Evaluation batch size: 8.
- Real training images are sampled randomly with replacement.

Evaluation:

- Each logical qubit is evaluated through an ideal Pauli-Z expectation value.
- The expectation values are reshaped into the corresponding image dimensions.
- The image-gradient metric is applied.
- The target image is not used directly by this evaluation metric.

### Amplitude preset

The amplitude preset uses the same seven generated grayscale gradient images.

The number of pixels equals the Hilbert-space dimension `2^n`:

- Four qubits: 16 pixels arranged as 4x4.
- Eight qubits: 256 pixels arranged as 16x16.
- A nominal sixteen-qubit case would require 65,536 pixels arranged as 256x256.

For each image with non-negative pixel values `x_i` and contrast exponent `c=1`:

```math
p_i = \frac{x_i^c}{\sum_j x_j^c},
\qquad
a_i = \sqrt{p_i}.
```

The normalized amplitudes `a_i` are loaded using a quantum state-preparation circuit.

Encoding and batches:

- Training batch size: 4.
- Evaluation batch size: 8.
- A training batch selects four of the seven real-image preparation circuits.

Evaluation:

- Exact generated computational-basis probabilities are obtained from ideal simulation.
- The probability vector is reshaped into an image.
- The image-gradient metric is applied.
- The target image is not used directly by this metric.

The sixteen-qubit amplitude configuration is not part of the completed analysis because its state-preparation circuits were too large and deep to transpile.

## Random generator input

The generator can be preceded by a randomizer circuit. For the analyzed experiments, the active randomizer contains one parameterized RY gate per qubit. If the randomness level is `r`, every randomizer parameter is sampled independently as:

```math
\theta_i \sim \operatorname{Uniform}(0,2\pi r).
```

The complete randomness sweep contains:

```math
r \in \{0,0.1,0.25,0.5,1\}.
```

At `r=0`:

- The randomizer circuit is disabled.
- The generator receives `|0...0>`.
- Repeated generated outputs are identical for a fixed generator parameter vector because no random input is applied.

At `r>0`:

- A fresh batch of random angles is generated during training and evaluation.
- `r` controls the range of the input rotations.
- `r=1` permits angles across `[0,2pi)` but does not produce Haar-random quantum states.

## Training protocol

The main simulator convergence experiments use:

- Maximum training budget: 1000 epochs.
- Discriminator updates per epoch: 1.
- Generator updates per epoch: 1.
- Generator-quality evaluations per epoch: 1.
- Generator optimizer: Adam.
- Discriminator optimizer: Adam.
- Learning rate for both optimizers: 0.005.
- Numerical type: double precision.
- Main training seeds: 0, 1, and 2.

One epoch consists of:

1. One discriminator update.
2. One generator update.
3. One generator-quality evaluation.
4. Recording generator loss, discriminator loss, evaluation score, and elapsed wall-clock time.
5. Updating the stored best generator parameters if the new evaluation score is lower.

There is no performance-based early stopping. The epoch of the minimum evaluation score is recorded retrospectively.

Parameter initialization is:

```math
\theta \sim \operatorname{Uniform}(-\pi,\pi) \times 0.1,
```

so the initial trainable parameters lie in `[-0.1pi,0.1pi]`.

The seed controls:

- Python randomness.
- NumPy randomness.
- PyTorch CPU randomness.
- PyTorch CUDA randomness where available.
- Aer simulator randomness.
- Transpiler randomness.
- SPSA perturbations.

Checkpoints preserve:

- Current epoch.
- Generator and discriminator parameters.
- Adam optimizer states.
- Initial generator parameters.
- Best generator parameters.
- Generator and discriminator losses.
- Evaluation scores.
- Per-epoch execution times.
- Python, NumPy, PyTorch, and CUDA random states.

## Adversarial objectives

For discriminator output `D(x)`, the internal losses are:

```math
L_D = -\mathbb{E}[D(x_{\mathrm{real}})]
      +\mathbb{E}[D(G(z))],

L_G = -\mathbb{E}[D(G(z))].
```

The recorded losses are rescaled as:

```math
L_{G,\mathrm{recorded}} = \frac{L_G-1}{2},
\qquad
L_{D,\mathrm{recorded}} = \frac{L_D-2}{4}.
```

They therefore lie approximately in `[-1,0]`. These losses describe adversarial training behavior and are distinct from the generator-quality metrics.

## Gradient methods

### Parameter-shift rule

- Identifier: `PSR`.
- Implemented with `ParamShiftEstimatorGradient`.
- Computes parameter derivatives through shifted circuit evaluations.
- Supported in noiseless, noisy, fake-real, and real-hardware `qml_torch` experiments.
- Its circuit-evaluation cost grows with the number of differentiated parameters.

### Simultaneous perturbation stochastic approximation

- Identifier: `SPSA`.
- Implemented with `SPSAEstimatorGradient`.
- Perturbation epsilon: 0.01.
- SPSA seed: run seed.
- Estimates the gradient through simultaneous parameter perturbations.
- Used for noiseless, noisy, fake-real, and hardware experiments.

### Reverse estimator gradient

- Identifier: `REG`.
- Implemented with `ReverseEstimatorGradient`.
- Restricted to compatible local noiseless `qml_torch` execution.
- Not used for noisy or real-hardware execution.

## Execution environments

### Noiseless simulation

Noiseless training uses:

- Local Qiskit `AerSimulator`.
- Statevector method.
- Double precision.
- Aer `EstimatorV2`.
- No noise model.
- No finite-shot sampling.
- Exact expectation values.
- Simulator seed equal to the run seed.

The configured estimator precision is ignored for noiseless training because the backend is explicitly configured with no shots and precision zero.

### Noisy simulation

Noisy training uses:

- Local Qiskit `AerSimulator`.
- Density-matrix method.
- Double precision.
- Aer `EstimatorV2`.
- Requested precision: 0.015625.
- Shot count in local simulation: `1/(0.015625)^2 = 4096`.
- Simulator seed equal to the run seed.

The noisy backend is constructed from a cached calibration snapshot of `ibm_basquecountry`. The simulator receives:

- The backend-derived noise model.
- Physical coupling map.
- Backend basis gates.
- Backend target.

The configured mapping mode is `hardware`, so the experiments include both calibrated noise and hardware-connectivity constraints. This differs from a `noise_model`-only simulation in which a small circuit would not be mapped onto the complete backend topology.

The relevant calibration snapshot is time-dependent. Cached backend files exist under `qgan/backends` and beside the real-hardware configurations. Their filesystem timestamps are from September 2026, but the exact calibration acquisition timestamps have not been established independently.

### Common evaluation environment

Training execution and generator-quality evaluation use different backends.

After every epoch, evaluation is always performed on a separate ideal noiseless Aer simulator, even if training uses:

- Noisy simulation.
- Fake-real execution.
- Physical quantum-hardware execution.

For base and amplitude encoding:

- The randomizer-generator circuit is transpiled for an ideal evaluation backend.
- A `SaveProbabilities` instruction obtains exact computational-basis probabilities.

For angle encoding:

- The randomizer-generator circuit is evaluated with one Pauli-Z observable per logical qubit.
- Exact ideal expectation values are obtained.

Consequently, evaluation curves measure the quality of the generator parameter vector under a common ideal evaluation environment. They do not directly measure the output quality that would be observed by repeatedly sampling the trained circuit on noisy hardware.

### Physical quantum hardware

The physical backend is:

- Backend name: `ibm_basquecountry`.
- Processor family: IBM Heron.
- Processor revision: 2.
- Physical width: 156 qubits.
- Native two-qubit interaction: CZ.

Runtime execution uses:

- `QiskitRuntimeService`.
- IBM Runtime `EstimatorV2`.
- Runtime `Session`.
- Maximum session time: 8 hours.
- Requested default precision: 0.015625.
- Resilience level: 1.
- Dynamical decoupling enabled.
- Optimization level 3.
- SABRE layout and routing.

The physical-hardware analysis uses only `qml_torch` runs. Both runs use:

- Base preset.
- Four logical qubits.
- No generator-input randomness.
- Seed 0.
- KL evaluation on the separate ideal evaluation backend.

The two available hardware cases are:

- PSR: 190 completed epochs.
- SPSA: 238 completed epochs.

Their saved configurations request 1000 epochs, but execution ended before this budget because of hardware-session and execution-time constraints. They enter the hardware case-study population because they contain usable metrics; they do not enter the completed 1000-epoch simulator population.

The exact physical qubits selected by SABRE, transpiled circuit depth, CZ count, and precise calibration acquisition time are not currently summarized in the notebook.

### Fake-real execution

The `fake_real` mode uses:

- Qiskit's `FakeSherbrooke` backend.
- Hardware-style Runtime estimator execution.
- Hardware-aware transpilation.
- No physical IBM Quantum submission.

Its purpose is to validate alternative execution implementations and circuit-packing strategies. It is not part of the real-hardware evidence and is excluded from the feasibility matrix.

## Implementations

`qml_torch` uses:

- Qiskit Machine Learning `EstimatorQNN`.
- PyTorch `TorchConnector`.
- Separate QNN objects for generator, fake-data discriminator, and real-data discriminator branches.

`runtime_packed` uses:

- A primitive-backed implementation.
- Packed compatible same-template evaluations.
- Separate or joined discriminator execution.
- CPU-hosted PyTorch parameters.
- Separate packing for base, angle, and amplitude presets.
- Joined packing for base and angle only.
- No joined amplitude packing.

## Convergence experiment battery

The feasibility analysis expands `train_conv_gpu.yaml` as the representative convergence battery.

Requested noiseless configurations for four and eight qubits:

- Presets: base, ang, amp.
- Gradients: PSR, SPSA, REG.
- Randomness: 0, 0.1, 0.25, 0.5, 1.
- Seeds: 0, 1, 2.
- Requested configurations: 270.

Requested noiseless configurations for sixteen qubits:

- Presets: base and ang.
- Gradients: PSR, SPSA, REG.
- Randomness: 0 and 1.
- Seeds: 0, 1, 2.
- Requested configurations: 36.

Total noiseless convergence status:

- Requested: 306.
- Complete: 305.
- Not complete: 1.

Requested noisy configurations for four and eight qubits:

- Presets: base, ang, amp.
- Gradients: SPSA and PSR.
- Randomness: 0 and 1.
- Seeds: 0, 1, 2.
- Requested configurations: 72.

Requested noisy configurations for sixteen qubits:

- Presets: base and ang.
- Gradients: SPSA and PSR.
- Randomness: 0 and 1.
- Seeds: 0, 1, 2.
- Requested configurations: 24.

Total noisy convergence status:

- Requested: 96.
- Complete: 37.
- Not complete: 59.

Additional completed historical configurations are present in `qgan/data/train`. This is why the final device-deduplicated simulator population contains 387 completed runs rather than only the completed cells of the currently selected feasibility battery.

## Timing experiments

A timing run is usable when it contains five completed epoch-time measurements.

The timing analysis includes:

- 226 completed five-epoch runs.
- Seed 0.
- Randomness levels 0 and 1.
- Noiseless and noisy execution.
- Four, eight, and sixteen qubits where feasible.
- PSR, SPSA, and REG for noiseless execution.
- PSR and SPSA for noisy execution.
- Base, angle, and amplitude presets where feasible.

Timing environments are classified as follows.

CPU/1:

- CPU simulator execution.
- No `cpu4` label.
- Typically one allocated CPU for the dedicated CPU timing battery.

CPU/4:

- CPU simulator execution.
- Run label `cpu4` or four configured simulator threads.

GPU:

- Simulator or PyTorch device is GPU.
- The notebook labels this environment `RTX6000`.
- `RTX6000` is a manually configured analysis label rather than hardware metadata loaded automatically from every checkpoint.

Real QPU:

- Execution type is `real`.

The dedicated Slurm timing configurations specify:

- CPU/1 timing job: one CPU per task and 16 GB memory.
- RTX6000 timing job: one RTX6000 GPU, four CPUs per task, and 16 GB memory.

The exact physical CPU model, CUDA version, GPU driver, and operating-system image are not recorded in the notebook.

For each timing run:

- Every epoch time remains a repeated measurement.
- The run-level timing statistic is the median time per epoch.
- Epochs are not independent experimental replicates.
- The first epoch may include initialization overhead, motivating the median.

The representative feasibility matrix contains:

- Noiseless timing: 96 requested and 96 complete.
- Noisy timing: 56 requested and 56 complete.
- Real timing: 9 requested and 2 usable.

The 226-run timing population additionally includes completed CPU/4 and historical timing configurations beyond the selected battery matrix.

A projected 1000-epoch duration is:

```math
T_{1000} = 1000 \times \operatorname{median}(T_{\mathrm{epoch}}).
```

This is a linear extrapolation from five measured epochs rather than an observed 1000-epoch duration.

## Analyses performed by `results_analysis.ipynb`

### Matched runtime by preset

Three separate matched-runtime comparisons are produced for base, angle, and amplitude.

For each preset:

- Implementation: `qml_torch`.
- Randomness: 0.
- Separate noiseless and noisy panels.
- Runs grouped by qubit count and gradient method.
- CPU/1, CPU/4, GPU, and available Real-QPU measurements compared.
- Real-QPU measurements placed in the noisy panel for matching base q4 gradient groups.
- Speed-up calculated relative to the matched CPU/1 run.
- Groups without CPU/1 remain visible but have no speed-up value.

### Raw timing distributions

Timing distributions are examined by:

- Encoding preset.
- Execution type at q4.
- Execution type at q8.
- Execution type at q16.
- Randomness level.

These views pool remaining eligible timing factors unless they are explicitly fixed, so they are descriptive population views rather than controlled single-factor experiments.

### Runtime scaling

The implemented `runtime_scaling()` analysis includes:

- `qml_torch`.
- Noiseless and noisy simulation.
- Four, eight, and sixteen qubits.
- CPU/1, CPU/4, and GPU.
- All available presets, gradient methods, and randomness levels.

Each thin trajectory represents a workload matched across qubit counts. The workload is identified by all scientific fields except qubit count. The heavy summary at each qubit count is the median and IQR across available timing configurations.

The notebook contains an older markdown statement that runtime scaling fixes angle encoding, SPSA, and zero randomness. This does not match the current `runtime_scaling()` call; the implemented analysis uses the broader timing population above.

### Projected training cost

The simulator projection uses:

- Angle preset.
- `qml_torch`.
- Noiseless execution.
- SPSA.
- Four qubits.
- Randomness 0 and 1.
- CPU/1, CPU/4, and GPU.
- Five measured epochs per run.

The Real-QPU entries use:

- Base preset.
- `qml_torch`.
- Four qubits.
- Randomness 0.
- PSR and SPSA.

Real-QPU speed-up relative to CPU/1 is not calculated because the hardware runs do not form a controlled environment-scaling experiment.

### Feasibility analysis

The convergence population is based on `train_conv_gpu.yaml`. The timing population is based on:

- `train_times_cpu.yaml`.
- `train_times_gpu.yaml`.
- `train_times_rh.yaml`.

A requested run is complete only if its checkpoint reaches the epoch budget specified by the relevant battery. Fake-real runs are excluded.

Recorded limitations are:

- q4 noisy PSR: comprehensive convergence was too time-expensive.
- q8 noisy PSR: comprehensive convergence was too time-expensive.
- q16 amplitude: state preparation was too large and deep to transpile.
- q16 noisy CPU: density-matrix simulation exceeded available memory.
- q16 noisy GPU: projected execution exceeded the available allocation.
- Real hardware: insufficient execution time for the planned timing matrix.

### Preset comparison

The preset comparison fixes:

- Implementation: `qml_torch`.
- Execution: noiseless.
- Gradient: PSR.
- Qubits: 4.
- Randomness: 0.
- Seeds: 0, 1, 2.

The compared presets are base, angle, and amplitude. The analysis contains generator loss, discriminator loss, evaluation score, best evaluation, and epoch of best evaluation. Base uses KL divergence; angle and amplitude use the image-gradient metric.

### Initial, final, best, and target outputs

The qualitative preset examples use:

- `qml_torch`.
- Noiseless execution.
- Four qubits.
- PSR.
- Randomness 0.
- Training seed 0.
- Qualitative sampling seed 0.

The selected runs are:

- `base-qml_torch-q4-noiseless-PSR-aerCPU-rand0-seed0`.
- `ang-qml_torch-q4-noiseless-PSR-aerCPU-rand0-seed0`.
- `amp-qml_torch-q4-noiseless-PSR-aerCPU-rand0-seed0`.

For each run, the notebook reconstructs the generator output at initialization, the final checkpoint, the stored best parameters, and the target.

### Noiseless versus noisy simulation

The execution comparison fixes:

- Implementation: `qml_torch`.
- Gradient: SPSA.
- Qubits: 4.
- Randomness: 0.
- Seeds: 0, 1, 2.

Noiseless and noisy execution are compared separately for base, angle, and amplitude. The analysis contains evaluation trajectories, best evaluation scores, and epochs of best evaluation.

### Real-hardware comparison

The hardware comparison fixes:

- Implementation: `qml_torch`.
- Preset: base.
- Qubits: 4.
- Randomness: 0.
- Evaluation metric: KL divergence.
- Hardware seed: 0.

Separate PSR and SPSA cases are produced. Matching completed noiseless and noisy simulator trajectories are truncated to the observed hardware budget:

- PSR: 190 epochs.
- SPSA: 238 epochs.

The hardware sample size is one run per gradient method.

### Gradient-method comparison

The actual notebook code fixes:

- Preset: amplitude.
- Implementation: `qml_torch`.
- Execution: noiseless.
- Qubits: 4.
- Randomness: 0.
- Evaluation metric: image-gradient.
- Seeds: 0, 1, 2.

PSR, REG, and SPSA are compared. This differs from the older analysis plan that described a base-preset comparison; the current notebook uses amplitude.

### Complete randomness sweep

The currently rendered randomness sweep fixes:

- Preset: angle.
- Gradient: PSR.
- Execution: noiseless.
- Evaluation metric: image-gradient.
- Seeds: 0, 1, 2.

It compares randomness levels 0, 0.1, 0.25, 0.5, and 1. Two complete matched sweeps satisfy these filters: q4 and q8. Sweep index 0 currently selects q4.

The analysis contains evaluation trajectories, best evaluation score, epoch of best evaluation, and evaluation-step volatility.

### Fresh randomized outputs

The fresh-output example uses:

- Preset: amplitude.
- Implementation: `qml_torch`.
- Execution: noiseless.
- Gradient: PSR.
- Qubits: 4.
- Randomness: 0.25.
- Training seed: 0.
- Stored best generator parameters.
- Four new outputs.
- Random input seed: `None`.

Because the input seed is `None`, new random inputs are sampled and the exact examples are not deterministic across notebook executions.

### Paired randomness comparison

The paired comparison fixes:

- Preset: base.
- Implementation: `qml_torch`.
- Execution: noiseless.
- Evaluation metric: KL divergence.

It constructs scientifically matched pairs between baseline randomness 0 and treatment randomness 1, retaining the same gradient method, qubit count, seed, batch settings, training configuration, simulator configuration, and remaining scientific fields.

Treatment-minus-baseline differences are calculated for:

- Best evaluation score.
- Epoch of best evaluation.
- Evaluation-step volatility.

### Scaling by preset

The filters are:

- Implementation: `qml_torch`.
- Execution: noiseless.
- Gradient: PSR.
- Randomness: 0.
- Seeds: 0, 1, 2.

The facets are base, angle, and amplitude. Within each facet, qubit count is compared:

- Base: q4, q8, q16.
- Angle: q4, q8, q16.
- Amplitude: q4 and q8.

### Scaling by execution type

The filters are:

- Preset: angle.
- Implementation: `qml_torch`.
- Gradient: SPSA.
- Randomness: 0.
- Seeds: 0, 1, 2 where completed.

Noiseless and noisy execution are faceted separately, and available qubit counts are compared. No noisy q16 conclusion is produced.

### Scaling by gradient method

The filters are:

- Preset: base.
- Implementation: `qml_torch`.
- Execution: noiseless.
- Randomness: 0.
- Seeds: 0, 1, 2.

PSR, REG, and SPSA are faceted separately. Within each facet, q4, q8, and available q16 results are compared.

### Scaling by randomness

The filters are:

- Preset: angle.
- Implementation: `qml_torch`.
- Execution: noiseless.
- Gradient: PSR.
- Seeds: 0, 1, 2.

The complete-sweep figure compares q4 and q8 at randomness 0, 0.1, 0.25, 0.5, and 1. A second comparison focuses on randomness 0 and 1 across all available qubit counts.

### Scaling-curve transformation

The notebook currently sets:

```text
SCALING_SCORE_TRANSFORM = "standardize"
```

Every individual convergence series displayed in the scaling-dynamics figures is transformed independently as:

```math
E'_t = \frac{E_t-\operatorname{mean}(E)}{\operatorname{std}(E)}.
```

This affects only the displayed scaling convergence curves. Best evaluation scores, epochs of best evaluation, saved checkpoint values, and other raw data remain untransformed. A constant finite series is transformed to zero.

### Implementation and packing validation

The validation comparison fixes:

- Execution: `fake_real`.
- Qubits: 4.
- Gradient: SPSA.
- Randomness: 0.
- Seed: 0.
- Training length: 1000 epochs.

Eight runs are available:

- Base: `qml_torch`, `runtime_packed/separate`, and `runtime_packed/joined`.
- Angle: `qml_torch`, `runtime_packed/separate`, and `runtime_packed/joined`.
- Amplitude: `qml_torch` and `runtime_packed/separate`.

There is no amplitude `runtime_packed/joined` run because joined packing does not support amplitude encoding. The population contains one run per implementation/preset combination.

## Evaluation metrics

### KL divergence

KL divergence is used only by the base preset:

```math
D_{\mathrm{KL}}(P\|Q) = \sum_i P_i \log\left(\frac{P_i}{Q_i}\right),
```

where `P` is the target probability distribution and `Q` is the generated probability distribution.

- Generated probabilities are clamped to at least `10^-10` before the logarithm.
- PyTorch batch-mean reduction is used.
- Lower values are better.
- Zero represents equal target and generated distributions.

### Image-gradient metric

The angle and amplitude presets use a monotonic-gradient penalty. For a generated image `I`:

```math
\Delta_h I_{i,j} = I_{i,j+1}-I_{i,j},
\qquad
\Delta_v I_{i,j} = I_{i+1,j}-I_{i,j}.
```

The score is:

```math
M_{\mathrm{gradient}} = \frac{1}{2}\left[
-\operatorname{mean}(\min(\Delta_h I,0))
-\operatorname{mean}(\min(\Delta_v I,0))
\right].
```

- Only negative differences are penalized.
- Lower is better.
- Zero means there are no decreases from left to right or top to bottom.
- It measures the monotonic structure expected in the dataset.
- It does not measure direct similarity to a particular target image.
- A zero score does not prove exact image reconstruction.
- Displayed negative-zero values are floating-point representations of zero.

KL-divergence and image-gradient values belong to different metric families and have different numerical meanings.

## Derived quantities

Best evaluation:

```math
E_{\mathrm{best}} = \min_t E_t.
```

Epoch of best evaluation:

```math
t_{\mathrm{best}} = \arg\min_t E_t.
```

Evaluation-step volatility:

```math
V = \operatorname{median}_t |E_t-E_{t-1}|.
```

Other derived quantities are:

- Final evaluation: last finite recorded value.
- Median time per epoch: median recorded wall-clock epoch duration.
- Timing IQR: 25th and 75th percentiles of epoch durations.
- Projected 1000-epoch time: 1000 times median epoch duration.
- Time to best evaluation: sum of epoch times through the best epoch.

## Statistical unit and aggregation

For quality analysis:

- One completed training run is one independent observation.
- Seeds 0, 1, and 2 are independent training replicates.
- Epochs within a run are not independent observations.
- Individual seed trajectories are retained.
- Pointwise trajectory summaries use the median.
- Pointwise spread uses the interquartile range.
- Best-score summaries use individual run values.
- Best-score points are identified by seed.
- CPU and GPU duplicates are removed.

For timing analysis:

- One five-epoch timing run is the configuration-level observation.
- Its five epoch durations are repeated measurements.
- Its representative value is the median epoch duration.
- CPU/1, CPU/4, GPU, and Real-QPU executions remain separate.

For real hardware:

- There is one run per gradient method.
- The hardware data are case studies rather than replicated estimates.

For fake-real validation:

- There is one run per implementation/preset combination.
- The data demonstrate executable training paths rather than statistical equivalence or superiority.

## Feasibility boundaries

The analyzed population is constrained by:

- Exponential statevector memory requirements.
- `4^n` density-matrix memory scaling.
- Parameter-shift circuit-evaluation cost.
- State-preparation circuit depth for amplitude encoding.
- IBM Runtime session duration.
- Available QPU execution allocation.
- CPU memory limits.
- GPU allocation and runtime limits.

Specific unavailable or incomplete regions are:

- q4 noisy PSR convergence: too time-expensive for comprehensive completion.
- q8 noisy PSR convergence: too time-expensive for comprehensive completion.
- q16 amplitude encoding: state-preparation transpilation failure.
- q16 noisy CPU: density-matrix out-of-memory failure.
- q16 noisy GPU: projected execution exceeded the available allocation.
- Physical hardware: only two usable `qml_torch` base/q4 runs rather than the intended timing and implementation matrix.

## Software environment

The project declares:

- Python >=3.10.
- NumPy 2.2.6.
- PyYAML 6.0.3.
- Qiskit 2.4.2.
- Qiskit Aer 0.17.2.
- Qiskit Algorithms 0.4.0.
- Qiskit IBM Runtime 0.47.0.
- Qiskit Machine Learning 0.9.0.
- PyTorch 2.12.1.
- Matplotlib 3.10.9 for analysis figures.
- Pandas for analysis tables.

The repository revision associated with the experiment-data commit is:

```text
869d9f632c9b99d82ac33442c449d2d715a92c6c
```

The exact CPU model, operating system, CUDA version, GPU driver, physical-qubit layouts, transpiled depths, and backend calibration acquisition timestamps are not currently established by `results_analysis.ipynb`.
