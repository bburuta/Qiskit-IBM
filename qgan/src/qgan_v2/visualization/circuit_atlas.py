"""Build and display the circuit diagrams used by the qGAN implementations."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from qiskit.quantum_info import Statevector

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qgan_v2.circuits.factory import get_circuits
from qgan_v2.circuits.encoding import (
    create_randomizer_circuit,
    generate_ang_circuit,
    generate_amp_circuits,
    images_to_amp,
)
from qgan_v2.datasets.images import get_images_dataset, image_to_angles
from qgan_v2.datasets.quantum import create_quantum_dataset_circuits
from qgan_v2.config.defaults import apply_experiment_preset, create_config_ids
from qgan_v2.config.loader import load_run_config
from qgan_v2.config.validation import validate_config
from qgan_v2.models.packed_circuits import (
    create_angle_disc_circuit,
    create_direct_disc_circuit,
)
from qgan_v2.models.qnn import compose_circuits
from qgan_v2.visualization.utils import plot_basis_probabilities


class CircuitAtlas:
    """Create representative circuits and optionally export their diagrams.

    Circuit construction is local and does not create a backend, training run,
    or saved dataset. Real circuit 0 is displayed for presets with many samples.
    """

    PRESETS = ("base", "ang", "amp")
    IMPLEMENTATIONS = (
        "qml_torch", "runtime-packed-sep", "runtime-packed-join"
    )
    RANDOM_CIRCUIT_TYPES = (0, 1, 2)

    def __init__(
        self,
        qgan_dir,
        *,
        figure_dir=None,
        config_file=None,
        n_qubits=4,
        randomness=1.0,
        random_circuit=1,
        packed_batch_size=4,
    ):
        self.qgan_dir = Path(qgan_dir).expanduser().resolve()
        self.figure_dir = Path(
            figure_dir or self.qgan_dir / "figures" / "circuit_atlas"
        ).expanduser().resolve()
        self.config_file = Path(
            config_file
            or self.qgan_dir / "configs" / "singles"
            / "base-qml_torch-q4-noiseless-SPSA-aerCPU-rand0-seed0_config.yaml"
        ).expanduser().resolve()
        self.example_config = load_run_config(self.config_file)
        self.n_qubits = n_qubits
        self.randomness = randomness
        self.random_circuit = random_circuit
        self.packed_batch_size = packed_batch_size

    def _config(self, preset, implementation):
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        if implementation not in self.IMPLEMENTATIONS:
            raise ValueError(f"Unknown implementation: {implementation}")
        if implementation == "runtime-packed-join" and preset == "amp":
            raise ValueError("Joined runtime packing does not support the amp preset.")

        config = deepcopy(self.example_config)
        runtime_packed = implementation != "qml_torch"
        config["experiment"]["preset"] = preset
        config["experiment"]["n_qubits"] = self.n_qubits
        config["experiment"]["execution_type"] = (
            "noisy" if runtime_packed else "noiseless"
        )
        config["implementation"]["name"] = (
            "runtime_packed" if runtime_packed else "qml_torch"
        )
        config["implementation"]["discriminator_packing"] = (
            "joined" if implementation == "runtime-packed-join" else "separate"
        )
        apply_experiment_preset(config)
        config["encoding"]["randomness"] = self.randomness
        config["encoding"]["random_circuit"] = self.random_circuit
        batch_size = self.packed_batch_size if runtime_packed else 1
        if runtime_packed and preset == "base":
            batch_size = 1
        config["encoding"]["batch_size"] = batch_size
        # Force in-memory dataset generation even if a cache already exists.
        config["dataset"]["reset"] = True
        config["run"]["id"] = None
        config["dataset"]["id"] = None
        create_config_ids(config)
        return validate_config(config)

    def build_data_preparation(self, preset):
        """Prepare sample 0 using the same dataset and encoding as a training run."""
        config = self._config(preset, "qml_torch")
        if preset == "base":
            circuit = create_quantum_dataset_circuits(config)[0]
            image = normalized = None
        else:
            image = np.asarray(get_images_dataset(config, save_file=False)[0], dtype=float)
            if preset == "ang":
                template = generate_ang_circuit(self.n_qubits)
                circuit = template.assign_parameters(image_to_angles(image).ravel())
                probabilities = Statevector(circuit).probabilities()
                basis = np.arange(len(probabilities))
                normalized = np.array([
                    np.dot(probabilities, 1 - 2 * ((basis >> qubit) & 1))
                    for qubit in range(self.n_qubits)
                ]).reshape(image.shape)
            else:
                amplitudes = images_to_amp(
                    torch.as_tensor(image[None]), config["encoding"]["contrast"]
                )
                circuit = generate_amp_circuits(self.n_qubits, amplitudes)[0]
                normalized = Statevector(circuit).probabilities().reshape(image.shape)
        return {
            "config": config,
            "circuit": circuit,
            "image": image,
            "probabilities": Statevector(circuit).probabilities(),
            "normalized": normalized,
        }

    def build_random_circuit_types(self, preset):
        """Return all randomizer circuits accepted by the experiment config."""
        config = self._config(preset, "qml_torch")
        config["encoding"]["randomness"] = 1.0
        circuits = []
        for circuit_type in self.RANDOM_CIRCUIT_TYPES:
            config["encoding"]["random_circuit"] = circuit_type
            circuits.append((circuit_type, create_randomizer_circuit(config)))
        return circuits

    def show_random_circuit_types(self, preset, *, save_figures=False):
        """Display the supported randomizer choices for a preset."""
        from IPython.display import Markdown, display

        descriptions = {
            0: "Identity",
            1: "one RY rotation per qubit",
            2: "EfficientSU2",
        }
        display(Markdown(f"### {preset} · {self.n_qubits} qubits"))
        for circuit_type, circuit in self.build_random_circuit_types(preset):
            description = descriptions[circuit_type]
            if preset == "ang" and circuit_type:
                description = "one RY rotation per qubit (angle preset)"
            display(Markdown(f"**Type {circuit_type}: {description}**"))
            fig = circuit.draw(output="mpl")
            self._display_preparation_figure(
                fig, preset, f"random_type_{circuit_type}", save_figures
            )

    def show_data_preparation(self, preset, *, save_figures=False):
        """Display the input, bound real circuit, probabilities, and measured image."""
        import matplotlib.pyplot as plt
        from IPython.display import Markdown, display

        case = self.build_data_preparation(preset)
        circuit = case["circuit"]
        display(Markdown(f"### {preset} · {self.n_qubits} qubits"))
        if case["image"] is not None:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            input_plot = axes[0].imshow(case["image"], cmap="gray", vmin=0, vmax=1,
                                        interpolation="nearest")
            axes[0].set_title("Sample gradient image (sample 0)", pad=16)
            axes[0].axis("off")
            fig.colorbar(input_plot, ax=axes[0], label="pixel value")

            if preset == "ang":
                measured_plot = axes[1].imshow(
                    case["normalized"], cmap="gray_r", vmin=-1, vmax=1,
                    interpolation="nearest",
                )
                axes[1].set_title("Measured Z expectation: cos(π × pixel)", pad=16)
                label = "⟨Z⟩ (−1 white, 1 black)"
            else:
                measured_plot = axes[1].imshow(
                    case["normalized"], cmap="gray", vmin=0,
                    vmax=float(case["normalized"].max()), interpolation="nearest",
                )
                axes[1].set_title("Measured image: normalized probabilities", pad=16)
                label = "probability (sum = 1)"
            axes[1].axis("off")
            fig.colorbar(measured_plot, ax=axes[1], label=label)
            fig.tight_layout(pad=2)
            self._display_preparation_figure(fig, preset, "images", save_figures)

        display_circuit = circuit.decompose() if preset == "amp" else circuit
        fig = display_circuit.draw(output="mpl")
        fig.suptitle("Target circuit" if preset == "base" else "Real circuit for sample 0")
        fig.subplots_adjust(top=0.82)
        self._display_preparation_figure(fig, preset, "circuit", save_figures)

        fig, ax = plt.subplots(figsize=(9, 3))
        plot_basis_probabilities(ax, case["probabilities"], self.n_qubits,
                                 title="Sample probability distribution", color="C1")
        fig.tight_layout()
        self._display_preparation_figure(fig, preset, "probabilities", save_figures)

        return case

    def _display_preparation_figure(self, fig, preset, slug, save_figures):
        import matplotlib.pyplot as plt
        from IPython.display import display

        try:
            if save_figures:
                self.figure_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(self.figure_dir / f"data_preparation_{preset}_{slug}.png",
                            dpi=300, bbox_inches="tight")
            display(fig)
        finally:
            plt.close(fig)

    @staticmethod
    def _boxed_job(blocks, name):
        """Keep each real or fake branch intact inside a labelled circuit box."""
        n_qubits = blocks[0][1].num_qubits
        overview = QuantumCircuit(n_qubits * len(blocks), name=name)
        for index, (label, branch) in enumerate(blocks):
            input_params = [
                param for param in branch.parameters
                if param.name.startswith("θ_r")
            ]
            if input_params:
                unique_inputs = ParameterVector(f"{name}_input_{index}", len(input_params))
                branch = branch.assign_parameters(
                    dict(zip(input_params, unique_inputs)), inplace=False
                )
            qubits = range(index * n_qubits, (index + 1) * n_qubits)
            overview.append(branch.to_instruction(label=label), list(qubits))
        return overview

    def build_case(self, preset, implementation):
        """Return the config, real sample count, and named circuit diagrams."""
        config = self._config(preset, implementation)
        generator, discriminator, randomizer, real_circuits = get_circuits(
            config, save_file=False, save_dataset=False
        )
        real = real_circuits[0]
        fake_branch = compose_circuits(randomizer, generator, discriminator)
        real_branch = compose_circuits(real, discriminator)
        diagrams = [
            ("generator", "Generator", generator),
            ("discriminator", "Discriminator", discriminator),
            ("real", "Real input (sample 0)", real),
            ("random", "Random input", randomizer),
            (
                "random_gen", "Random → generator",
                compose_circuits(randomizer, generator),
            ),
            (
                "gen_disc", "Generator → discriminator",
                compose_circuits(generator, discriminator),
            ),
        ]

        if implementation == "qml_torch":
            diagrams.extend([
                ("real_disc", "Real → discriminator", real_branch),
                ("random_gen_disc", "Random → generator → discriminator", fake_branch),
            ])
        else:
            diagrams.extend([
                (
                    "fake_branch", "Generator training / fake discriminator circuit",
                    fake_branch,
                ),
                ("real_branch", "Real discriminator circuit", real_branch),
            ])
            batch_size = config["encoding"]["batch_size"]
            generator_blocks = [
                (f"Fake {index + 1}: R-G-D", fake_branch)
                for index in range(batch_size)
            ]

            if implementation == "runtime-packed-sep":
                diagrams.append((
                    "generator_job", "Generator training / fake discriminator packed job",
                    self._boxed_job(generator_blocks, "generator_job"),
                ))
                real_blocks = [
                    (
                        f"Real {index + 1}: X-D",
                        compose_circuits(
                            real_circuits[index % len(real_circuits)], discriminator
                        ),
                    )
                    for index in range(batch_size)
                ]
                diagrams.append((
                    "real_discriminator_job", "Real discriminator packed job",
                    self._boxed_job(real_blocks, "real_discriminator_job"),
                ))
            else:
                if preset == "base":
                    joined_pair, _ = create_direct_disc_circuit(
                        randomizer, generator, discriminator, real
                    )
                else:
                    joined_pair, _ = create_angle_disc_circuit(
                        randomizer, generator, discriminator, real, 2
                    )
                diagrams.append((
                    "joined_discriminator_pair",
                    "Joined discriminator circuit: one real and one fake branch",
                    joined_pair,
                ))
                diagrams.append((
                    "generator_job", "Generator training packed job",
                    self._boxed_job(generator_blocks, "generator_job"),
                ))
                half_batch = 1 if preset == "base" else batch_size // 2
                discriminator_blocks = [
                    (f"Real {index + 1}: X-D", real_branch)
                    for index in range(half_batch)
                ] + [
                    (f"Fake {index + 1}: R-G-D", fake_branch)
                    for index in range(half_batch)
                ]
                diagrams.append((
                    "joined_discriminator_job", "Joined discriminator packed job",
                    self._boxed_job(discriminator_blocks, "joined_discriminator_job"),
                ))

        return {
            "config": config,
            "real_circuit_count": len(real_circuits),
            "diagrams": diagrams,
        }

    def show_case(self, preset, implementation, *, save_figures=False):
        """Display each circuit as an image; optionally save the same image as PNG."""
        import matplotlib.pyplot as plt
        from IPython.display import Markdown, display

        case = self.build_case(preset, implementation)
        config = case["config"]
        display(Markdown(
            f"**{implementation} / {preset}** · "
            f"encoding: `{config['encoding']['type']}` · "
            f"real circuits: {case['real_circuit_count']} · "
            f"randomness: {self.randomness}"
        ))
        if case["real_circuit_count"] > 1:
            display(Markdown("Showing real sample 0; other samples use the same encoding path."))

        saved_paths = []
        if save_figures:
            self.figure_dir.mkdir(parents=True, exist_ok=True)
        for slug, label, circuit in case["diagrams"]:
            display(Markdown(f"#### {label} — {circuit.num_qubits} qubits"))
            display_circuit = (
                circuit.decompose() if preset == "amp" and slug == "real" else circuit
            )
            fig = display_circuit.draw(output="mpl")
            try:
                fig.suptitle(f"{implementation} / {preset}: {label}", fontsize=12)
                fig.subplots_adjust(top=0.82)
                if save_figures:
                    path = self.figure_dir / f"{implementation}_{preset}_{slug}.png"
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    saved_paths.append(path)
                display(fig)
            finally:
                plt.close(fig)

        if saved_paths:
            print(f"Saved {len(saved_paths)} PNG files in {self.figure_dir}")
        return saved_paths
