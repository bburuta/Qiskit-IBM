"""Build and display the circuit diagrams used by the qGAN implementations."""

from copy import deepcopy
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qgan_v2.circuits.factory import get_circuits
from qgan_v2.config.defaults import apply_experiment_preset, create_config_ids
from qgan_v2.config.loader import load_run_config
from qgan_v2.config.validation import validate_config
from qgan_v2.models.packed_circuits import (
    create_angle_disc_circuit,
    create_direct_disc_circuit,
)
from qgan_v2.models.qnn import compose_circuits


class CircuitAtlas:
    """Create representative circuits and optionally export their diagrams.

    Circuit construction is local and does not create a backend, training run,
    or saved dataset. Real circuit 0 is displayed for presets with many samples.
    """

    PRESETS = ("base", "ang", "amp")
    IMPLEMENTATIONS = (
        "qml_torch", "runtime-packed-sep", "runtime-packed-join"
    )

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
            fig = circuit.draw(output="mpl")
            try:
                fig.suptitle(f"{implementation} / {preset}: {label}", fontsize=12)
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
