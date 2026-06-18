from qgan_v4.implementations.base import QGANImplementation


class AngleRuntimePackedImplementation(QGANImplementation):
    """Target home for qgan_rh_ang.ipynb packed real-hardware execution."""

    name = "angle_runtime_packed"

    def setup(self, config):
        raise NotImplementedError(
            "angle_runtime_packed is scaffolded in v4. "
            "Move packed circuit construction into qgan_v4.execution.packed_runtime first."
        )

    def train_discriminator_step(self, state):
        raise NotImplementedError

    def train_generator_step(self, state):
        raise NotImplementedError

    def evaluate(self, state):
        raise NotImplementedError

