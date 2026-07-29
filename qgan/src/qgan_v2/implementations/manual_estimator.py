from qgan_v2.implementations.base import QGANImplementation


class ManualEstimatorImplementation(QGANImplementation):
    """Target home for implementations without qiskit-machine-learning."""

    name = "manual_estimator"

    def setup(self, config):
        raise NotImplementedError(
            "manual_estimator is scaffolded in qgan_v2. "
            "Implement qgan_v2.models.manual_estimator.ManualEstimatorGANModel first."
        )

    def train_discriminator_step(self, state):
        raise NotImplementedError

    def train_generator_step(self, state):
        raise NotImplementedError

    def evaluate(self, state):
        raise NotImplementedError
