from dataclasses import dataclass
from typing import Any


@dataclass
class ManualEstimatorBatch:
    real_inputs: Any = None
    random_inputs: Any = None


class ManualEstimatorGANModel:
    """Interface for GAN models using Qiskit primitives directly."""

    def discriminator_values(self, params, batch):
        raise NotImplementedError

    def discriminator_gradient(self, params, batch):
        raise NotImplementedError

    def generator_values(self, params, batch):
        raise NotImplementedError

    def generator_gradient(self, params, batch):
        raise NotImplementedError

