from abc import ABC, abstractmethod


class QGANImplementation(ABC):
    """Small interface used by the generic training loop."""

    name = "base"

    @abstractmethod
    def setup(self, config):
        """Create or restore implementation-specific training state."""

    @abstractmethod
    def train_discriminator_step(self, state):
        """Run one discriminator update and return its loss."""

    @abstractmethod
    def train_generator_step(self, state):
        """Run one generator update and return its loss."""

    @abstractmethod
    def evaluate(self, state):
        """Evaluate the current generator."""

    def close(self, state):
        """Release sessions or other external resources."""
