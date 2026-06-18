from dataclasses import dataclass
from typing import Any


@dataclass
class PackedJob:
    """Transpiled packed circuit and metadata needed to bind batched values."""

    circuit: Any
    observables: list[Any]
    parameters: list[Any]
    logical_to_isa_indices: list[int]
    layout_groups: list[list[int]]
    categories: list[tuple[str, int]]


def build_packed_parameter_row(packed_job, per_copy_values):
    """Reorder logical packed values into the transpiled ISA parameter order."""
    import numpy as np

    per_copy_values = np.asarray(per_copy_values, dtype=float)
    logical_row = per_copy_values.reshape(1, -1)
    return logical_row[:, packed_job.logical_to_isa_indices]


def average_gradients(copy_gradients, width):
    """Average per-copy gradients from a packed Runtime job."""
    import numpy as np

    if not copy_gradients:
        return np.zeros(width, dtype=float)
    return np.mean(np.asarray(copy_gradients, dtype=float), axis=0)

