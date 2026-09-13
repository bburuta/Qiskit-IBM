import numpy as np
import torch

from qgan_v2.training.data import initialize_model_params


def test_initial_parameter_snapshot_does_not_follow_model_updates():
    np.random.seed(7)
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    initial = initialize_model_params(model, init_scale=0.5)
    snapshot = initial.copy()

    with torch.no_grad():
        model.weight.add_(1.0)

    np.testing.assert_array_equal(initial, snapshot)
