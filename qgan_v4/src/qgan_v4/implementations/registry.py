from qgan_v4.implementations.manual_estimator import ManualEstimatorImplementation
from qgan_v4.implementations.qml_torch import QMLTorchImplementation
from qgan_v4.implementations.runtime_packed import RuntimePackedImplementation


IMPLEMENTATIONS = {
    QMLTorchImplementation.name: QMLTorchImplementation,
    RuntimePackedImplementation.name: RuntimePackedImplementation,
    ManualEstimatorImplementation.name: ManualEstimatorImplementation,
}


def get_implementation(config):
    name = config["implementation"]["name"]
    try:
        return IMPLEMENTATIONS[name]()
    except KeyError as exc:
        known = ", ".join(sorted(IMPLEMENTATIONS))
        raise ValueError(f"Unknown implementation '{name}'. Known implementations: {known}") from exc
