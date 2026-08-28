from getpass import getpass

from qiskit_ibm_runtime import QiskitRuntimeService


# Save IBM Runtime credentials without exposing the token in shell history
def save_runtime_account():
    channel = input("Channel [ibm_quantum_platform]: ").strip() or "ibm_quantum_platform"
    token = getpass("API token: ").strip()
    instance = input("Instance CRN/name [optional]: ").strip() or "auto"

    QiskitRuntimeService.save_account(
        channel=channel,
        token=token,
        instance=instance,
        set_as_default=True,
        overwrite=True,
    )

    print("IBM Runtime account saved.")
