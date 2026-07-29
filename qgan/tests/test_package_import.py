from pathlib import Path

import qgan_v2


def test_package_imports():
    package_file = Path(qgan_v2.__file__).resolve()

    assert package_file.name == "__init__.py"
    assert package_file.parent.name == "qgan_v2"
