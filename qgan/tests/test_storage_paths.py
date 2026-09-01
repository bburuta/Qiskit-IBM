from pathlib import Path

from qgan_v2.storage.paths import (
    get_real_backend_filename,
    get_qgan_root,
    get_repository_root,
    get_run_path,
    resolve_data_path,
)


def test_project_roots():
    qgan_root = get_qgan_root()

    assert qgan_root.name == "qgan"
    assert (qgan_root / "src" / "qgan_v2").is_dir()
    assert get_repository_root() == qgan_root.parent


def test_resolve_data_path_accepts_qgan_relative_paths():
    qgan_root = get_qgan_root()

    assert resolve_data_path("data/test") == qgan_root / "data" / "test"
    assert resolve_data_path("qgan/data/test") == qgan_root / "data" / "test"


def test_resolve_data_path_accepts_absolute_paths(tmp_path):
    data_path = tmp_path / "runs"

    assert resolve_data_path(data_path) == data_path


def test_get_run_path():
    config = {
        "run": {
            "data_path": "data/test",
            "id": "example-run",
        }
    }

    assert get_run_path(config) == get_qgan_root() / "data" / "test" / "example-run"


def test_get_real_backend_filename_can_use_shared_storage():
    config = {
        "backend": {
            "real": {
                "id": "ibm_basquecountry",
                "info_storage": "shared",
            },
        },
    }

    assert get_real_backend_filename(config) == get_qgan_root() / "backends" / "ibm_basquecountry.pkl"


def test_get_real_backend_filename_can_use_run_storage():
    config = {
        "run": {
            "data_path": "data/test",
            "id": "example-run",
        },
        "backend": {
            "real": {
                "id": "ibm_basquecountry",
                "info_storage": "run",
            },
        },
    }

    assert get_real_backend_filename(config) == get_qgan_root() / "data" / "test" / "example-run" / "real_backend.pkl"
