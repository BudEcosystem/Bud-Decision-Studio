"""The models folder chosen on the System page: downloads, model processes and the Models page all follow it, a folder
that can't hold models is refused in words, and a disconnected disk never sends a download to the computer's own disk.

    .venv/bin/python -m pytest tests/test_models_dir.py -q
"""
import io
import os

import pytest

from basal import config, hub
from basal.catalog import CATALOG

SPEC = next(m for m in CATALOG if m.adapter != "fake")


@pytest.fixture(autouse=True)
def config_file():
    """Each test starts on the Hugging Face cache and leaves config.json as it found it."""
    before = config.CONFIG_FILE.read_text() if config.CONFIG_FILE.exists() else None
    config.save({"models_dir": None})
    yield
    if before is None:
        config.CONFIG_FILE.unlink(missing_ok=True)
    else:
        config.CONFIG_FILE.write_text(before)
    config.available.cache_clear()
    config.models_dir.cache_clear()


def test_the_default_is_the_hugging_face_cache():
    d = str(hub.default_cache())
    assert hub.hub_cache() == hub.default_cache()
    assert hub.models_folder() == {"path": d, "default": d, "custom": False, "available": True}


def test_a_chosen_folder_is_where_everything_looks(tmp_path):
    hub.choose_models_dir(str(tmp_path))
    assert config.load()["models_dir"] == str(tmp_path)
    assert hub.hub_cache() == tmp_path
    assert hub.repo_dir("org/name") == tmp_path / "models--org--name"
    assert hub.cache_env() == {"HF_HUB_CACHE": str(tmp_path)}      # what downloads, models and training are given
    assert hub.models_folder() == {"path": str(tmp_path), "default": str(hub.default_cache()), "custom": True,
                                   "available": True}


def test_models_in_the_chosen_folder_count_as_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(hub.META, "data", {})       # no Hub listing: any snapshot with weights counts
    hub.choose_models_dir(str(tmp_path))
    assert not hub.model_status(SPEC)["complete"]
    for r in SPEC.repos():
        snap = hub.repo_dir(r.id) / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"")
    assert hub.model_status(SPEC)["complete"]


def test_a_new_folder_is_made_inside_an_existing_one(tmp_path):
    hub.choose_models_dir(str(tmp_path / "models"))
    assert (tmp_path / "models").is_dir() and hub.hub_cache() == tmp_path / "models"


def test_the_path_is_tidied(tmp_path):
    hub.choose_models_dir(f"  {tmp_path}/a/../models/  ")
    assert hub.hub_cache() == tmp_path / "models"


@pytest.mark.parametrize("bad, words", [
    ("models", "full path"),
    ("{tmp}/unplugged/models", "does not exist"),
    ("{tmp}/notes.txt", "is a file"),
    (42, "full path of a folder"),
])
def test_folders_that_cannot_hold_models_are_refused(tmp_path, bad, words):
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match=words):
        hub.choose_models_dir(bad.format(tmp=tmp_path) if isinstance(bad, str) else bad)
    assert config.models_dir() is None


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs a folder this user cannot write to")
def test_a_folder_the_studio_cannot_write_to_is_refused(tmp_path):
    ro = tmp_path / "read-only"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        with pytest.raises(ValueError, match="Can't save files"):
            hub.choose_models_dir(str(ro))
    finally:
        ro.chmod(0o755)
    assert config.models_dir() is None and list(ro.iterdir()) == []


@pytest.mark.parametrize("back", [None, "", "default"])
def test_null_empty_or_the_default_path_go_back_to_the_default(tmp_path, back):
    hub.choose_models_dir(str(tmp_path))
    hub.choose_models_dir(str(hub.default_cache()) if back == "default" else back)
    assert config.models_dir() is None and "models_dir" not in config.load()


def test_saving_none_removes_only_that_key():
    config.save({"device": "cpu", "models_dir": "/x"})
    config.save({"models_dir": None})
    assert config.load()["device"] == "cpu" and "models_dir" not in config.load()


def test_a_disconnected_folder_is_reported_and_no_download_starts(tmp_path, monkeypatch):
    disk = tmp_path / "disk"
    hub.choose_models_dir(str(disk))
    disk.rmdir()                                    # the external disk is unplugged
    assert hub.models_folder()["available"] is False

    def no_download(*a, **k):
        raise AssertionError("a download started with the models folder missing")
    monkeypatch.setattr(hub.subprocess, "Popen", no_download)
    job = hub.Job(SPEC.id)
    hub.Downloader._run_logged(object.__new__(hub.Downloader), job, io.StringIO())
    assert "not available" in job.error and str(disk) in job.error
    assert not disk.exists()                        # nothing was written where the disk would be
