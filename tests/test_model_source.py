"""Local and HF CT2 artifacts share validation without any model conversion."""
from pathlib import Path
from unittest.mock import Mock

import pytest

import settings


@pytest.fixture(autouse=True)
def clean_source(monkeypatch):
    for key in ("MODEL_DIR", "HF_REPO_ID", "HF_SUBFOLDER", "HF_REVISION", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(key, raising=False)


def artifact(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in settings.REQUIRED_MODEL_FILES:
        (path / name).write_bytes(b"model asset")
    return path


def test_local_source_does_not_download(tmp_path, monkeypatch):
    local = artifact(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(local))
    download = Mock(side_effect=AssertionError("Unexpected HF download"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert settings.require_model(settings.model_path()) == local


@pytest.mark.parametrize("subfolder", ["", "runs/run1/ct2"])
def test_hf_revision_subfolder_and_only_runtime_assets(tmp_path, monkeypatch, subfolder):
    expected = artifact(tmp_path / subfolder)
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setenv("HF_REPO_ID", "private/model")
    monkeypatch.setenv("HF_REVISION", "pinned-sha")
    monkeypatch.setenv("HF_SUBFOLDER", subfolder)
    assert Path(settings.model_path()) == expected
    kwargs = download.call_args.kwargs
    assert kwargs["repo_id"] == "private/model"
    assert kwargs["revision"] == "pinned-sha"
    prefix = subfolder + "/" if subfolder else ""
    assert kwargs["allow_patterns"] == [prefix + name for name in settings.REQUIRED_MODEL_FILES]
    # Omitting token lets HF honor HF_TOKEN and saved private-repo credentials.
    assert "token" not in kwargs


def test_cached_model_can_run_offline(tmp_path, monkeypatch):
    artifact(tmp_path)
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    settings.download_model("private/model")
    assert download.call_args.kwargs["local_files_only"] is True


def test_hf_weights_are_rejected_without_conversion(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"HF weights")
    monkeypatch.setattr("huggingface_hub.snapshot_download", Mock(return_value=str(tmp_path)))
    with pytest.raises(ValueError, match="prepared CTranslate2 model"):
        settings.download_model("owner/merged")


@pytest.mark.parametrize("folder", ["../weights", "/weights", "run/../../weights", "run\\weights"])
def test_subfolder_cannot_escape_snapshot(folder, monkeypatch):
    download = Mock()
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    with pytest.raises(ValueError, match="relative path"):
        settings.download_model("owner/model", subfolder=folder)
    download.assert_not_called()


def test_conflicting_sources_are_rejected(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "local-model")
    monkeypatch.setenv("HF_REPO_ID", "owner/model")
    with pytest.raises(ValueError, match="not both"):
        settings.model_path()


@pytest.mark.parametrize("device,suffix", [("cpu", "int8"), ("gpu", "fp16")])
def test_default_local_source(device, suffix):
    assert settings.model_path(device).endswith("buzzasr-persian-ct2-" + suffix)


@pytest.mark.parametrize("launcher", ["run.sh", "scripts/verify.sh"])
def test_launchers_stop_when_model_resolution_fails(tmp_path, launcher):
    import os
    import subprocess

    env = dict(os.environ, MODEL_DIR=str(tmp_path), DEVICE="cpu")
    result = subprocess.run(
        ["bash", str(settings.ROOT / launcher)], env=env,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "Incomplete model" in result.stderr
    assert "server startup" not in result.stdout
