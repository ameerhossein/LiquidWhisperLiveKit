import os
import subprocess
from pathlib import Path

import pytest

import colab_support
import settings

REQUIRED = list(colab_support.REQUIRED_CT2_FILES)


def files_at(prefix: str) -> list[str]:
    return [f"{prefix}{name}" for name in REQUIRED]


def test_required_files_match_runtime_validation():
    assert set(colab_support.REQUIRED_CT2_FILES) == set(settings.REQUIRED_MODEL_FILES)


def test_find_root_ct2():
    assert colab_support.find_ct2_subfolder(files_at("")) == ""


def test_find_nested_ct2():
    assert colab_support.find_ct2_subfolder(files_at("runs/a/ct2/")) == "runs/a/ct2"


def test_preferred_subfolder_is_validated():
    files = files_at("a/") + files_at("b/")
    assert colab_support.find_ct2_subfolder(files, "b") == "b"


def test_incomplete_ct2_is_rejected():
    with pytest.raises(ValueError, match="No complete CTranslate2"):
        colab_support.find_ct2_subfolder(["model.bin", "config.json"])


def test_ambiguous_ct2_is_rejected():
    with pytest.raises(ValueError, match="Multiple complete CTranslate2"):
        colab_support.find_ct2_subfolder(files_at("a/") + files_at("b/"))


def test_requested_subfolder_must_be_complete():
    with pytest.raises(ValueError, match="not a complete CT2 directory"):
        colab_support.find_ct2_subfolder(files_at("a/"), "b")


def test_requested_dot_means_root():
    assert colab_support.find_ct2_subfolder(files_at(""), ".") == ""


def test_requested_dot_rejects_incomplete_root():
    with pytest.raises(ValueError, match="not a complete CT2 directory"):
        colab_support.find_ct2_subfolder(files_at("nested/") + ["model.bin"], ".")


def test_invalid_requested_lists_model_bin_directories():
    with pytest.raises(ValueError) as exc:
        colab_support.find_ct2_subfolder(files_at("a/") + ["b/model.bin"], "b")
    assert "Directories containing model.bin" in str(exc.value)
    assert "'b'" in str(exc.value)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("visit https://one-two.trycloudflare.com now", "https://one-two.trycloudflare.com"),
        ("https://a.trycloudflare.com\nhttps://b.trycloudflare.com", "https://a.trycloudflare.com"),
        ("not ready", None),
    ],
)
def test_parse_tunnel_url(text, expected):
    assert colab_support.parse_tunnel_url(text) == expected


def test_nvidia_library_dirs_only_returns_existing_dirs(tmp_path):
    for relative in ("nvidia/cudnn/lib", "nvidia/cublas/lib", "nvidia/cuda_runtime/lib"):
        (tmp_path / relative).mkdir(parents=True)
    assert colab_support.nvidia_library_dirs(tmp_path) == [
        str(tmp_path / "nvidia/cudnn/lib"),
        str(tmp_path / "nvidia/cublas/lib"),
        str(tmp_path / "nvidia/cuda_runtime/lib"),
    ]


def test_find_cudnn_library_in_venv(tmp_path):
    lib = tmp_path / "nvidia/cudnn/lib/libcudnn_ops.so.9.1.0"
    lib.parent.mkdir(parents=True)
    lib.touch()
    assert colab_support.find_cudnn_library(tmp_path, search_dirs=()) == str(lib)


def test_find_cudnn_library_none(tmp_path):
    assert colab_support.find_cudnn_library(tmp_path, search_dirs=()) is None


@pytest.mark.parametrize("value", ["0", "-1", "not-a-pid", ""])
def test_read_pid_rejects_invalid_values(tmp_path, value):
    pid_file = tmp_path / "pid"
    pid_file.write_text(value, encoding="utf-8")
    assert colab_support.read_pid(pid_file) is None


def test_pid_matches_the_current_process():
    assert colab_support.pid_matches(os.getpid(), "python")


def test_pid_matches_missing_process():
    assert not colab_support.pid_matches(99_999_999, "python")


def test_stop_process_terminates_a_matching_group(tmp_path):
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    pid_file = tmp_path / "pid"
    pid_file.write_text(str(process.pid), encoding="utf-8")
    try:
        assert colab_support.stop_process(pid_file, ("sleep",)) is True
        assert process.wait(timeout=5) is not None
        assert not pid_file.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_stop_process_skips_an_unexpected_command(tmp_path):
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    pid_file = tmp_path / "pid"
    pid_file.write_text(str(process.pid), encoding="utf-8")
    try:
        assert colab_support.stop_process(pid_file, ("cloudflared",)) is False
        assert process.poll() is None
        assert not pid_file.exists()
    finally:
        process.kill()
        process.wait()


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"ready": True}, True),
        ({"ready": False}, False),
        ({}, False),
        (None, False),
        ("ok", False),
    ],
)
def test_health_payload_is_ready(payload, expected):
    assert colab_support.health_payload_is_ready(payload) is expected
