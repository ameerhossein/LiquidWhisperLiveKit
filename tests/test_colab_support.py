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
