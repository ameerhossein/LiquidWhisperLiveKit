from pathlib import Path

import pytest

import build_ct2


def test_conversion_rejects_wrong_source_checksum(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(SystemExit, match="Checksum mismatch"):
        build_ct2.check_weights(tmp_path, 0, "0" * 64)


@pytest.mark.parametrize("failure", ["converter", "validation", "publish"])
def test_conversion_failure_preserves_previous_model(tmp_path, monkeypatch, failure):
    target = tmp_path / "published"
    target.mkdir()
    (target / "model.bin").write_bytes(b"previous working model")

    def convert(build, quantization, staged):
        staged.mkdir()
        (staged / "model.bin").write_bytes(b"replacement")
        if failure == "converter":
            raise RuntimeError("converter failed")

    def validate(*args, **kwargs):
        if failure == "validation":
            raise RuntimeError("validation failed")

    original_rename = Path.rename

    def rename(path, destination):
        if failure == "publish" and path.name == "model" and destination == target:
            raise RuntimeError("publish failed")
        return original_rename(path, destination)

    monkeypatch.setattr(build_ct2, "_convert_files", convert)
    monkeypatch.setattr(build_ct2, "validate", validate)
    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(RuntimeError):
        build_ct2.convert(tmp_path, "int8_float32", target)
    assert (target / "model.bin").read_bytes() == b"previous working model"


def test_conversion_publishes_only_after_validation(tmp_path, monkeypatch):
    target = tmp_path / "published"
    target.mkdir()
    (target / "model.bin").write_bytes(b"old")

    def convert(build, quantization, staged):
        staged.mkdir()
        (staged / "model.bin").write_bytes(b"new")

    def validate(staged, *args, **kwargs):
        assert (target / "model.bin").read_bytes() == b"old"
        assert (staged / "model.bin").read_bytes() == b"new"

    monkeypatch.setattr(build_ct2, "_convert_files", convert)
    monkeypatch.setattr(build_ct2, "validate", validate)
    build_ct2.convert(tmp_path, "int8_float32", target)
    assert (target / "model.bin").read_bytes() == b"new"


def test_metadata_validation_never_constructs_runtime_model(tmp_path, monkeypatch):
    import json
    from settings import REQUIRED_MODEL_FILES
    from prepare_tokenizer import CHECKS
    for name in REQUIRED_MODEL_FILES:
        (tmp_path / name).write_text("test")
    (tmp_path / "config.json").write_text(json.dumps({"lang_ids": [51908],
        "alignment_heads": build_ct2.LARGE_V3_ALIGNMENT_HEADS}))
    vocab = [f"token{i}" for i in range(53474)]
    vocab[220] = ""
    (tmp_path / "vocabulary.json").write_text(json.dumps(vocab))
    from types import SimpleNamespace
    tokenizer = SimpleNamespace(get_vocab_size=lambda: 53474, token_to_id=CHECKS.get)
    monkeypatch.setattr("tokenizers.Tokenizer.from_file", lambda *a: tokenizer)
    monkeypatch.setattr("ctranslate2.models.Whisper", lambda *a, **k: pytest.fail("Unexpected model load"))
    build_ct2.validate(tmp_path, "float16")
