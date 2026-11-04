import json
from pathlib import Path
from types import SimpleNamespace

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

    def convert(build, quantization, staged, **kwargs):
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

    def convert(build, quantization, staged, **kwargs):
        staged.mkdir()
        (staged / "model.bin").write_bytes(b"new")

    def validate(staged, *args, **kwargs):
        assert (target / "model.bin").read_bytes() == b"old"
        assert (staged / "model.bin").read_bytes() == b"new"

    monkeypatch.setattr(build_ct2, "_convert_files", convert)
    monkeypatch.setattr(build_ct2, "validate", validate)
    build_ct2.convert(tmp_path, "int8_float32", target)
    assert (target / "model.bin").read_bytes() == b"new"


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    from settings import REQUIRED_MODEL_FILES
    from prepare_tokenizer import CHECKS
    for name in REQUIRED_MODEL_FILES:
        (tmp_path / name).write_text("test")
    (tmp_path / "config.json").write_text(json.dumps({"lang_ids": [51908],
        "alignment_heads": build_ct2.LARGE_V3_ALIGNMENT_HEADS}))
    (tmp_path / "source_config.json").write_text(json.dumps({"decoder_layers": 32,
        "decoder_attention_heads": 20}))
    vocab = [f"token{i}" for i in range(53474)]
    for token, index in CHECKS.items():
        vocab[index] = token
    vocab[220] = "<|UNUSED220|>"
    (tmp_path / "vocabulary.json").write_text(json.dumps([
        "" if index == 220 else token for index, token in enumerate(vocab)]))
    tokenizer = SimpleNamespace(get_vocab_size=lambda: 53474, token_to_id=CHECKS.get,
                                id_to_token=vocab.__getitem__)
    monkeypatch.setattr("tokenizers.Tokenizer.from_file", lambda *a: tokenizer)
    return tmp_path


def test_metadata_validation_never_constructs_runtime_model(metadata, monkeypatch):
    monkeypatch.setattr("ctranslate2.models.Whisper", lambda *a, **k: pytest.fail("Unexpected model load"))
    build_ct2.validate(metadata, "float16")


@pytest.mark.parametrize("first,second", [(51908, 51909), (1609, 1610), (51973, 51974)])
def test_validation_rejects_vocabulary_tokenizer_disagreement(metadata, first, second):
    path = metadata / "vocabulary.json"
    vocab = json.loads(path.read_text())
    vocab[first], vocab[second] = vocab[second], vocab[first]
    path.write_text(json.dumps(vocab))
    with pytest.raises(ValueError, match="vocabulary/tokenizer mismatch"):
        build_ct2.validate(metadata, "float16")


@pytest.mark.parametrize("heads", [[], [[32, 0]], [[0, 20]], [[-1, 0]], [[True, 0]], [[1]]])
def test_validation_rejects_invalid_alignment_indices(metadata, heads):
    path = metadata / "config.json"
    cfg = json.loads(path.read_text())
    cfg["alignment_heads"] = heads
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="alignment"):
        build_ct2.validate(metadata, "float16")


@pytest.mark.parametrize("heads,fallback,expected", [
    ([[31, 19]], "converter", [[31, 19]]),
    ([[31, 19]], "large-v3", [[31, 19]]),
    (None, "converter", [[0, 0]]),
    (None, "large-v3", build_ct2.LARGE_V3_ALIGNMENT_HEADS),
])
def test_conversion_preserves_source_heads_and_requires_explicit_fallback(tmp_path, monkeypatch, heads, fallback, expected):
    build, dst = tmp_path / "input", tmp_path / "output"
    build.mkdir()
    (build / "config.json").write_text(json.dumps({"decoder_layers": 32, "decoder_attention_heads": 20}))
    (build / "generation_config.json").write_text(json.dumps({"alignment_heads": heads}))

    def converter(*args, **kwargs):
        dst.mkdir()
        (dst / "config.json").write_text(json.dumps({"alignment_heads": [[0, 0]]}))
        vocab = [f"token{i}" for i in range(53474)]
        vocab[220] = "<|UNUSED220|>"
        (dst / "vocabulary.json").write_text(json.dumps(vocab))

    monkeypatch.setattr(build_ct2.subprocess, "run", converter)
    build_ct2._convert_files(build, "float16", dst, alignment_fallback=fallback)
    assert json.loads((dst / "config.json").read_text())["alignment_heads"] == expected
    assert json.loads((dst / "source_config.json").read_text())["decoder_layers"] == 32


def test_large_v3_heads_rejected_for_smaller_architecture_before_conversion(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"decoder_layers": 2, "decoder_attention_heads": 2}))
    (tmp_path / "generation_config.json").write_text("{}")
    monkeypatch.setattr(build_ct2.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected conversion"))
    with pytest.raises(ValueError, match="Invalid alignment head"):
        build_ct2._convert_files(tmp_path, "float16", tmp_path / "output", alignment_fallback="large-v3")


def _sharded_source(root):
    import numpy as np
    from safetensors.numpy import save_file

    root.mkdir()
    mapping = {}
    for i in range(2):
        name = f"model-0000{i + 1}-of-00002.safetensors"
        key = f"weight{i}"
        save_file({key: np.array([i], dtype=np.float32)}, root / name)
        mapping[key] = name
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return mapping


def _source_metadata(root):
    (root / "config.json").write_text(json.dumps({"decoder_layers": 32, "decoder_attention_heads": 20}))
    for name in ["generation_config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "normalizer.json"]:
        (root / name).write_text("{}")


def test_sharded_weights_are_checked_and_assembled_without_copying_payloads(tmp_path):
    src = tmp_path / "source"
    mapping = _sharded_source(src)
    _source_metadata(src)
    weights = build_ct2.check_weights(src, 0)
    build = tmp_path / "build"
    build_ct2.assemble_hf_dir(src, src, build, weights)
    assert weights.index.name == "model.safetensors.index.json"
    assert json.loads((build / weights.index.name).read_text())["weight_map"] == mapping
    for name in mapping.values():
        assert (build / name).is_symlink()
        assert (build / name).resolve() == (src / name).resolve()
    # Existing build inputs cannot be retargeted by another invocation.
    with pytest.raises(FileExistsError):
        build_ct2.assemble_hf_dir(src, src, build, weights)


@pytest.mark.parametrize("failure", ["missing", "wrong_tensor", "traversal", "absolute", "bad_checksum", "missing_checksum"])
def test_sharded_weights_reject_incomplete_or_inconsistent_inputs(tmp_path, failure):
    src = tmp_path / "source"
    mapping = _sharded_source(src)
    expected = None
    if failure == "missing":
        (src / mapping["weight0"]).unlink()
    elif failure == "wrong_tensor":
        mapping["not_in_the_shard"] = mapping.pop("weight0")
    elif failure == "traversal":
        mapping["weight0"] = "../outside.safetensors"
    elif failure == "absolute":
        mapping["weight0"] = "/tmp/outside.safetensors"
    elif failure == "bad_checksum":
        expected = {name: "0" * 64 for name in mapping.values()}
    else:
        expected = {mapping["weight0"]: "0" * 64}
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises((ValueError, SystemExit)):
        build_ct2.check_weights(src, 0, expected_shards=expected)


def test_sharded_weights_verify_per_file_checksums_and_hub_blob_symlinks(tmp_path):
    from fetch import sha256

    src = tmp_path / "snapshot"
    mapping = _sharded_source(src)
    blob = tmp_path / "cached-blob"
    shard = src / mapping["weight0"]
    shard.rename(blob)
    shard.symlink_to(blob)
    checksums = {name: sha256(src / name) for name in mapping.values()}
    weights = build_ct2.check_weights(src, 0, expected_shards=checksums)
    assert len(weights.files) == 2
    with pytest.raises(ValueError, match="single file"):
        build_ct2.check_weights(src, 0, expected_sha256="0" * 64)


def test_concurrent_invocations_keep_their_own_sources(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    barrier = threading.Barrier(2)
    scratch = tmp_path / "shared-scratch"
    scratch.mkdir()
    sentinel = scratch / "tokenizer.json"
    sentinel.write_text("stale data must not be used or deleted")
    sources = []
    for i in range(2):
        source = tmp_path / f"source{i}"
        _sharded_source(source)
        _source_metadata(source)
        sources.append(source)
    seen = []

    def convert(build, quantization, dst, **kwargs):
        first = build / "model-00001-of-00002.safetensors"
        expected_source = sources[int(dst.parent.name[-1])]
        barrier.wait(timeout=10)
        assert first.resolve() == (expected_source / first.name).resolve()
        seen.append(build)

    monkeypatch.setattr(build_ct2, "convert", convert)
    def run(i):
        build_ct2.main(["--src", str(sources[i]), "--tok", str(sources[i]),
                        "--build-dir", str(scratch), "--out-prefix", str(tmp_path / f"output{i}"),
                        "--quantization", "float16", "--min-weight-bytes", "0"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, range(2)))
    assert len(seen) == 2 and seen[0] != seen[1]
    assert not any(path.exists() for path in seen)
    assert sentinel.read_text() == "stale data must not be used or deleted"


def test_real_sharded_whisper_conversion_and_cpu_generation(tmp_path):
    """Exercise both precisions on tiny random weights; this is not a WER test."""
    import numpy as np
    import torch
    import ctranslate2
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration
    from faster_whisper import WhisperModel
    from faster_whisper.tokenizer import Tokenizer as FasterTokenizer
    from prepare_tokenizer import CHECKS

    source = tmp_path / "source"
    source.mkdir()
    tokens = [f"<|UNUSED{i}|>" for i in range(53474)]
    for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()), start=1609):
        tokens[index] = token
    for token, index in CHECKS.items():
        tokens[index] = token
    for i in range(1501):
        tokens[51973 + i] = f"<|{i * .02:.2f}|>"
    tokenizer = Tokenizer(BPE(vocab={token: i for i, token in enumerate(tokens)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([token for token, index in CHECKS.items() if index < 51973])
    tokenizer.save(str(source / "tokenizer.json"))
    (source / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "WhisperTokenizer", "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>", "unk_token": "<|endoftext|>",
        "language": "fa", "task": "transcribe",
    }))
    (source / "normalizer.json").write_text("{}")
    WhisperFeatureExtractor(feature_size=128).save_pretrained(source)
    config = WhisperConfig(
        vocab_size=53474, num_mel_bins=128, d_model=8,
        encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=16, decoder_ffn_dim=16, max_source_positions=16, max_target_positions=16,
        bos_token_id=51865, eos_token_id=51865, pad_token_id=51865, decoder_start_token_id=51866,
        suppress_tokens=[], begin_suppress_tokens=[],
    )
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(config)
    generation = model.generation_config
    generation._from_model_config = False
    generation.lang_to_id = {"<|fa|>": 51908}
    generation.task_to_id = {"transcribe": 51968, "translate": 51967}
    generation.no_timestamps_token_id = 51972
    generation.language = "fa"
    generation.task = "transcribe"
    generation.alignment_heads = [[0, 1]]
    model.save_pretrained(source, max_shard_size="1MB")
    assert (source / "model.safetensors.index.json").is_file()
    del model
    build_ct2.main(["--src", str(source), "--tok", str(source), "--min-weight-bytes", "0",
                    "--build-dir", str(tmp_path / "scratch"), "--out-prefix", str(tmp_path / "converted"),
                    "--quantization", "both", "--validate-runtime"])
    for quantization, name in build_ct2.TARGETS.items():
        output = tmp_path / "converted" / name
        compute_type = "int8_float32" if quantization.startswith("int8") else "float32"
        runtime = WhisperModel(str(output), device="cpu", compute_type=compute_type, cpu_threads=1)
        tokenizer = FasterTokenizer(runtime.hf_tokenizer, runtime.model.is_multilingual, task="transcribe", language="fa")
        prompt = [*tokenizer.sot_sequence, tokenizer.no_timestamps]
        assert prompt == [51866, 51908, 51968, 51972]
        assert runtime.feature_extractor.mel_filters.shape[0] == 128
        assert json.loads((output / "config.json").read_text())["alignment_heads"] == [[0, 1]]
        features = ctranslate2.StorageView.from_array(np.zeros((1, 128, 32), dtype=np.float32))
        results = runtime.model.generate(features, [prompt], beam_size=1, max_length=8)
        assert len(results) == 1
        assert all(0 <= token < 53474 for token in results[0].sequences_ids[0])
        del runtime
