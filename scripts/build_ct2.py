#!/usr/bin/env python3
"""Convert BuzzASR/persian (HF Whisper) into CTranslate2 models for WhisperLiveKit.

Produces two directories:
  models/buzzasr-persian-ct2-int8  -> CPU   (int8)
  models/buzzasr-persian-ct2-fp16  -> GPU   (float16)

Post-processing the stock converter does not do:

  * vocabulary.json[220] = "" -- CTranslate2 decides whether a Whisper model is
    multilingual by testing for the empty-string token (`to_id("") != unk_id`).
    BuzzASR's Persian BPE has no empty token, so CT2 otherwise reports the model
    as English-only and faster-whisper drops the <|fa|><|transcribe|> prompt.
    Id 220 is a dead placeholder row in this checkpoint, so recycling it is safe
    (faster-whisper decodes with tokenizer.json, not with this file).

  * config.json lang_ids=[51908] -- the converter reads
    generation_config.lang_to_id, but transformers 5 drops unknown GenerationConfig
    fields, leaving lang_ids empty and detect_language() unusable.

  * Preserve source alignment_heads and validate their layer/head indices.
    Without source heads, keep the converter's architecture-aware fallback.
    --alignment-heads-fallback large-v3 explicitly opts into OpenAI's heads;
    these still require Persian word-boundary validation.
"""
import argparse
from dataclasses import dataclass
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# openai/whisper-large-v3 generation_config.json
LARGE_V3_ALIGNMENT_HEADS = [
    [7, 0], [10, 17], [12, 18], [13, 12], [16, 1],
    [17, 14], [19, 11], [21, 4], [24, 1], [25, 6],
]

TARGETS = {
    # int8_float32 (not int8_float16): CPU wants float32 activations, otherwise
    # CTranslate2 re-converts the weights at every model load.
    "int8_float32": "buzzasr-persian-ct2-int8",
    "float16": "buzzasr-persian-ct2-fp16",
}


@dataclass(frozen=True)
class WeightFiles:
    files: tuple[Path, ...]
    index: Path | None = None


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _shard_path(src, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
        raise ValueError(f"Invalid shard path: {name}")
    # Hub snapshots legitimately symlink shards into a shared blob cache.
    # Reject index path traversal while permitting these explicit file links.
    return src / path


def check_weights(src: Path, min_size: int, expected_sha256=None, expected_shards=None):
    src = Path(src)
    single = src / "model.safetensors"
    index = src / "model.safetensors.index.json"
    weight_map = None
    if single.is_file():
        weights = WeightFiles((single,))
    elif index.is_file():
        weight_map = _read_json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()
        ):
            raise ValueError(f"Invalid weight_map in {index}")
        weights = WeightFiles(tuple(_shard_path(src, name) for name in sorted(set(weight_map.values()))), index)
        if expected_sha256:
            raise ValueError("--weights-sha256 requires a single file; use --weights-checksums for shards")
    else:
        sys.exit(f"missing model.safetensors or model.safetensors.index.json in {src}")
    for path in weights.files:
        if not path.is_file():
            raise ValueError(f"Missing weight shard: {path}")
    size = sum(path.stat().st_size for path in weights.files)
    if size < min_size:
        sys.exit(f"{src} weights look incomplete ({size} bytes total)")
    if expected_sha256 is None and single.is_file() and single.resolve() == (ROOT / "models/buzzasr-persian/model.safetensors").resolve():
        manifest = _read_json(ROOT / "model-manifest.json")
        expected_sha256 = manifest["files"]["model.safetensors"]["sha256"]
    checksums = dict(expected_shards or {})
    if expected_shards is not None and set(checksums) != {str(path.relative_to(src)) for path in weights.files}:
        raise ValueError("--weights-checksums must specify every weight file exactly once")
    if expected_sha256:
        checksums["model.safetensors"] = expected_sha256
    from fetch import sha256
    from safetensors import safe_open

    tensor_count = 0
    for path in weights.files:
        name = str(path.relative_to(src))
        expected = checksums.get(name)
        if expected is not None and sha256(path) != expected.lower():
            sys.exit(f"Checksum mismatch: {path}")
        with safe_open(str(path), framework="pt") as stream:
            tensors = set(stream.keys())
        if weight_map is not None and tensors != {key for key, shard in weight_map.items() if shard == name}:
            raise ValueError(f"Shard tensors do not match weight_map: {name}")
        tensor_count += len(tensors)
    print(f"weights OK: {size / 1e9:.2f} GB, {tensor_count} tensors in {len(weights.files)} file(s)")
    return weights


def assemble_hf_dir(src: Path, tok: Path, build: Path, weights: WeightFiles):
    # Never mutate a previous invocation's inputs. main() provides a unique
    # directory; direct callers must also choose a new directory.
    build.mkdir(parents=True, exist_ok=False)
    for name in ("config.json", "generation_config.json"):
        shutil.copy2(src / name, build / name)
    pre = src / "preprocessor_config.json"
    if pre.is_file():
        shutil.copy2(pre, build / "preprocessor_config.json")
    else:
        # transformers >=5 saves one processor_config.json with the feature
        # extractor nested inside; CTranslate2/faster-whisper still want the
        # standalone preprocessor_config.json.
        processor = json.loads((src / "processor_config.json").read_text(encoding="utf-8"))
        feature_extractor = processor.get("feature_extractor")
        if not isinstance(feature_extractor, dict):
            sys.exit(f"missing preprocessor_config.json in {src} and no "
                     "feature_extractor in processor_config.json")
        (build / "preprocessor_config.json").write_text(
            json.dumps(feature_extractor, indent=2), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "normalizer.json"):
        shutil.copy2(tok / name, build / name)
    if weights.index is not None:
        shutil.copy2(weights.index, build / weights.index.name)
    for path in weights.files:
        link = build / path.relative_to(src)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(path.resolve())


def _validate_alignment_heads(heads, model_config):
    layers = model_config["decoder_layers"]
    count = model_config["decoder_attention_heads"]
    if not isinstance(heads, list) or not heads:
        raise ValueError("alignment_heads must be a nonempty list")
    for pair in heads:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(value) is not int for value in pair)
                or not 0 <= pair[0] < layers or not 0 <= pair[1] < count):
            raise ValueError(f"Invalid alignment head {pair!r} for {layers} layers and {count} heads")
    return heads


def _convert_files(build: Path, quantization: str, dst: Path, alignment_fallback="converter"):
    model_config = _read_json(build / "config.json")
    generation = _read_json(build / "generation_config.json")
    source_heads = generation.get("alignment_heads")
    if alignment_fallback not in ("converter", "large-v3"):
        raise ValueError(f"Unknown alignment fallback: {alignment_fallback}")
    requested_heads = source_heads
    if requested_heads is None and alignment_fallback == "large-v3":
        requested_heads = LARGE_V3_ALIGNMENT_HEADS
    if requested_heads is not None:
        _validate_alignment_heads(requested_heads, model_config)
    print(f"\n=== {quantization}: {build} -> {dst}")
    cmd = [
        str(Path(sys.executable).with_name("ct2-transformers-converter")),
        "--model", str(build),
        "--output_dir", str(dst),
        "--quantization", quantization,
        "--copy_files", "tokenizer.json", "tokenizer_config.json",
        "preprocessor_config.json", "normalizer.json",
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["lang_ids"] = [51908]
    if requested_heads is not None:
        cfg["alignment_heads"] = requested_heads
    _validate_alignment_heads(cfg.get("alignment_heads"), model_config)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    shutil.copy2(build / "config.json", dst / "source_config.json")
    print(f"config.json: lang_ids=[51908] (<|fa|>), alignment_heads={len(cfg['alignment_heads'])} "
          f"({'source' if source_heads is not None else alignment_fallback})")

    vocab_path = dst / "vocabulary.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if len(vocab) != 53474 or vocab[220] not in ("<|UNUSED220|>", ""):
        raise ValueError("Cannot apply multilingual patch: row 220 is not the unused BuzzASR placeholder")
    if "" in vocab[:220] + vocab[221:]:
        raise ValueError("Unexpected empty token outside vocabulary row 220")
    vocab[220] = ""
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print("patched vocabulary[220] -> '' (marks model multilingual)")
    print("size:", round(sum(p.stat().st_size for p in dst.iterdir()) / 1e9, 2), "GB")


def validate(dst: Path, quantization: str, runtime=False):
    from settings import require_model
    from prepare_tokenizer import CHECKS
    import tokenizers

    require_model(dst)
    cfg = json.loads((dst / "config.json").read_text())
    vocab = json.loads((dst / "vocabulary.json").read_text())
    if cfg.get("lang_ids") != [51908]:
        raise ValueError("Missing BuzzASR language patch")
    _validate_alignment_heads(cfg.get("alignment_heads"), _read_json(dst / "source_config.json"))
    if len(vocab) != 53474 or vocab[220] != "":
        raise ValueError("Missing multilingual vocabulary patch")
    tok = tokenizers.Tokenizer.from_file(str(dst / "tokenizer.json"))
    if tok.get_vocab_size() != 53474:
        raise ValueError("Unexpected vocabulary size")
    for token, expected in CHECKS.items():
        if tok.token_to_id(token) != expected:
            raise ValueError(f"Wrong token id for {token}")
    for index, token in enumerate(vocab):
        expected = tok.id_to_token(index)
        if index == 220:
            if expected != "<|UNUSED220|>":
                raise ValueError("Tokenizer row 220 is not the expected unused placeholder")
        elif token != expected:
            raise ValueError(f"CT2 vocabulary/tokenizer mismatch at ID {index}: {token!r} != {expected!r}")
    if runtime and quantization.startswith("int8"):
        from faster_whisper import WhisperModel
        model = WhisperModel(str(dst), device="cpu", compute_type="int8_float32")
        try:
            if not model.model.is_multilingual:
                raise ValueError("Runtime reports English-only model")
        finally:
            del model
    print(f"Validated {dst.name} ({'runtime' if runtime and quantization.startswith('int8') else 'metadata'})")


def convert(build: Path, quantization: str, dst: Path, runtime=False, alignment_fallback="converter"):
    """Build/validate away from the live artifact; restore it on publish failure."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dst.name}-build-", dir=dst.parent) as temporary:
        staged = Path(temporary) / "model"
        _convert_files(build, quantization, staged, alignment_fallback=alignment_fallback)
        validate(staged, quantization, runtime=runtime)
        # Keep the backup outside the temporary build tree: if rollback itself
        # fails (e.g. filesystem error), cleanup must never delete the old model.
        backup = dst.parent / f".{dst.name}-previous-{uuid.uuid4().hex}"
        had_previous = dst.exists() or dst.is_symlink()
        try:
            if had_previous:
                dst.rename(backup)
            staged.rename(dst)
        except BaseException:
            if backup.exists() or backup.is_symlink():
                backup.rename(dst)
            raise
        if backup.is_symlink():
            backup.unlink()
        elif backup.exists():
            shutil.rmtree(backup)



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantization", choices=["int8_float32", "float16", "both"], default="both")
    ap.add_argument("--src", default=str(ROOT / "models/buzzasr-persian"))
    ap.add_argument("--tok", default=str(ROOT / "models/buzzasr-persian-tok"))
    ap.add_argument("--build-dir", default=str(ROOT / "models/buzzasr-persian-hf"),
                    help="Scratch parent; each invocation uses a unique temporary subdirectory")
    ap.add_argument("--out-prefix", default=str(ROOT / "models"))
    ap.add_argument("--min-weight-bytes", type=int, default=3_000_000_000)
    ap.add_argument("--weights-sha256", help="Expected digest for a custom source checkpoint")
    ap.add_argument("--weights-checksums", type=Path,
                    help="JSON mapping every weight filename to its SHA256 (supports sharded checkpoints)")
    ap.add_argument("--alignment-heads-fallback", choices=["converter", "large-v3"], default="converter",
                    help="When source heads are absent: keep converter heads, or explicitly use OpenAI large-v3 heads")
    ap.add_argument("--validate-runtime", action="store_true",
                    help="Also load each int8 model once on CPU; fp16 checks stay metadata-only")
    args = ap.parse_args(argv)

    src, tok = Path(args.src), Path(args.tok)
    checksums = _read_json(args.weights_checksums) if args.weights_checksums else None
    weights = check_weights(src, args.min_weight_bytes, args.weights_sha256, checksums)
    scratch = Path(args.build_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    targets = TARGETS if args.quantization == "both" else {args.quantization: TARGETS[args.quantization]}
    dirs = {q: Path(args.out_prefix) / name for q, name in targets.items()}
    with tempfile.TemporaryDirectory(prefix=".ct2-source-", dir=scratch) as temporary:
        build = Path(temporary) / "hf"
        assemble_hf_dir(src, tok, build, weights)
        for quant, dst in dirs.items():
            convert(build, quant, dst, runtime=args.validate_runtime,
                    alignment_fallback=args.alignment_heads_fallback)

    print("\nOK – ready for WhisperLiveKit")


if __name__ == "__main__":
    main()
