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

  * config.json alignment_heads -- BuzzASR ships no alignment heads, so the
    converter falls back to "last half of layers x every head" (320 heads). Word
    timestamps then collapse onto identical values and DTW gets slow. The
    fine-tune kept whisper-large-v3's attention, so its 10 published heads are
    the right ones (openai/whisper-large-v3 generation_config.json).
"""
import argparse
import json
import shutil
import subprocess
import sys
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


def check_weights(src: Path, min_size: int):
    w = src / "model.safetensors"
    if not w.is_file():
        sys.exit(f"missing weights: {w}")
    size = w.stat().st_size
    if size < min_size:
        sys.exit(f"{w} looks incomplete ({size} bytes)")
    from safetensors import safe_open

    with safe_open(str(w), framework="pt") as f:
        tensors = list(f.keys())
    print(f"weights OK: {size / 1e9:.2f} GB, {len(tensors)} tensors")
    return w


def assemble_hf_dir(src: Path, tok: Path, build: Path, weights: Path):
    build.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "preprocessor_config.json"):
        shutil.copy2(src / name, build / name)
    for name in ("tokenizer.json", "tokenizer_config.json", "normalizer.json"):
        shutil.copy2(tok / name, build / name)
    link = build / "model.safetensors"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(weights.resolve())


def convert(build: Path, quantization: str, dst: Path):
    print(f"\n=== {quantization}: {build} -> {dst}")
    if dst.exists():
        shutil.rmtree(dst)
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
    cfg["alignment_heads"] = LARGE_V3_ALIGNMENT_HEADS
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("patched config.json: lang_ids=[51908] (<|fa|>), "
          f"alignment_heads={len(LARGE_V3_ALIGNMENT_HEADS)} (large-v3)")

    vocab_path = dst / "vocabulary.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if "" not in vocab:
        old = vocab[220]
        vocab[220] = ""
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        print(f"patched vocabulary[220]: {old!r} -> '' (marks model multilingual)")
    print("size:", round(sum(p.stat().st_size for p in dst.iterdir()) / 1e9, 2), "GB")


def validate(dst: Path, quantization: str):
    import ctranslate2
    import tokenizers

    ctm = ctranslate2.models.Whisper(str(dst))
    print(f"  {dst.name}: multilingual={ctm.is_multilingual} "
          f"num_languages={ctm.num_languages} n_mels={ctm.n_mels}")
    if not ctm.is_multilingual:
        sys.exit("model still reported as English-only")

    tok = tokenizers.Tokenizer.from_file(str(dst / "tokenizer.json"))
    ids = {
        t: tok.token_to_id(t)
        for t in ("<|endoftext|>", "<|startoftranscript|>", "<|fa|>",
                  "<|transcribe|>", "<|notimestamps|>", "<|0.00|>")
    }
    print("  tokenizer ids:", ids)
    assert ids["<|fa|>"] == 51908 and ids["<|notimestamps|>"] == 51972

    # float16 weights only run on CUDA; on this CPU box just check metadata.
    if quantization.startswith("int8"):
        from faster_whisper import WhisperModel

        model = WhisperModel(str(dst), device="cpu", compute_type="int8")
        assert model.model.is_multilingual
        print("  faster-whisper: loads on CPU, multilingual OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantization", choices=["int8_float32", "float16", "both"], default="both")
    ap.add_argument("--src", default=str(ROOT / "models/buzzasr-persian"))
    ap.add_argument("--tok", default=str(ROOT / "models/buzzasr-persian-tok"))
    ap.add_argument("--build-dir", default=str(ROOT / "models/buzzasr-persian-hf"))
    ap.add_argument("--out-prefix", default=str(ROOT / "models"))
    ap.add_argument("--min-weight-bytes", type=int, default=3_000_000_000)
    args = ap.parse_args()

    src, tok = Path(args.src), Path(args.tok)
    weights = check_weights(src, args.min_weight_bytes)
    build = Path(args.build_dir)
    assemble_hf_dir(src, tok, build, weights)

    targets = TARGETS if args.quantization == "both" else {args.quantization: TARGETS[args.quantization]}
    dirs = {q: Path(args.out_prefix) / name for q, name in targets.items()}
    for quant, dst in dirs.items():
        convert(build, quant, dst)

    print("\n=== validating")
    for quant, dst in dirs.items():
        validate(dst, quant)
    print("\nOK – ready for WhisperLiveKit")


if __name__ == "__main__":
    main()
