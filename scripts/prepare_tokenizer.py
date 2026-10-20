#!/usr/bin/env python3
"""Build a WhisperLiveKit/faster-whisper-compatible tokenizer for BuzzASR/persian.

BuzzASR's vocab.json packs the 1609 special/timestamp tokens at ids 0..1608 *as
plain BPE entries*, while the same strings also live in added_tokens.json at
51865..53473 -- the ids the checkpoint actually uses (generation_config.json:
<|endoftext|>=51865, <|startoftranscript|>=51866, <|fa|>=51908,
<|transcribe|>=51968, <|notimestamps|>=51972, first timestamp <|0.00|>=51973).

transformers resolves a token string against the BPE vocab first, so it returns
the shadowed low ids (eot=0, sot=1, notimestamps=7, timestamp_begin=8...), which
silently breaks faster-whisper/CT2 decoding and timestamp handling.

Fix: keep all 51865 BPE ids (so text ids 1609..51864 stay put and the total is
exactly 53474) but rename the shadowed placeholder strings, then register the
real special tokens as added tokens in ascending id order starting at 51865.
"""
import argparse
import json
import shutil
from pathlib import Path

UNUSED = "<|UNUSED{}|>"

CHECKS = {
    "<|endoftext|>": 51865,
    "<|startoftranscript|>": 51866,
    "<|en|>": 51867,
    "<|fa|>": 51908,
    "<|yue|>": 51966,
    "<|translate|>": 51967,
    "<|transcribe|>": 51968,
    "<|startoflm|>": 51969,
    "<|startofprev|>": 51970,
    "<|nospeech|>": 51971,
    "<|notimestamps|>": 51972,
    "<|0.00|>": 51973,
    "<|30.00|>": 53473,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/buzzasr-persian")
    ap.add_argument("--dst", default="models/buzzasr-persian-tok")
    args = ap.parse_args()

    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    vocab = json.loads((src / "vocab.json").read_text(encoding="utf-8"))
    added = json.loads((src / "added_tokens.json").read_text(encoding="utf-8"))

    renamed = 0
    for token, idx in list(vocab.items()):
        if token in added:
            assert added[token] != idx, f"{token} already at its real id"
            del vocab[token]
            vocab[UNUSED.format(idx)] = idx
            renamed += 1
    print(f"bpe vocab: {len(vocab)} entries, renamed {renamed} shadowed specials")

    merges = [
        tuple(line.split())
        for line in (src / "merges.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and len(line.split()) == 2
    ]
    tok = Tokenizer(
        BPE(vocab=vocab, merges=merges, dropout=None, fuse_unk=False,
            continuing_subword_prefix="", end_of_word_suffix="")
    )
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    tokens_by_id = [t for t, _ in sorted(added.items(), key=lambda kv: kv[1])]
    assert len(tokens_by_id) == 1609, len(tokens_by_id)
    assert tokens_by_id[0] == "<|endoftext|>"
    added_count = tok.add_special_tokens(tokens_by_id)
    print(f"added {added_count} special tokens, ids "
          f"{tok.token_to_id('<|endoftext|>')}..{tok.token_to_id('<|30.00|>')}")

    tok.save(str(dst / "tokenizer.json"))

    # Sanity checks against the raw `tokenizers` object before involving transformers.
    ok = True
    for token, want in CHECKS.items():
        got = tok.token_to_id(token)
        ok &= got == want
        print(f"  [{'ok ' if got == want else 'BAD'}] {token:22s} {got} (want {want})")

    sample = "سلام، این یک آزمایش است. چطور هستید؟"
    enc = tok.encode(sample)
    back = tok.decode(enc.ids)
    print(f"round-trip: {sample!r}\n         -> {enc.ids[:12]}...\n         -> {back!r}")
    ok &= back.strip() == sample.strip()

    # Now make it loadable by transformers / CTranslate2 (full component set).
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(dst, use_fast=True)
    hf.save_pretrained(dst)
    print(f"transformers: {type(hf).__name__} len={len(hf)} (want 53474)")
    ok &= len(hf) == 53474
    for token, want in CHECKS.items():
        got = hf.convert_tokens_to_ids(token)
        ok &= got == want

    shutil.copy2(src / "normalizer.json", dst / "normalizer.json")

    if not ok:
        raise SystemExit("tokenizer validation failed")
    print("wrote", sorted(p.name for p in dst.iterdir()))


if __name__ == "__main__":
    main()
