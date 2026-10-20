#!/usr/bin/env python3
"""Create a tiny random Whisper checkpoint that uses BuzzASR's exact tokenizer.

The real checkpoint is 3 GB and still downloading; this stand-in has the same
vocabulary (53474 ids), mel bin count and tokenizer files but only 2 layers, so
the CTranslate2 conversion + WhisperLiveKit plumbing can be validated cheaply.
Its transcripts are meaningless -- it only proves the wiring works.
"""
import json
import shutil
from pathlib import Path

import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

SRC = Path("models/buzzasr-persian")
TOK = Path("models/buzzasr-persian-tok")
DST = Path("models/tiny-test-src")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    real = json.loads((SRC / "config.json").read_text())
    cfg = WhisperConfig(
        vocab_size=real["vocab_size"],
        num_mel_bins=real["num_mel_bins"],
        d_model=128,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=512,
        decoder_ffn_dim=512,
        max_source_positions=real["max_source_positions"],
        max_target_positions=real["max_target_positions"],
        bos_token_id=real["bos_token_id"],
        eos_token_id=real["eos_token_id"],
        pad_token_id=real["pad_token_id"],
        decoder_start_token_id=real["decoder_start_token_id"],
    )
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(cfg)
    model.generation_config.language = "fa"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.save_pretrained(DST, safe_serialization=True)

    for name in ("tokenizer.json", "tokenizer_config.json", "normalizer.json"):
        shutil.copy2(TOK / name, DST / name)
    shutil.copy2(SRC / "preprocessor_config.json", DST / "preprocessor_config.json")

    print("wrote", sorted(p.name for p in DST.iterdir()))
    print("params:", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
