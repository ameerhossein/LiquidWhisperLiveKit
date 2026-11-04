#!/usr/bin/env python3
"""Score raw/normalized Persian text and optional manually aligned word times.

Normalization: NFKC, Persian kaf/yeh, ASCII digits, no vowel marks, ZWNJ as
space, collapsed whitespace. Display/output text is never modified.
"""
import argparse
import json
from pathlib import Path
import unicodedata

import numpy as np


def normalize(text):
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans(
        "كيى۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "کیی01234567890123456789"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.replace("\u200c", " ").split())


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def score(reference, hypothesis):
    result = {}
    for prefix, ref, hyp in (("raw", reference["text"], hypothesis["text"]),
                             ("normalized", normalize(reference["text"]), normalize(hypothesis["text"]))):
        result[prefix + "_cer"] = distance(ref, hyp) / max(len(ref), 1)
        result[prefix + "_wer"] = distance(ref.split(), hyp.split()) / max(len(ref.split()), 1)
    if reference.get("words") and hypothesis.get("words"):
        ref_words, hyp_words = reference["words"], hypothesis["words"]
        if [normalize(w["word"]) for w in ref_words] == [normalize(w["word"]) for w in hyp_words]:
            errors = [abs(float(r[key]) - float(h[key])) for r, h in zip(ref_words, hyp_words)
                      for key in ("start", "end")]
            result["boundary_mae_s"] = float(np.mean(errors))
            result["boundary_p95_s"] = float(np.percentile(errors, 95))
        else:
            result["boundary_status"] = "Word sequences differ; boundary comparison requires matching words"
    result["metrics"] = hypothesis.get("metrics", {})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="JSON containing text and optional manually aligned words")
    parser.add_argument("hypothesis", help="JSON from transcribe.py or stream_test.py")
    args = parser.parse_args()
    print(json.dumps(score(json.loads(Path(args.reference).read_text()),
                           json.loads(Path(args.hypothesis).read_text())), indent=2, ensure_ascii=False))
