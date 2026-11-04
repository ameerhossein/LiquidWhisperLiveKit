#!/usr/bin/env python3
"""Fail acceptance checks on empty text, broken timings or a missed reference."""
import argparse
import json
import math
from pathlib import Path


def validate(result, expected=None, words=False):
    text = result.get("text", "").strip()
    if not text:
        raise ValueError("No transcript text")
    if expected:
        # Whitespace differences do not change the acceptance sentence.
        if " ".join(expected.split()) != " ".join(text.split()):
            raise ValueError(f"Expected {expected!r}, received {text!r}")
    if words:
        timings = result.get("words") or []
        if not timings:
            raise ValueError("No word timestamps")
        previous = 0.0
        progressed = False
        for word in timings:
            start, end = float(word["start"]), float(word["end"])
            if not math.isfinite(start + end) or start < previous - 0.02 or end < start:
                raise ValueError("Invalid or nonmonotonic word timestamps")
            previous = start
            progressed |= end > start
        if not progressed:
            raise ValueError("Word timestamps collapsed")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--expected-file")
    parser.add_argument("--words", action="store_true")
    args = parser.parse_args()
    expected = Path(args.expected_file).read_text().strip() if args.expected_file else None
    print(validate(json.loads(Path(args.result).read_text()), expected, args.words))
