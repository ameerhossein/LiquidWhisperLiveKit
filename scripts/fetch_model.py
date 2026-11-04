#!/usr/bin/env python3
"""Fetch and integrity-check every asset of the pinned BuzzASR snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from fetch import UA, download_file, sha256

ROOT = Path(__file__).resolve().parent.parent


def git_blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def fetch_model(destination, manifest, offline=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, checks in manifest["files"].items():
        target = destination / name
        url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{name}"
        if "sha256" in checks:
            if offline:
                if not target.is_file() or sha256(target) != checks["sha256"]:
                    raise ValueError(f"Missing or corrupt asset: {target}")
            else:
                download_file(url, target, expected_sha256=checks["sha256"])
        else:
            if target.is_file() and git_blob(target.read_bytes()) == checks["git_blob"]:
                continue
            if offline:
                raise ValueError(f"Missing or corrupt asset: {target}")
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as response:
                data = response.read(8 * 1024 * 1024)
            if git_blob(data) != checks["git_blob"]:
                raise ValueError(f"Integrity check failed: {name}")
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
    print(f"Verified {manifest['repo']} at {manifest['revision']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst", default=str(ROOT / "models/buzzasr-persian"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    fetch_model(args.dst, json.loads((ROOT / "model-manifest.json").read_text()), args.offline)
