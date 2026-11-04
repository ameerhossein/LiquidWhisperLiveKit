#!/usr/bin/env python3
"""Bounded, resumable range downloads with atomic publication and optional SHA256.

New append-only parts live in OUT.download-v2/. Legacy sparse .partN files are
never trusted. The manifest binds parts to the resource, validator and layout.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "BuzzASR-fetch/3", "Accept-Encoding": "identity"}


class ProtocolError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Download:
    def __init__(self, origin, size, validator=None):
        self.origin, self.size, self.validator = origin, size, validator
        self.url, self.resolved_at = None, 0.0
        self.lock = threading.Lock()

    def current_url(self, force=False):
        with self.lock:
            if force or self.url is None or time.monotonic() - self.resolved_at > 900:
                with urllib.request.urlopen(urllib.request.Request(
                        self.origin, headers=UA, method="HEAD"), timeout=30) as response:
                    if int(response.headers["Content-Length"]) != self.size:
                        raise ProtocolError("Resource size changed during download")
                    if self.validator and response.headers.get("ETag") != self.validator:
                        raise ProtocolError("Resource validator changed during download")
                    self.url = response.geturl()
                    self.resolved_at = time.monotonic()
            return self.url


def fetch_range(download, part, start, end, stop, retries=8):
    part = Path(part)
    length = part.stat().st_size if part.exists() else 0
    if length > end - start + 1:
        raise ProtocolError(f"Part too large: {part}")
    pos = start + length
    failures = 0
    while pos <= end and not stop.is_set():
        try:
            headers = {**UA, "Range": f"bytes={pos}-{end}"}
            if download.validator and not download.validator.startswith("W/"):
                headers["If-Range"] = download.validator
            with urllib.request.urlopen(urllib.request.Request(
                    download.current_url(), headers=headers), timeout=30) as response:
                expected = f"bytes {pos}-{end}/{download.size}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ProtocolError(f"Expected HTTP 206 with Content-Range {expected!r}")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ProtocolError("Encoded range responses are unsupported")
                if response.headers.get("Content-Length") is not None and int(
                        response.headers["Content-Length"]) != end - pos + 1:
                    raise ProtocolError("Wrong range Content-Length")
                with part.open("ab") as stream:
                    while pos <= end and not stop.is_set():
                        block = response.read(min(65536, end - pos + 1))
                        if not block:
                            raise IOError("Premature EOF in range response")
                        stream.write(block)
                        stream.flush()
                        pos += len(block)
                    os.fsync(stream.fileno())
            if pos > end:
                return
        except ProtocolError:
            stop.set()
            raise
        except (OSError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                if exc.code in (401, 403):
                    download.current_url(force=True)
                elif exc.code not in (408, 429, 500, 502, 503, 504):
                    stop.set()
                    raise
            failures += 1
            if failures > retries:
                stop.set()
                raise RuntimeError(f"Range {start}-{end} exhausted retries") from exc
            stop.wait(min(2 ** failures, 20))
    if pos <= end:
        raise RuntimeError("Download cancelled before completion")


def download_file(url, out, connections=8, expected_sha256=None, retries=8):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # One writer per destination, including across processes. Keep the lock
    # inode stable: unlinking it would let another process lock a different inode.
    with Path(str(out) + ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another download is writing {out}") from exc
        return _download_file(url, out, connections, expected_sha256, retries)


def _download_file(url, out, connections=8, expected_sha256=None, retries=8):
    if connections < 1 or retries < 0:
        raise ValueError("connections must be positive and retries nonnegative")
    if expected_sha256 and not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise ValueError("Invalid SHA256 digest")
    expected_sha256 = expected_sha256.lower() if expected_sha256 else None
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and expected_sha256 and sha256(out) == expected_sha256:
        print(f"Verified existing {out}", flush=True)
        return out
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA, method="HEAD"), timeout=30) as response:
        size = int(response.headers["Content-Length"])
        validator = response.headers.get("ETag")
    if size <= 0:
        raise ValueError("Resource must have a positive Content-Length")
    chunks = min(connections, size)
    step = (size + chunks - 1) // chunks
    ranges = [(s, min(size - 1, s + step - 1)) for s in range(0, size, step)]
    manifest = dict(version=2, url=url, size=size, validator=validator,
                    sha256=expected_sha256, ranges=[list(r) for r in ranges])
    work = Path(str(out) + ".download-v2")
    work.mkdir(exist_ok=True)
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ProtocolError(f"Resume manifest mismatch: use the original resource/layout or move {work}")
        if not expected_sha256 and (not validator or validator.startswith("W/")):
            raise ProtocolError("Safe resume requires SHA256 or a strong ETag")
    else:
        if any(work.iterdir()):
            raise ProtocolError(f"Unidentified download parts in {work}")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    download = Download(url, size, validator)
    stop = threading.Event()
    try:
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [pool.submit(fetch_range, download, work / f"part{i}", s, e, stop, retries)
                       for i, (s, e) in enumerate(ranges)]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                stop.set()
                raise
        with tempfile.NamedTemporaryFile(dir=out.parent, prefix=out.name + ".assembling-", delete=False) as assembled:
            temporary = Path(assembled.name)
            try:
                digest = hashlib.sha256()
                for i, (start, end) in enumerate(ranges):
                    part = work / f"part{i}"
                    if part.stat().st_size != end - start + 1:
                        raise ProtocolError(f"Incomplete part {i}")
                    with part.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            assembled.write(block)
                            digest.update(block)
                assembled.flush()
                os.fsync(assembled.fileno())
                if expected_sha256 and digest.hexdigest() != expected_sha256:
                    shutil.rmtree(work)  # Do not resume content known to be corrupt.
                    raise ProtocolError("SHA256 mismatch; corrupt download parts discarded")
                os.replace(temporary, out)
            finally:
                temporary.unlink(missing_ok=True)
        shutil.rmtree(work)
        print(f"OK {out} ({size} bytes, SHA256 {digest.hexdigest()})", flush=True)
        return out
    finally:
        stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("out")
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--sha256")
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    download_file(args.url, args.out, args.connections, args.sha256, args.retries)


if __name__ == "__main__":
    main()
