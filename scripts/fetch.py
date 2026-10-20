#!/usr/bin/env python3
"""Resumable multi-connection downloader for flaky links.

Hugging Face's CDN redirects to a *signed* URL that expires after ~1 h, so the
final URL is cached but re-resolved every 15 min and immediately on HTTP 401/403.

The file is split into fixed byte ranges (one .partN per range) written
sequentially. Each part is pre-allocated as a sparse file, so a restart resumes
from SEEK_HOLE -- for a sequentially written sparse file that is exactly the end
of the data written so far. The range layout only depends on file size and
--connections, so keep --connections unchanged when resuming.

Usage: fetch.py URL OUTFILE [--connections N]
"""
import argparse
import os
import threading
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) fetch/2.0"}
REFRESH_EVERY = 900  # seconds
_lock = threading.Lock()
_state = {"url": None, "ts": 0.0, "origin": None}


def resolve(origin):
    req = urllib.request.Request(origin, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.geturl(), int(resp.headers["Content-Length"])


def current_url(force=False):
    with _lock:
        stale = time.time() - _state["ts"] > REFRESH_EVERY
        if force or stale or not _state["url"]:
            url, _ = resolve(_state["origin"])
            _state["url"] = url
            _state["ts"] = time.time()
        return _state["url"]


def written_prefix(path):
    """Bytes already written into a sequentially-written sparse part file."""
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.lseek(fd, 0, os.SEEK_HOLE)
    finally:
        os.close(fd)


def fetch_range(part, start, end, stop_flag, stats, idx):
    pos = start + min(written_prefix(part), end - start + 1)
    attempts = 0
    while pos <= end and not stop_flag.is_set():
        try:
            req = urllib.request.Request(
                current_url(), headers={**UA, "Range": f"bytes={pos}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status not in (200, 206):
                    raise IOError(f"http {resp.status}")
                with open(part, "r+b") as f:
                    f.seek(pos - start)
                    while pos <= end and not stop_flag.is_set():
                        block = resp.read(min(1 << 16, end - pos + 1))
                        if not block:
                            break
                        f.seek(pos - start)
                        f.write(block)
                        pos += len(block)
                        stats[idx] = pos - start
            attempts = 0
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                print(f"[part{idx}] {exc.code}: refreshing signed url", flush=True)
                try:
                    current_url(force=True)
                except Exception as inner:  # noqa: BLE001
                    print(f"[part{idx}] refresh failed: {inner!r}", flush=True)
            attempts += 1
            time.sleep(min(2 ** min(attempts, 5), 20))
        except Exception as exc:  # noqa: BLE001 - retry everything
            attempts += 1
            delay = min(2 ** min(attempts, 6), 30)
            print(f"[part{idx}] {exc!r}; retry in {delay}s", flush=True)
            time.sleep(delay)
    stats[idx] = pos - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--connections", type=int, default=8)
    args = ap.parse_args()

    _state["origin"] = args.url
    url, size = resolve(args.url)
    with _lock:
        _state["url"] = url
        _state["ts"] = time.time()
    print(f"final url: {url}\nsize: {size} bytes", flush=True)

    chunks = max(1, args.connections)
    step = (size + chunks - 1) // chunks
    ranges = [(i * step, min(size - 1, (i + 1) * step - 1)) for i in range(chunks)]

    stop = threading.Event()
    stats = [0] * len(ranges)
    threads = []
    for i, (s, e) in enumerate(ranges):
        part = f"{args.out}.part{i}"
        if not os.path.exists(part):
            with open(part, "wb") as f:
                f.truncate(e - s + 1)
        stats[i] = min(written_prefix(part), e - s + 1)
        if stats[i]:
            print(f"resuming part{i} at {stats[i] / 1e6:.1f} MB", flush=True)
        t = threading.Thread(target=fetch_range, args=(part, s, e, stop, stats, i), daemon=True)
        t.start()
        threads.append(t)

    start_time = time.time()
    base = sum(stats)
    try:
        last = -1
        while any(t.is_alive() for t in threads):
            time.sleep(15)
            done = sum(stats)
            if done != last:
                elapsed = max(time.time() - start_time, 1)
                rate = (done - base) / elapsed if done > base else 0
                eta = (size - done) / rate if rate else 0
                print(
                    f"{done/1e6:8.1f}/{size/1e6:.1f} MB ({100*done/size:5.1f}%) "
                    f"{rate/1024:6.1f} KB/s eta {eta/3600:5.2f} h",
                    flush=True,
                )
                last = done
    except KeyboardInterrupt:
        stop.set()
        raise SystemExit("interrupted; re-run to resume")

    with open(args.out, "wb") as out:
        for i in range(len(ranges)):
            with open(f"{args.out}.part{i}", "rb") as f:
                while True:
                    block = f.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
    got = os.path.getsize(args.out)
    if got != size:
        raise SystemExit(f"size mismatch: got {got}, want {size}")
    for i in range(len(ranges)):
        os.unlink(f"{args.out}.part{i}")
    print(f"OK {args.out} ({got} bytes)", flush=True)


if __name__ == "__main__":
    main()
