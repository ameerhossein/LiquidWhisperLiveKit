#!/usr/bin/env python3
"""Rebuild installable wheels from uv's unpacked wheel cache (archive-v0).

The sandbox blocks writes to ~/.cache/uv, so a copy was made inside the project.
uv's cache stores *extracted* wheel trees; this script repackages each tree into
a proper .whl (name/version/tags taken from its .dist-info) so that
`uv pip install --no-index --find-links wheels/ ...` works fully offline.
"""
import os
import re
import sys
import zipfile
from pathlib import Path

CACHE = Path(sys.argv[1] if len(sys.argv) > 1 else ".cache/uv/archive-v0")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "wheels")
OUT.mkdir(parents=True, exist_ok=True)


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name)


def meta(path: Path):
    name = version = None
    tags = []
    for line in (path / "METADATA").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Name: ") and name is None:
            name = line[6:].strip()
        elif line.startswith("Version: ") and version is None:
            version = line[9:].strip()
        if name and version:
            break
    wheel_file = path / "WHEEL"
    if wheel_file.is_file():
        for line in wheel_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Tag: "):
                tags.append(line[5:].strip())
    return name, version, tags


def build(src: Path, whl: Path) -> int:
    count = 0
    with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, _dirs, files in os.walk(src):
            for f in files:
                full = Path(root) / f
                arc = str(full.relative_to(src))
                info = zipfile.ZipInfo(arc)
                info.compress_type = zipfile.ZIP_DEFLATED
                mode = full.stat().st_mode
                info.external_attr = (mode & 0xFFFF) << 16
                if full.is_symlink():
                    info.external_attr |= 0xA000 << 16
                    zf.writestr(info, os.readlink(full))
                else:
                    zf.writestr(info, full.read_bytes())
                count += 1
    return count


total = 0
seen = set()
for entry in sorted(CACHE.iterdir()):
    if not entry.is_dir():
        continue
    dist_infos = [p for p in entry.iterdir() if p.is_dir() and p.name.endswith(".dist-info")]
    if len(dist_infos) != 1:
        print(f"skip {entry.name}: {len(dist_infos)} dist-info", file=sys.stderr)
        continue
    di = dist_infos[0]
    name, version, tags = meta(di)
    if not (name and version and tags):
        print(f"skip {entry.name}: incomplete metadata", file=sys.stderr)
        continue
    # Compressed tag set (PEP 425): join component-wise with ".", e.g.
    # ["py2-none-any", "py3-none-any"] -> "py2.py3-none-any".
    parts = [t.split("-") for t in tags]
    if len({len(p) for p in parts}) == 1:
        tag_str = "-".join(
            ".".join(dict.fromkeys(col)) for col in zip(*parts)
        )
    else:  # mixed tag shapes: fall back to the first tag
        tag_str = tags[0]
    whl_name = f"{normalise(name)}-{version}-{tag_str}.whl"
    if whl_name in seen:
        continue
    seen.add(whl_name)
    out = OUT / whl_name
    n = build(entry, out)
    total += n
    print(f"{whl_name}  ({n} files)")

print(f"\n{len(seen)} wheels, {total} files -> {OUT}", file=sys.stderr)
