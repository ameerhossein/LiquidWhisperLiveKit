"""Small, dependency-free helpers used by the Colab notebook.

Keeping these out of the notebook makes the layout and log parsing rules
testable without a GPU or a running tunnel.
"""
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Union

REQUIRED_CT2_FILES = (
    "model.bin",
    "config.json",
    "vocabulary.json",
    "tokenizer.json",
    "preprocessor_config.json",
)

_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _files_by_directory(files: Iterable[str]) -> dict[str, set[str]]:
    directories: dict[str, set[str]] = defaultdict(set)
    for path in files:
        parent, _, name = path.rpartition("/")
        if name:
            directories[parent].add(name)
    return directories


def find_ct2_subfolder(files: Iterable[str], requested: str = "") -> str:
    """Return the one directory in an HF repo that holds a complete CT2 model.

    ``""`` means the repository root. ``requested`` is an explicit
    ``HF_SUBFOLDER`` value; when supplied it must itself be complete.
    """
    directories = _files_by_directory(files)
    complete = {
        parent
        for parent, names in directories.items()
        if set(REQUIRED_CT2_FILES).issubset(names)
    }
    requested = requested.strip().strip("/")

    if requested:
        if requested not in complete:
            candidates = ", ".join(sorted(repr(path or "<root>") for path in complete)) or "none"
            raise ValueError(
                f"{requested!r} is not a complete CT2 directory. "
                f"Complete directories: {candidates}."
            )
        return requested

    if not complete:
        with_model = sorted(
            parent for parent, names in directories.items() if "model.bin" in names
        )
        candidates = ", ".join(repr(path or "<root>") for path in with_model) or "none"
        required = ", ".join(REQUIRED_CT2_FILES)
        raise ValueError(
            "No complete CTranslate2 directory found. "
            f"Directories containing model.bin: {candidates}. "
            f"Required files: {required}."
        )

    if len(complete) > 1:
        candidates = ", ".join(sorted(repr(path or "<root>") for path in complete))
        raise ValueError(
            "Multiple complete CTranslate2 directories found: "
            f"{candidates}. Set HF_SUBFOLDER to one of them."
        )

    return next(iter(complete))


def parse_tunnel_url(text: str) -> Union[str, None]:
    """Return the first Cloudflare quick-tunnel URL in a log, if present."""
    match = _TUNNEL_URL.search(text or "")
    return match.group(0) if match else None


def nvidia_library_dirs(site_packages: Union[str, Path]) -> list[str]:
    """Return existing NVIDIA library directories shipped in a venv."""
    root = Path(site_packages) / "nvidia"
    return [
        str(root / component / "lib")
        for component in ("cudnn", "cublas", "cuda_runtime")
        if (root / component / "lib").is_dir()
    ]
