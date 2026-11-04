# Source this before running any uv/python command in this workspace.
# All caches live inside the project so the sandbox (workspace-write) allows writes.
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.cache/uv-python"
export HF_HOME="$PROJECT_ROOT/.cache/hf"
export HF_HUB_CACHE="$PROJECT_ROOT/.cache/hf/hub"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache/xdg"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
