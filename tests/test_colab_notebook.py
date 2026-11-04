import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_t4_streaming.ipynb"


def notebook():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert data["cells"]
    return data


def code(data):
    return "\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code")


def text(data):
    return "\n".join("".join(cell["source"]) for cell in data["cells"])


def test_notebook_exists_and_is_valid_json():
    assert NOTEBOOK.is_file()
    notebook()


def test_config_defaults():
    source = code(notebook())
    assert 'HF_REPO_ID = "ameerhossein/liquidwhisper-neyshekar-ct2-gpu"' in source
    assert "WLK_BEAM = 1" in source
    assert 'WLK_TEMPERATURE = "0.0"' in source
    assert 'ASR_COALESCE = "0"' in source
    assert "PORT = 8000" in source


def test_uses_secret_and_tested_helpers():
    source = code(notebook())
    assert 'userdata.get("HF_TOKEN")' in source
    assert "find_ct2_subfolder(files, HF_SUBFOLDER)" in source
    assert "parse_tunnel_url(" in source


def test_runtime_contract():
    source = code(notebook())
    assert "nvidia-smi" in source
    assert "No CUDA GPU detected" in source
    assert "T4 GPU" in source
    assert "401" in source and "403" in source and "404" in source
    assert '"HOST": "0.0.0.0"' in source
    assert '"DEVICE": "gpu"' in source
    assert '"--pcm-input"' in source
    assert '"ASR_COALESCE": str(ASR_COALESCE)' in source
    assert '"nvidia-cudnn-cu12==9.*"' in source
    assert 'get_supported_compute_types("cuda")' in source
    assert "float16" in source
    assert "/health" in source
    assert '"ready"' in source
    assert '"--no-autoupdate"' in source
    assert "trycloudflare.com" in source
    assert "testdata/verify-fa.wav" in source


def test_public_tunnel_is_warned_about():
    assert "public while" in text(notebook()).lower()


def test_no_hf_token_literal():
    source = text(notebook())
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source)
