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
    assert 'WLK_VAD_FILTER = "1"' in source
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
    assert '"WLK_VAD_FILTER": str(WLK_VAD_FILTER)' in source
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


def test_stop_cell_is_opt_in():
    source = code(notebook())
    assert 'STOP_SERVER = False #@param {type:"boolean"}' in source
    assert "if not STOP_SERVER" in source


def test_self_test_runs_before_the_public_tunnel():
    data = notebook()
    cells = ["".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code"]
    self_test = next(i for i, source in enumerate(cells) if "testdata/verify-fa.wav" in source)
    tunnel = next(i for i, source in enumerate(cells) if '"--no-autoupdate"' in source)
    assert self_test < tunnel


def test_server_and_tunnel_drop_hf_token():
    source = code(notebook())
    assert 'env.pop("HF_TOKEN", None)' in source
    assert 'env.pop("HUGGING_FACE_HUB_TOKEN", None)' in source
    assert 'tunnel_env.pop("HF_TOKEN", None)' in source


def test_setup_shares_the_project_hf_cache():
    source = code(notebook())
    assert '"HF_HOME": str(PROJECT_DIR / ".cache" / "hf")' in source
    assert '"HF_HUB_CACHE": str(PROJECT_DIR / ".cache" / "hf" / "hub")' in source


def test_cudnn_is_checked_and_remediated():
    source = code(notebook())
    assert "find_cudnn_library(" in source
    assert '"float16", "compute type"' in source


def test_gpu_check_guards_nvidia_smi():
    assert 'shutil.which("nvidia-smi")' in code(notebook())


def test_all_code_cells_compile():
    for index, cell in enumerate(notebook()["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_notebook_structure_and_form_cells():
    data = notebook()
    assert len(data["cells"]) == 12
    assert data["cells"][0]["cell_type"] == "markdown"
    assert data["cells"][-1]["cell_type"] == "markdown"
    code_cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 10
    assert code_cells[0]["metadata"].get("cellView") == "form"
    assert code_cells[-1]["metadata"].get("cellView") == "form"


def test_stop_logs_are_opt_in():
    source = code(notebook())
    assert 'PRINT_LOGS = False #@param {type:"boolean"}' in source
    assert "if PRINT_LOGS" in source


def test_stop_process_checks_the_expected_command():
    source = code(notebook())
    assert 'stop_process(SERVER_PID, ("run.sh", "scripts/serve.py"))' in source
    assert 'stop_process(TUNNEL_PID, ("cloudflared",))' in source


def test_tunnel_health_requires_ready_and_does_not_link_unverified():
    data = notebook()
    cells = ["".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code"]
    tunnel = next(source for source in cells if '"--no-autoupdate"' in source)
    assert "health_payload_is_ready(" in code(data)
    assert "except ValueError" in tunnel
    assert "if tunnel_health:" in tunnel
    assert tunnel.index("display(HTML") > tunnel.index("if tunnel_health:")
    assert "unverified" in tunnel.lower()
