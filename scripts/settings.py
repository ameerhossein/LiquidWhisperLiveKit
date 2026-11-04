"""Shared, explicit runtime and decoding configuration for every entry point."""
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_MODEL_FILES = ("model.bin", "config.json", "vocabulary.json",
                        "tokenizer.json", "preprocessor_config.json")


def device_name(value=None):
    value = (value or os.environ.get("DEVICE", "cpu")).lower()
    if value == "gpu":
        value = "cuda"
    if value not in ("cpu", "cuda"):
        raise ValueError("DEVICE must be cpu, gpu, or cuda; Metal is not supported")
    return value


def model_path(device=None):
    device = device_name(device)
    local = os.environ.get("MODEL_DIR")
    repo = os.environ.get("HF_REPO_ID")
    if local and repo:
        raise ValueError("Set MODEL_DIR or HF_REPO_ID, not both")
    if local:
        return local
    if repo:
        return str(download_model(repo, os.environ.get("HF_REVISION"),
                                  os.environ.get("HF_SUBFOLDER", "")))
    return str(ROOT / "models" / (
        "buzzasr-persian-ct2-int8" if device == "cpu"
        else "buzzasr-persian-ct2-fp16"))


def download_model(repo_id, revision=None, subfolder=""):
    """Resolve a prepared CT2 artifact using the HF cache and saved credentials."""
    from pathlib import PurePosixPath
    from huggingface_hub import snapshot_download

    folder = PurePosixPath(subfolder or ".")
    if folder.is_absolute() or ".." in folder.parts or "\\" in subfolder:
        raise ValueError("HF_SUBFOLDER must be a relative path within the repo")
    prefix = "" if folder == PurePosixPath(".") else folder.as_posix() + "/"
    snapshot = snapshot_download(
        repo_id=repo_id, revision=revision or None,
        allow_patterns=[prefix + name for name in REQUIRED_MODEL_FILES],
        local_files_only=os.environ.get("HF_HUB_OFFLINE", "").upper() in ("1", "ON", "YES", "TRUE"),
    )
    try:
        return require_model(Path(snapshot) / folder)
    except ValueError as exc:
        raise ValueError(
            f"{repo_id}/{prefix} must contain a prepared CTranslate2 model. "
            "Export HF weights to CT2 in the stt project first. " + str(exc)
        ) from exc


def model_options(device=None, threads=None):
    device = device_name(device)
    if device == "cuda":
        import ctranslate2
        if ctranslate2.get_cuda_device_count() < 1:
            raise RuntimeError("CUDA was requested, but CTranslate2 found no CUDA device")
    threads = int(os.environ.get("WLK_CPU_THREADS", "0")) if threads is None else threads
    workers = int(os.environ.get("WLK_NUM_WORKERS", "1"))
    if threads < 0 or workers < 1:
        raise ValueError("WLK_CPU_THREADS must be >= 0 and WLK_NUM_WORKERS must be >= 1")
    return dict(device=device, compute_type="int8_float32" if device == "cpu" else "float16",
                cpu_threads=threads, num_workers=workers)


def decode_options(profile=None, beam=None):
    profile = profile or os.environ.get("WLK_PROFILE", "tuned")
    if profile not in ("tuned", "baseline"):
        raise ValueError("WLK_PROFILE must be tuned or baseline")
    tuned = profile == "tuned"
    opts = dict(beam_size=int(os.environ.get("WLK_BEAM", "1" if tuned else "5")),
                condition_on_previous_text=os.environ.get(
                    "WLK_CONDITION_ON_PREVIOUS", "0" if tuned else "1") != "0",
                no_repeat_ngram_size=int(os.environ.get("WLK_NO_REPEAT_NGRAM", "3" if tuned else "0")),
                repetition_penalty=float(os.environ.get("WLK_REPETITION_PENALTY", "1.2" if tuned else "1")),
                vad_filter=os.environ.get("WLK_VAD_FILTER", "0") == "1")
    threshold = os.environ.get("WLK_NO_SPEECH_THRESHOLD", "0.8" if tuned else "0.6")
    if threshold.lower() not in ("", "none"):
        opts["no_speech_threshold"] = float(threshold)
    temperature = os.environ.get("WLK_TEMPERATURE", "0.0" if tuned else "")
    if temperature.lower() not in ("", "none"):
        opts["temperature"] = float(temperature)
    if beam is not None:
        opts["beam_size"] = beam
    if opts["beam_size"] < 1 or opts["no_repeat_ngram_size"] < 0 or opts["repetition_penalty"] <= 0:
        raise ValueError("Invalid beam size, ngram size, or repetition penalty")
    for key in ("temperature", "repetition_penalty", "no_speech_threshold"):
        if key in opts and (not math.isfinite(opts[key]) or opts[key] < 0):
            raise ValueError(f"Invalid {key}")
    if opts.get("no_speech_threshold", 0) > 1:
        raise ValueError("no_speech_threshold must be between 0 and 1")
    return opts


def print_settings(options, runtime):
    print("Resolved settings: " + json.dumps({"decode": options, "runtime": runtime}, sort_keys=True), flush=True)


def require_model(path):
    path = Path(path)
    for name in REQUIRED_MODEL_FILES:
        file = path / name
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"Incomplete model: missing or empty {file}")
    return path


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-model")
    action.add_argument("--resolve-model", action="store_true")
    try:
        args = parser.parse_args()
        if args.resolve_model:
            print(require_model(model_path()))
        else:
            require_model(args.check_model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
