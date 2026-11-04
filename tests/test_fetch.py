import hashlib
import io
import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

import fetch
from fetch_model import fetch_model, git_blob


class Response(io.BytesIO):
    def __init__(self, data=b"", status=206, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}

    def geturl(self):
        return "https://test.invalid/file"


def test_resume_starts_after_actual_bytes_not_allocated_blocks(tmp_path):
    part = tmp_path / "part"
    part.write_bytes(b"a" * 1001)
    request_ranges = []
    download = fetch.Download("https://test.invalid/file", 5000)
    download.current_url = lambda: download.origin

    def open_request(request, **kwargs):
        request_ranges.append(request.headers["Range"])
        return Response(b"b" * 3999, headers={"Content-Range": "bytes 1001-4999/5000"})

    with patch.object(fetch.urllib.request, "urlopen", open_request):
        fetch.fetch_range(download, part, 0, 4999, threading.Event())
    assert request_ranges == ["bytes=1001-4999"]
    assert part.read_bytes() == b"a" * 1001 + b"b" * 3999


@pytest.mark.parametrize("status,header", [(200, None), (206, "bytes 0-4/10"), (206, "bytes 5-9/11")])
def test_bad_range_does_not_write_any_bytes(tmp_path, status, header):
    download = fetch.Download("https://test.invalid/file", 10)
    download.current_url = lambda: download.origin
    part = tmp_path / "part"
    with patch.object(fetch.urllib.request, "urlopen", return_value=Response(
            b"01234", status, {"Content-Range": header})):
        with pytest.raises(fetch.ProtocolError):
            fetch.fetch_range(download, part, 5, 9, threading.Event())
    assert not part.exists()


def fake_server(data):
    def open_request(request, **kwargs):
        if request.get_method() == "HEAD":
            return Response(headers={"Content-Length": str(len(data)), "ETag": '"version1"'})
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        return Response(data[start:end + 1], headers={
            "Content-Range": f"bytes {start}-{end}/{len(data)}"})
    return open_request


def test_full_download_creates_parents_and_ignores_legacy_sparse_parts(tmp_path):
    output = tmp_path / "models" / "weights"
    output.parent.mkdir()
    Path(str(output) + ".part0").write_bytes(b"corrupt legacy data")
    data = b"correct checkpoint" * 200
    with patch.object(fetch.urllib.request, "urlopen", fake_server(data)):
        fetch.download_file("https://test.invalid/file", output, 8, hashlib.sha256(data).hexdigest())
    assert output.read_bytes() == data
    assert not Path(str(output) + ".download-v2").exists()


def test_digest_failure_preserves_previous_output(tmp_path):
    output = tmp_path / "model"
    output.write_bytes(b"previous usable model")
    with patch.object(fetch.urllib.request, "urlopen", fake_server(b"bad replacement")):
        with pytest.raises(fetch.ProtocolError, match="SHA256"):
            fetch.download_file("https://test.invalid/file", output, 3, "0" * 64)
    assert output.read_bytes() == b"previous usable model"


def test_premature_eof_retries_are_bounded(tmp_path):
    download = fetch.Download("https://test.invalid/file", 5)
    download.current_url = lambda: download.origin
    stop = threading.Event()
    with patch.object(fetch.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(
            headers={"Content-Range": "bytes 0-4/5"})), patch.object(stop, "wait"):
        with pytest.raises(RuntimeError, match="exhausted"):
            fetch.fetch_range(download, tmp_path / "part", 0, 4, stop, retries=1)


def test_manifest_mismatch_refuses_reusing_parts(tmp_path):
    output = tmp_path / "model"
    work = Path(str(output) + ".download-v2")
    work.mkdir()
    (work / "manifest.json").write_text(json.dumps({"version": 1}))
    with patch.object(fetch.urllib.request, "urlopen", fake_server(b"test")):
        with pytest.raises(fetch.ProtocolError, match="manifest mismatch"):
            fetch.download_file("https://test.invalid/file", output)
    assert not output.exists()


def test_model_bootstrap_downloads_small_assets_and_verifies_offline(tmp_path):
    asset = b'{"test":true}'
    manifest = {"repo": "test/model", "revision": "pinned", "files": {
        "config.json": {"git_blob": git_blob(asset)}}}
    with patch.object(fetch.urllib.request, "urlopen", return_value=Response(asset)):
        fetch_model(tmp_path / "fresh", manifest)
    fetch_model(tmp_path / "fresh", manifest, offline=True)
    (tmp_path / "fresh/config.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        fetch_model(tmp_path / "fresh", manifest, offline=True)
