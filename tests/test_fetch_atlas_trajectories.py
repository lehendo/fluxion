"""The ATLAS trajectory downloader: chunked ranged GETs, resume, retry, integrity checks. Uses a local HTTP server; no network access."""
import io
import re
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from scripts.fetch_atlas_trajectories import download, sha256, total_size, verify_and_extract


def make_zip(n: int = 200_000) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("p.pdb", "ATOM\n" * 10)
        z.writestr("p_R1.xtc", bytes(range(256)) * (n // 256))
    return buf.getvalue()


class Server:
    def __init__(self, payload: bytes, fail_first: int = 0, ignore_range: bool = False):
        outer = self
        self.payload, self.fail_first, self.requests, self.fails = payload, fail_first, [], 0

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                rng = self.headers.get("Range")
                outer.requests.append(rng)
                if ignore_range or not rng:
                    self.send_response(200); self.send_header("Content-Length", str(len(outer.payload))); self.end_headers(); self.wfile.write(outer.payload); return
                m = re.match(r"bytes=(\d+)-(\d+)", rng)
                a, b = int(m.group(1)), min(int(m.group(2)), len(outer.payload) - 1)
                if b > 0 and outer.fails < outer.fail_first:        # fail data chunks (not the 0-0 size probe)
                    outer.fails += 1
                    self.send_response(503); self.end_headers(); return
                body = outer.payload[a:b + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(outer.payload)}"); self.send_header("Content-Length", str(len(body)))
                self.send_header("Last-Modified", "Sat, 30 Sep 2023 09:37:09 GMT"); self.end_headers(); self.wfile.write(body)

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/ATLAS/protein/xxxx_A"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def srv():
    made = []
    def make(*a, **k):
        s = Server(*a, **k); made.append(s); return s
    yield make
    for s in made:
        s.close()


def test_total_size_reads_content_range(srv):
    s = srv(make_zip())
    n, h = total_size(s.url, requests.Session())
    assert n == len(s.payload) and "last-modified" in {k.lower() for k in h}


def test_full_download_is_chunked_and_exact(srv, tmp_path):
    s = srv(make_zip())
    info = download(s.url, tmp_path / "a.zip", chunk=50_000)
    assert (tmp_path / "a.zip").read_bytes() == s.payload and info["bytes"] == len(s.payload)
    assert len([r for r in s.requests if r != "bytes=0-0"]) >= 4                 # several ranged chunks, not one stream
    assert not (tmp_path / "a.zip.part").exists()


def test_resume_continues_from_the_partial_file(srv, tmp_path):
    s = srv(make_zip())
    (tmp_path / "a.zip.part").write_bytes(s.payload[:120_000])
    download(s.url, tmp_path / "a.zip", chunk=50_000)
    assert (tmp_path / "a.zip").read_bytes() == s.payload
    assert all(int(re.match(r"bytes=(\d+)-", r).group(1)) in (0, 120_000, 170_000, 220_000) for r in s.requests if r)   # never re-requested the first 120k bytes


def test_transient_failures_are_retried_on_the_same_range(srv, tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.fetch_atlas_trajectories.time.sleep", lambda s: None)
    s = srv(make_zip(), fail_first=3)
    download(s.url, tmp_path / "a.zip", chunk=70_000)
    assert (tmp_path / "a.zip").read_bytes() == s.payload and s.fails == 3


def test_server_without_range_support_is_an_error_not_a_silent_full_download(srv, tmp_path):
    s = srv(make_zip(), ignore_range=True)
    with pytest.raises(RuntimeError, match="206"):
        download(s.url, tmp_path / "a.zip")


def test_a_partial_file_longer_than_the_resource_is_discarded(srv, tmp_path):
    s = srv(make_zip())
    (tmp_path / "a.zip.part").write_bytes(b"x" * (len(s.payload) + 10))
    download(s.url, tmp_path / "a.zip", chunk=90_000)
    assert (tmp_path / "a.zip").read_bytes() == s.payload


def test_extraction_goes_into_a_new_directory_and_refuses_an_existing_one(tmp_path):
    z = tmp_path / "p.zip"; z.write_bytes(make_zip())
    out = tmp_path / "out"
    info = verify_and_extract(z, out)
    assert set(info["members"]) == {"p.pdb", "p_R1.xtc"} and (out / "p_R1.xtc").exists()
    with pytest.raises(FileExistsError):
        verify_and_extract(z, out)


def test_corrupt_archive_is_detected_before_anything_is_extracted(tmp_path):
    data = bytearray(make_zip()); data[5000] ^= 0xFF                              # flip a byte inside stored member data: CRC mismatch
    z = tmp_path / "bad.zip"; z.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="corrupt member"):
        verify_and_extract(z, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_sha256_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "f"; p.write_bytes(b"abc" * 1000)
    assert sha256(p) == hashlib.sha256(b"abc" * 1000).hexdigest()
