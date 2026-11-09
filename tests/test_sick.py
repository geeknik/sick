"""Unit tests for sick.py -- SiCK (Suspicious Image Collection Kit).

Two groups:
  * TestCrawl / TestMain / TestProcessLocalFolder -- regression tests for the
    crawler bugs (visited-set, off-site links, missing timeout, per-page CSV
    overwrite, dead --verbose flag, wrong progress totals). These were written
    FIRST and failed against the buggy code (RED), then the fixes made them pass.
  * TestImageHelpers -- characterization tests for the pure image-analysis
    helpers (they exercise existing, unchanged behaviour).
"""
import base64
import csv
import io
import json
import logging
import shutil
import sys
from unittest import mock
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

import sick

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_row(url):
    """Return a 16-column result row like process_image produces."""
    return [url] + ["x"] * 15


def make_png_bytes(color=(128, 128, 128), size=(32, 32)):
    """Return PNG file content for a solid-color image."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def make_png_file(path, color=(128, 128, 128), size=(32, 32)):
    """Write a solid-color PNG to *path* and return the path."""
    Path(path).write_bytes(make_png_bytes(color, size))
    return Path(path)


def fake_requests_get(url_map):
    """Build a mock sick.requests.get serving HTML from url_map.

    Unknown URLs raise requests.ConnectionError (as a real network failure
    would), which crawl() must tolerate.
    """
    def _get(url, **_kwargs):
        if url not in url_map:
            raise requests.ConnectionError(f"unexpected URL: {url}")
        response = mock.MagicMock()
        response.content = url_map[url].encode()
        response.raise_for_status.return_value = None
        return response

    return mock.MagicMock(side_effect=_get)


@pytest.fixture(autouse=True)
def restore_log_level():
    """Keep logger level changes from leaking between tests."""
    yield
    sick.logger.setLevel(logging.INFO)


class TestCrawl:
    """Regression tests for the crawler bugs."""

    def test_does_not_refetch_visited_pages(self):
        """A<->B link cycles must not re-fetch an already visited page."""
        pages = {
            "http://site.test/a": '<a href="/b">b</a>',
            "http://site.test/b": '<a href="/a">a</a>',
        }
        get = fake_requests_get(pages)
        with mock.patch("sick.requests.get", get), \
                mock.patch("sick.process_image", return_value=None):
            sick.crawl("http://site.test/a", depth=5)
        fetched = sorted(call.args[0] for call in get.call_args_list)
        assert fetched == ["http://site.test/a", "http://site.test/b"]

    def test_stays_on_start_host(self):
        """Links pointing at another host must never be fetched."""
        pages = {"http://site.test/a": '<a href="http://evil.test/b">x</a>'}
        get = fake_requests_get(pages)
        with mock.patch("sick.requests.get", get), \
                mock.patch("sick.process_image", return_value=None):
            sick.crawl("http://site.test/a", depth=3)
        fetched = [call.args[0] for call in get.call_args_list]
        assert fetched == ["http://site.test/a"]

    def test_ignores_non_http_links(self):
        """mailto:/javascript:/tel: links must not be fetched."""
        pages = {
            "http://site.test/a": (
                '<a href="mailto:a@b.c">m</a>'
                '<a href="javascript:void(0)">j</a>'
                '<a href="tel:+1234">t</a>'
            )
        }
        get = fake_requests_get(pages)
        with mock.patch("sick.requests.get", get), \
                mock.patch("sick.process_image", return_value=None):
            sick.crawl("http://site.test/a", depth=2)
        assert get.call_count == 1  # only the start page

    def test_fragment_links_do_not_refetch_page(self):
        """href='#section' resolves to the current page and must be skipped."""
        pages = {"http://site.test/a": '<a href="#top">top</a>'}
        get = fake_requests_get(pages)
        with mock.patch("sick.requests.get", get), \
                mock.patch("sick.process_image", return_value=None):
            sick.crawl("http://site.test/a", depth=3)
        assert get.call_count == 1

    def test_fetch_uses_timeout_and_user_agent(self):
        """Page fetches must have a timeout and a User-Agent header."""
        pages = {"http://site.test/a": "<html></html>"}
        get = fake_requests_get(pages)
        with mock.patch("sick.requests.get", get), \
                mock.patch("sick.process_image", return_value=None):
            sick.crawl("http://site.test/a", depth=1)
        kwargs = get.call_args.kwargs
        assert kwargs.get("timeout") == 10
        assert "User-Agent" in kwargs.get("headers", {})

    def test_duplicate_images_processed_once(self):
        """The same image URL on two pages must be processed only once."""
        pages = {
            "http://site.test/a": '<img src="/shared.png"><a href="/b">b</a>',
            "http://site.test/b": '<img src="/shared.png">',
        }
        get = fake_requests_get(pages)
        process = mock.MagicMock(return_value=None)
        with mock.patch("sick.requests.get", get), mock.patch("sick.process_image", process):
            sick.crawl("http://site.test/a", depth=2)
        assert process.call_count == 1
        assert process.call_args.args[0] == "http://site.test/shared.png"

    def test_accumulates_results_across_pages(self):
        """crawl() returns rows from every page, not just the last one."""
        pages = {
            "http://site.test/a": '<img src="/one.png"><a href="/b">b</a>',
            "http://site.test/b": '<img src="/two.png">',
        }
        get = fake_requests_get(pages)

        def fake_process(url):
            return make_row(url)

        display = mock.MagicMock()
        with mock.patch("sick.requests.get", get), \
                mock.patch("sick.process_image", side_effect=fake_process), \
                mock.patch("sick.display_results", display):
            results = sick.crawl("http://site.test/a", depth=2)
        assert sorted(row[0] for row in results) == [
            "http://site.test/one.png",
            "http://site.test/two.png",
        ]
        # per-page display/CSV writes were the data-loss bug; only main() reports
        display.assert_not_called()


    def test_crawl_respects_max_pages(self):
        """A page cap must bound total fetches on unbounded sites."""
        pages = {f"http://site.test/p{i}": f'<a href="/p{i+1}">next</a>' for i in range(20)}
        get = fake_requests_get(pages)
        with mock.patch("sick.requests.get", get), \
                mock.patch("sick.process_image", return_value=None):
            sick.crawl("http://site.test/p0", depth=50, max_pages=5)
        assert get.call_count == 5

    def test_crawl_survives_deep_sites_without_recursion(self):
        """A 3000-page chain must not blow the recursion stack."""
        pages = {f"http://site.test/p{i}": f'<a href="/p{i+1}">next</a>' for i in range(3000)}
        get = fake_requests_get(pages)
        with mock.patch("sick.requests.get", get), \
                mock.patch("sick.process_image", return_value=None):
            results = sick.crawl("http://site.test/p0", depth=5000, max_pages=3000)
        assert get.call_count == 3000
        assert results == []

    def test_crawl_one_failing_image_does_not_abort_page(self):
        """An exception from one image must not kill the page's other rows."""
        pages = {"http://site.test/a": '<img src="/one.png"><img src="/two.png">'}
        get = fake_requests_get(pages)
        process = mock.MagicMock(side_effect=[Exception("boom"), make_row("http://site.test/two.png")])
        with mock.patch("sick.requests.get", get), mock.patch("sick.process_image", process):
            results = sick.crawl("http://site.test/a", depth=1)
        assert len(results) == 1
        assert results[0][0] == "http://site.test/two.png"

    def test_crawl_redirect_landing_images_only_no_foreign_links(self):
        """Redirected landings keep correct image URLs but foreign links
        are NOT followed: the crawl's scope is the operator's start host."""
        landing = mock.MagicMock()
        landing.url = "http://final.test/landing"
        landing.content = b'<img src="pic.png"><a href="other.html">x</a>'
        landing.raise_for_status.return_value = None
        get = mock.MagicMock(return_value=landing)
        process = mock.MagicMock(return_value=None)
        with mock.patch("sick.requests.get", get), mock.patch("sick.process_image", process):
            sick.crawl("http://start.test/go", depth=3)
        process.assert_called_once_with("http://final.test/pic.png")
        fetched = [call.args[0] for call in get.call_args_list]
        assert fetched == ["http://start.test/go"]  # no foreign link followed

    def test_crawl_open_redirect_cannot_chain_hosts(self):
        """An open redirect must not widen the crawl to another site."""
        evil = mock.MagicMock()
        evil.url = "http://evil.test/entry"
        evil.content = b'<a href="/more">m</a><a href="http://evil2.test/x">x</a>'
        evil.raise_for_status.return_value = None
        get = mock.MagicMock(return_value=evil)
        with mock.patch("sick.requests.get", get), \
                mock.patch("sick.process_image", return_value=None):
            sick.crawl("http://start.test/redirect", depth=10)
        assert get.call_count == 1

    def test_crawl_same_host_ignores_default_port_and_case(self):
        """SITE.TEST:80 and site.test are the same http host."""
        pages = {"http://site.test/a": '<a href="HTTP://SITE.TEST:80/b">b</a>',
                 "http://site.test/b": "<html></html>"}
        get = fake_requests_get(pages)
        with mock.patch("sick.requests.get", get), \
                mock.patch("sick.process_image", return_value=None):
            sick.crawl("http://site.test/a", depth=2)
        assert get.call_count == 2


class TestNudeDetectorInit:
    """The lazy model load must be thread-safe under the thread pool."""

    def test_initialize_nude_detector_loads_model_once(self):
        import threading
        import time
        calls = []

        class SlowDetector:
            """Test double that models slow NudeDetector model loading."""
            def __init__(self):
                time.sleep(0.05)
                calls.append(1)

        previous = sick.nude_detector
        sick.nude_detector = None
        try:
            with mock.patch("sick.NudeDetector", SlowDetector):
                threads = [threading.Thread(target=sick.initialize_nude_detector)
                           for _ in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
        finally:
            sick.nude_detector = previous
        assert len(calls) == 1


class TestMain:
    """Regression tests for argument handling."""

    def test_verbose_flag_enables_debug_logging(self, monkeypatch):
        """-v must actually change the log level."""
        monkeypatch.setattr(sys, "argv", ["sick.py", "-v", "-u", "http://site.test/"])
        with mock.patch("sick.crawl", return_value=[]), \
                mock.patch("sick.display_results"):
            sick.main()
        assert sick.logger.getEffectiveLevel() == logging.DEBUG

    def test_crawl_results_displayed_once(self, monkeypatch):
        """main() passes the accumulated crawl results to display exactly once."""
        rows = [make_row("http://site.test/a.png")]
        monkeypatch.setattr(sys, "argv", ["sick.py", "-u", "http://site.test/"])
        crawl = mock.MagicMock(return_value=rows)
        display = mock.MagicMock()
        with mock.patch("sick.crawl", crawl), mock.patch("sick.display_results", display):
            sick.main()
        crawl.assert_called_once_with("http://site.test/", 1, 100)
        display.assert_called_once_with(rows, "results.csv")


class TestProcessLocalFolder:
    """Regression test for the progress-bar total."""

    def test_progress_total_counts_only_image_files(self, tmp_path):
        """The progress bar total must equal the number of image files."""
        make_png_file(tmp_path / "one.png")
        make_png_file(tmp_path / "two.jpg")
        (tmp_path / "notes.txt").write_text("not an image")

        with mock.patch("sick.process_image", return_value=None) as process, \
                mock.patch("sick.Progress") as progress_cls:
            sick.process_local_folder(str(tmp_path))

        instance = progress_cls.return_value.__enter__.return_value
        assert instance.add_task.call_args.kwargs["total"] == 2
        assert instance.update.call_count == 2
        assert process.call_count == 2


class TestProcessImageOutput:
    """Regression tests: OCR and EXIF are computed and must reach the output."""

    def test_row_includes_ocr_and_exif(self, tmp_path):
        path = make_png_file(tmp_path / "photo.png")
        exif = {"Make": "TestCam", "Model": "X100"}
        with mock.patch("sick.initialize_nude_detector"), \
                mock.patch("sick.get_dominant_color", return_value=(1, 2, 3)), \
                mock.patch("sick.detect_faces", return_value=False), \
                mock.patch("sick.detect_steganography", return_value=(False, 0.0, None)), \
                mock.patch("sick.detect_age", return_value=None), \
                mock.patch("sick.detect_ai_generated", return_value=(False, 0.1)), \
                mock.patch("sick.detect_potential_child_exploitation", return_value=(False, None)), \
                mock.patch("sick.detect_nudity", return_value={"result": [], "score": 0}), \
                mock.patch("sick.extract_text_from_image", return_value="  hello   stego  world  "), \
                mock.patch("sick.extract_exif_data", return_value=exif):
            row = sick.process_image(str(path))
        assert row is not None
        assert len(row) == 16
        assert "hello stego world" in row       # OCR text, whitespace collapsed
        assert json.dumps(exif) in row          # EXIF surfaced as JSON

    def test_row_marks_missing_ocr_and_exif(self, tmp_path):
        path = make_png_file(tmp_path / "photo.png")
        with mock.patch("sick.initialize_nude_detector"), \
                mock.patch("sick.get_dominant_color", return_value=(1, 2, 3)), \
                mock.patch("sick.detect_faces", return_value=False), \
                mock.patch("sick.detect_steganography", return_value=(False, 0.0, None)), \
                mock.patch("sick.detect_age", return_value=None), \
                mock.patch("sick.detect_ai_generated", return_value=(False, 0.1)), \
                mock.patch("sick.detect_potential_child_exploitation", return_value=(False, None)), \
                mock.patch("sick.detect_nudity", return_value={"result": [], "score": 0}), \
                mock.patch("sick.extract_text_from_image", return_value="   "), \
                mock.patch("sick.extract_exif_data", return_value={}):
            row = sick.process_image(str(path))
        assert row is not None
        assert row[-3:] == ["None detected", "None", row[-1]]


    def test_ocr_failure_does_not_kill_row(self, tmp_path):
        """A missing/failing OCR backend must not drop the whole image row."""
        path = make_png_file(tmp_path / "photo.png")
        with mock.patch("sick.initialize_nude_detector"), \
                mock.patch("sick.get_dominant_color", return_value=(1, 2, 3)), \
                mock.patch("sick.detect_faces", return_value=False), \
                mock.patch("sick.detect_steganography", return_value=(False, 0.0, None)), \
                mock.patch("sick.detect_age", return_value=None), \
                mock.patch("sick.detect_ai_generated", return_value=(False, 0.1)), \
                mock.patch("sick.detect_potential_child_exploitation", return_value=(False, None)), \
                mock.patch("sick.detect_nudity", return_value={"result": [], "score": 0}), \
                mock.patch("sick.extract_text_from_image", side_effect=Exception("tesseract missing")), \
                mock.patch("sick.extract_exif_data", return_value={}):
            row = sick.process_image(str(path))
        assert row is not None
        assert len(row) == 16
        assert row[-3] == "None detected"


class TestDisplayResults:
    """display_results table/CSV output shape."""

    def test_csv_contains_all_rows_and_new_headers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rows = [make_row("http://a.test/1.png"), make_row("http://a.test/2.png")]
        sick.display_results(rows)
        with open("results.csv", newline="", encoding="utf-8") as f:
            data = list(csv.reader(f))
        assert data[0][-3:] == ["Extracted Text (OCR)", "EXIF Data", "Perceptual Hash"]
        assert len(data) == 3  # header + 2 rows
        assert all(len(r) == 16 for r in data)


def gradient_array(height=64, width=256):
    """Return an HxWx3 structured test image (smooth horizontal ramp)."""
    ramp = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    return np.stack([ramp] * 3, axis=-1)


def lsb_embed(arr, rng, fraction=1.0):
    """Sequentially LSB-replace the first `fraction` of samples, like real tools."""
    out = arr.copy()
    flat = out.reshape(-1)
    bits = rng.integers(0, 2, size=flat.size, dtype=np.uint8)
    k = int(flat.size * fraction)
    flat[:k] = (flat[:k] & np.uint8(0xFE)) | bits[:k]
    return out


def png_with_text(path, text_dict):
    """Write a PNG carrying tEXt chunks (like SD/ComfyUI write)."""
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    for key, value in text_dict.items():
        info.add_text(key, value)
    Image.new("RGB", (32, 32), (100, 150, 90)).save(path, pnginfo=info)
    return path


def append_png_chunk(path, chunk_type, data):
    """Append a custom chunk to a PNG after IEND (readers ignore it)."""
    import struct
    import zlib as _zlib
    chunk = struct.pack(">I", len(data)) + chunk_type + data
    chunk += struct.pack(">I", _zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    with open(path, "ab") as f:
        f.write(chunk)


SD_PARAMS = (
    "masterpiece, best quality\n"
    "Negative prompt: blurry, lowres\n"
    "Steps: 28, Sampler: DPM++ 2M Karras, CFG scale: 7, "
    "Seed: 1234567890, Size: 832x1216, Model: animagineXL_v31"
)


class TestAIDetectionMetadata:
    """Generator-metadata provenance scanning (deterministic signals)."""

    def test_flags_stable_diffusion_parameters_chunk(self, tmp_path):
        path = png_with_text(tmp_path / "sd.png", {"parameters": SD_PARAMS})
        indicators = sick.scan_generator_metadata(str(path))
        assert any("Stable Diffusion" in i for i in indicators)
        detected, score = sick.detect_ai_generated(str(path))
        assert detected and score >= 0.9

    def test_flags_comfyui_workflow_chunks(self, tmp_path):
        graph = '{"3": {"class_type": "KSampler", "inputs": {}}}'
        path = png_with_text(tmp_path / "comfy.png",
                             {"prompt": graph, "workflow": graph})
        assert any("ComfyUI" in i for i in sick.scan_generator_metadata(str(path)))

    def test_flags_generator_software_exif(self, tmp_path):
        exif = Image.Exif()
        exif[305] = "Midjourney Bot"  # EXIF Software tag
        path = tmp_path / "mj.jpg"
        Image.new("RGB", (32, 32), (10, 10, 10)).save(path, exif=exif.tobytes())
        assert any("Midjourney" in i for i in sick.scan_generator_metadata(str(path)))

    def test_flags_c2pa_manifest(self, tmp_path):
        path = png_with_text(tmp_path / "c2pa.png", {})
        append_png_chunk(path, b"caBX", b"\x00\x00\x00" + b"c2pa.claim manifest data")
        indicators = sick.scan_generator_metadata(str(path))
        assert any("C2PA" in i for i in indicators)
        detected, score = sick.detect_ai_generated(str(path))
        assert detected and score >= 0.9

    def test_stray_c2pa_bytes_alone_are_not_a_manifest(self, tmp_path):
        """Bare 'c2pa' bytes occur by chance in compressed data; require the box."""
        path = png_with_text(tmp_path / "stray.png", {})
        with open(path, "ab") as f:
            f.write(b"c2pa")
        assert sick.scan_generator_metadata(str(path)) == []

    def test_flags_iptc_trained_algorithmic_media(self, tmp_path):
        """XMP declaring trainedAlgorithmicMedia must be flagged."""
        xmp = (
            '<x:xmpmk xmlns:Iptc4xmpCore="...">'
            '<Iptc4xmpCore:DigitalSourceType>trainedAlgorithmicMedia'
            '</Iptc4xmpCore:DigitalSourceType></x:xmpmk>'
        )
        path = png_with_text(tmp_path / "iptc.png", {"XML:com.adobe.xmp": xmp})
        assert any("trained algorithmic media" in i for i in sick.scan_generator_metadata(str(path)))

    def test_clean_png_has_no_indicators(self, tmp_path):
        path = make_png_file(tmp_path / "clean.png")
        assert sick.scan_generator_metadata(str(path)) == []
        detected, score = sick.detect_ai_generated(str(path))
        assert not detected and score < 0.5


class TestAISpectralAnalysis:
    """Fourier-domain synthesis-artifact heuristics."""

    def test_periodic_spectral_artifact_scores_high(self, tmp_path):
        """A strong periodic spectral peak (the GAN grid-artifact class)."""
        rng = np.random.default_rng(5)
        noise = rng.integers(0, 256, size=(128, 128), dtype=np.uint8).astype(np.float64)
        stripes = 60 * np.sin(2 * np.pi * 24 * np.arange(128) / 128.0)
        synthetic = np.clip(noise + stripes, 0, 255).astype(np.uint8)
        path = tmp_path / "grid.png"
        Image.fromarray(synthetic, "L").save(path)
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
        gan_grid, _ = sick.spectral_ai_analysis(gray)
        assert gan_grid > 0.5

    def test_natural_noise_image_not_flagged(self, tmp_path):
        rng = np.random.default_rng(6)
        noise = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
        path = tmp_path / "noise.png"
        Image.fromarray(noise, "L").save(path)
        detected, score = sick.detect_ai_generated(str(path))
        assert not detected
        assert score < 0.5


def lsb_embed_plane(plane, data, traverse="row", bitorder="big"):
    """LSB-embed `data` into a single 2-D channel plane in a chosen order."""
    out = plane.copy()
    height, width = out.shape
    flat = out.T.reshape(-1) if traverse == "col" else out.reshape(-1)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder=bitorder)
    bits = np.concatenate([bits, np.zeros(len(flat) - len(bits), dtype=np.uint8)])
    flat[:] = (flat & np.uint8(0xFE)) | bits
    return flat.reshape(width, height).T if traverse == "col" else flat.reshape(height, width)


class TestHiddenTextExtraction:
    """Multi-order LSB payload recovery (channels, traversal, bit order)."""

    def test_finds_text_in_single_channel_plane(self):
        arr = gradient_array().copy()
        arr[:, :, 0] = lsb_embed_plane(arr[:, :, 0], b"HIDDENGEM")
        result = sick.detect_hidden_text(arr)
        assert result is not None and "HIDDENGEM" in result

    def test_finds_text_in_column_traversal(self):
        arr = gradient_array().copy()
        arr[:, :, 1] = lsb_embed_plane(arr[:, :, 1], b"VERTICALGEM", traverse="col")
        result = sick.detect_hidden_text(arr)
        assert result is not None and "VERTICALGEM" in result

    def test_finds_text_with_lsb_first_bit_order(self):
        arr = gradient_array().copy()
        arr[:, :, 2] = lsb_embed_plane(arr[:, :, 2], b"LITTLEENDIAN", bitorder="little")
        result = sick.detect_hidden_text(arr)
        assert result is not None and "LITTLEENDIAN" in result

    def test_reports_embedded_archive_signature(self):
        arr = gradient_array().copy()
        payload = b"PK\x03\x04\x14\x00" + b"secret_archive_data" * 4
        arr[:, :, 0] = lsb_embed_plane(arr[:, :, 0], payload)
        result = sick.detect_hidden_text(arr)
        assert result is not None and "ZIP" in result

    def test_reports_bzip2_with_block_magic(self):
        arr = gradient_array().copy()
        payload = b"BZh9\x31\x41\x59\x26\x53\x59" + b"compressed payload data" * 2
        arr[:, :, 1] = lsb_embed_plane(arr[:, :, 1], payload)
        result = sick.detect_hidden_text(arr)
        assert result is not None and "bzip2" in result

    def test_reports_gzip_with_valid_header(self):
        arr = gradient_array().copy()
        payload = b"\x1f\x8b\x08\x00" + b"\x00\x00\x00\x00" + b"\x02\xff" + b"payload bytes here" * 2
        arr[:, :, 2] = lsb_embed_plane(arr[:, :, 2], payload)
        result = sick.detect_hidden_text(arr)
        assert result is not None and "gzip" in result

    def test_finds_bit_plane_2_payload(self):
        """Plane-2 embedding must also be recovered."""
        arr = gradient_array().copy()
        plane = arr[:, :, 0]
        flat = plane.reshape(-1)
        bits = np.unpackbits(np.frombuffer(b"SECONDPLANE", dtype=np.uint8))
        flat[:len(bits)] = (flat[:len(bits)] & np.uint8(0xFD)) | (bits << 1)
        result = sick.detect_hidden_text(arr)
        assert result is not None and "SECONDPLANE" in result

    def test_no_signature_claims_on_noise_images(self):
        """Chance byte matches must never produce 'embedded archive' claims.

        At 1024x1024 the 32 plans scan ~13 MB of candidate bytes; unvalidated
        3-byte magics fire on most such images (verified: 9/10 seeds)."""
        for seed in (0, 1):
            rng = np.random.default_rng(seed)
            noise = rng.integers(0, 256, size=(1024, 1024, 3), dtype=np.uint8)
            result = sick.detect_hidden_text(noise)
            assert result is None or "embedded via LSB" not in result, result

    def test_rgb_interleaved_order_still_found(self):
        """The original interleaved-RGB order must remain in the catalog."""
        arr = gradient_array().copy()
        bits = np.unpackbits(np.frombuffer(b"INTERLEAVED", dtype=np.uint8))
        padded = np.concatenate([bits, np.zeros(3 - len(bits) % 3, dtype=np.uint8)])
        triplets = padded.reshape(-1, 3)
        arr = arr.reshape(-1, 3)
        arr[:len(triplets), 0] = (arr[:len(triplets), 0] & np.uint8(0xFE)) | triplets[:, 0]
        arr[:len(triplets), 1] = (arr[:len(triplets), 1] & np.uint8(0xFE)) | triplets[:, 1]
        arr[:len(triplets), 2] = (arr[:len(triplets), 2] & np.uint8(0xFE)) | triplets[:, 2]
        arr = arr.reshape(64, 256, 3)
        result = sick.detect_hidden_text(arr)
        assert result is not None and "INTERLEAVED" in result

    def test_clean_image_finds_nothing(self):
        assert sick.detect_hidden_text(gradient_array()) is None


def make_gif(path, n_frames=3, size=(24, 24), colors=((255, 0, 0), (0, 255, 0), (0, 0, 255))):
    """Write a deterministic animated GIF."""
    frames = [Image.new("RGB", size, colors[i % len(colors)]) for i in range(n_frames)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return path


class TestGifAnalysis:
    """GIFs must be analyzed per-frame instead of being skipped."""

    def test_extract_gif_frames_returns_png_frames(self, tmp_path):
        gif = make_gif(tmp_path / "anim.gif", n_frames=3)
        frames = sick.extract_gif_frames(str(gif), max_frames=4)
        try:
            assert 1 <= len(frames) <= 4
            assert len(frames) == 3  # 3-frame GIF, max 4 -> all frames
            for frame_path in frames:
                assert frame_path.endswith(".png")
                with Image.open(frame_path) as img:
                    assert img.format == "PNG" and img.size == (24, 24)
        finally:
            for frame_path in frames:
                Path(frame_path).unlink(missing_ok=True)

    def test_extract_gif_frames_spreads_long_animations(self, tmp_path):
        """A 40-frame GIF with max_frames=4 must sample across the animation."""
        gif = make_gif(tmp_path / "long.gif", n_frames=40, size=(8, 8))
        frames = sick.extract_gif_frames(str(gif), max_frames=4)
        try:
            assert len(frames) == 4
        finally:
            for frame_path in frames:
                Path(frame_path).unlink(missing_ok=True)

    def test_gif_nudity_runs_on_every_frame_and_reports_worst(self, tmp_path, monkeypatch):
        gif = make_gif(tmp_path / "anim.gif", n_frames=3)
        monkeypatch.setattr(sick.tempfile, "tempdir", str(tmp_path / "tmp"))
        (tmp_path / "tmp").mkdir()
        scores = iter([{"result": [], "score": 0.2},
                       {"result": [{"class": "EXPOSED_BREAST", "score": 0.9}], "score": 0.9},
                       {"result": [], "score": 0.1}])
        nudity = mock.MagicMock(side_effect=lambda p: next(scores))
        with mock.patch("sick.initialize_nude_detector"), \
                mock.patch("sick.detect_nudity", nudity), \
                mock.patch("sick.detect_faces", return_value=False), \
                mock.patch("sick.detect_age", return_value=None), \
                mock.patch("sick.get_dominant_color", return_value=(1, 2, 3)), \
                mock.patch("sick.detect_potential_child_exploitation", return_value=(False, None)):
            row = sick.process_image(str(gif))
        assert row is not None
        assert nudity.call_count == 3
        assert all(call.args[0].endswith(".png") for call in nudity.call_args_list)
        assert "0.90" in row[4] and "EXPOSED_BREAST" in row[4]

    def test_gif_frames_are_cleaned_up(self, tmp_path, monkeypatch):
        gif = make_gif(tmp_path / "anim.gif", n_frames=3)
        tmpdir = tmp_path / "tmp"
        tmpdir.mkdir()
        monkeypatch.setattr(sick.tempfile, "tempdir", str(tmpdir))
        with mock.patch("sick.initialize_nude_detector"), \
                mock.patch("sick.detect_nudity", return_value={"result": [], "score": 0}), \
                mock.patch("sick.detect_faces", return_value=False), \
                mock.patch("sick.get_dominant_color", return_value=(1, 2, 3)), \
                mock.patch("sick.detect_potential_child_exploitation", return_value=(False, None)):
            row = sick.process_image(str(gif))
        assert row is not None
        assert list(tmpdir.glob("*.png")) == []  # frames removed

    def test_gif_reports_frame_count_and_duration(self, tmp_path, monkeypatch):
        gif = make_gif(tmp_path / "anim.gif", n_frames=3)
        tmpdir = tmp_path / "tmp"
        tmpdir.mkdir()
        monkeypatch.setattr(sick.tempfile, "tempdir", str(tmpdir))
        with mock.patch("sick.initialize_nude_detector"), \
                mock.patch("sick.detect_nudity", return_value={"result": [], "score": 0}), \
                mock.patch("sick.detect_faces", return_value=False), \
                mock.patch("sick.get_dominant_color", return_value=(1, 2, 3)), \
                mock.patch("sick.detect_potential_child_exploitation", return_value=(False, None)):
            row = sick.process_image(str(gif))
        assert row is not None
        assert row[8] == "3"     # Frame Count
        assert row[9] == "100"   # Duration (ms)


    def test_gif_sampling_covers_animation_tail(self, tmp_path):
        """Even sampling must include the LAST frame, not just the head."""
        grays = [Image.new("L", (8, 8), i * 30) for i in range(8)]
        gif = tmp_path / "gradual.gif"
        grays[0].save(gif, save_all=True, append_images=grays[1:], duration=100, loop=0)
        frames = sick.extract_gif_frames(str(gif), max_frames=4)
        try:
            sampled = []
            for frame_path in frames:
                with Image.open(frame_path) as img:
                    sampled.append(img.getpixel((0, 0))[0])
            assert sampled[0] == 0          # first frame
            assert sampled[-1] == 7 * 30    # last frame analyzed
        finally:
            for frame_path in frames:
                Path(frame_path).unlink(missing_ok=True)

    def test_extract_gif_frames_cleans_up_on_failure(self, tmp_path, monkeypatch):
        """A corrupt GIF must not leak decoded frame PNGs into the temp dir."""
        full = make_gif(tmp_path / "anim.gif", n_frames=8)
        data = full.read_bytes()
        truncated = tmp_path / "truncated.gif"
        truncated.write_bytes(data[:int(len(data) * 0.5)])
        tmpdir = tmp_path / "tmp"
        tmpdir.mkdir()
        monkeypatch.setattr(sick.tempfile, "tempdir", str(tmpdir))
        with pytest.raises(Exception):
            sick.extract_gif_frames(str(truncated))
        assert list(tmpdir.glob("*.png")) == []

    def test_truncated_gif_still_returns_row_without_leaking(self, tmp_path, monkeypatch):
        """Frame-decode failure must degrade the row, not drop it or leak files."""
        full = make_gif(tmp_path / "anim.gif", n_frames=8)
        data = full.read_bytes()
        truncated = tmp_path / "truncated.gif"
        truncated.write_bytes(data[:int(len(data) * 0.5)])
        tmpdir = tmp_path / "tmp"
        tmpdir.mkdir()
        monkeypatch.setattr(sick.tempfile, "tempdir", str(tmpdir))
        with mock.patch("sick.initialize_nude_detector"), \
                mock.patch("sick.detect_nudity", return_value={"result": [], "score": 0}), \
                mock.patch("sick.detect_faces", return_value=False), \
                mock.patch("sick.detect_age", return_value=None), \
                mock.patch("sick.get_dominant_color", return_value=(1, 2, 3)), \
                mock.patch("sick.detect_potential_child_exploitation", return_value=(False, None)):
            row = sick.process_image(str(truncated))
        assert row is not None
        assert "extraction failed" in row[4]
        assert row[8] == "N/A"  # frame count unknowable on corrupt file
        assert list(tmpdir.glob("*.png")) == []

    def test_gif_child_exploitation_uses_worst_frame_nudity(self, tmp_path, monkeypatch):
        """The CSAM screen must consult the worst frame, not re-run frame 0."""
        gif = make_gif(tmp_path / "anim.gif", n_frames=3)
        tmpdir = tmp_path / "tmp"
        tmpdir.mkdir()
        monkeypatch.setattr(sick.tempfile, "tempdir", str(tmpdir))
        scores = iter([{"result": [], "score": 0.1},
                       {"result": [], "score": 0.95},
                       {"result": [], "score": 0.1}])
        nudity = mock.MagicMock(side_effect=lambda p: next(scores))
        with mock.patch("sick.initialize_nude_detector"), \
                mock.patch("sick.detect_nudity", nudity), \
                mock.patch("sick.detect_faces", return_value=True), \
                mock.patch("sick.detect_age", return_value=15), \
                mock.patch("sick.get_dominant_color", return_value=(1, 2, 3)):
            row = sick.process_image(str(gif))
        assert row is not None
        assert row[11] == "Yes"                     # Child Exploitation column
        assert "0.95" in row[12]                    # worst-frame score cited


    def test_extract_gif_frames_max_one_samples_middle_frame(self, tmp_path):
        """A single-frame sample takes the middle of the animation."""
        grays = [Image.new("L", (8, 8), i * 30) for i in range(8)]
        gif = tmp_path / "gradual.gif"
        grays[0].save(gif, save_all=True, append_images=grays[1:], duration=100, loop=0)
        frames = sick.extract_gif_frames(str(gif), max_frames=1)
        try:
            assert len(frames) == 1
            with Image.open(frames[0]) as img:
                assert img.getpixel((0, 0))[0] == 3 * 30  # middle frame index 3
        finally:
            for frame_path in frames:
                Path(frame_path).unlink(missing_ok=True)

    def test_extract_gif_frames_rejects_invalid_max_frames(self, tmp_path):
        gif = make_gif(tmp_path / "anim.gif", n_frames=3)
        with pytest.raises(ValueError):
            sick.extract_gif_frames(str(gif), max_frames=0)


class TestCsvSafety:
    """results.csv must be safe to open in spreadsheet applications."""

    def test_neutralizes_formula_cells(self):
        assert sick.sanitize_csv_cell("=1+1") == "'=1+1"
        assert sick.sanitize_csv_cell("+SUM(A1:A2)") == "'+SUM(A1:A2)"
        assert sick.sanitize_csv_cell("-2+cmd") == "'-2+cmd"
        assert sick.sanitize_csv_cell("@SUM(A1)") == "'@SUM(A1)"
        assert sick.sanitize_csv_cell("\t=cmd") == "'\t=cmd"

    def test_leaves_normal_cells_alone(self):
        assert sick.sanitize_csv_cell("No (Score: 0.00)") == "No (Score: 0.00)"
        assert sick.sanitize_csv_cell("http://site.test/img.png") == "http://site.test/img.png"
        assert sick.sanitize_csv_cell("") == ""
        assert sick.sanitize_csv_cell(None) is None

    def test_written_csv_has_no_executable_cells(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rows = [make_row("=cmd|' /C calc'!A0")]
        rows[0][7] = "+HYPERLINK(\"http://evil.test\")"  # Hidden Text column
        sick.display_results(rows)
        with open("results.csv", newline="", encoding="utf-8") as f:
            written = list(csv.reader(f))
        for cell in written[1]:
            assert not cell.startswith(("=", "+", "-", "@", "\t", "\r")), cell

    def test_output_creates_missing_directories(self, tmp_path, monkeypatch):
        """-o into a nonexistent directory must not lose the report."""
        monkeypatch.chdir(tmp_path)
        rows = [make_row("http://a.test/1.png")]
        sick.display_results(rows, str(tmp_path / "nested" / "dirs" / "report.csv"))
        assert (tmp_path / "nested" / "dirs" / "report.csv").exists()


class TestOutputOption:
    """-o/--output must control where the CSV report lands."""

    def test_output_flag_writes_custom_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        folder = tmp_path / "images"
        folder.mkdir()
        rows = [make_row("http://a.test/1.png")]
        monkeypatch.setattr(
            sys, "argv", ["sick.py", "-f", str(folder), "-o", str(tmp_path / "report.csv")])
        with mock.patch("sick.process_local_folder", return_value=rows):
            sick.main()
        assert (tmp_path / "report.csv").exists()
        assert not (tmp_path / "results.csv").exists()

    def test_default_output_is_results_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        folder = tmp_path / "images"
        folder.mkdir()
        rows = [make_row("http://a.test/1.png")]
        monkeypatch.setattr(sys, "argv", ["sick.py", "-f", str(folder)])
        with mock.patch("sick.process_local_folder", return_value=rows):
            sick.main()
        assert (tmp_path / "results.csv").exists()


def make_poster(size=(64, 64)):
    """Structured test image (text-like bars) with real detail."""
    from PIL import ImageDraw
    img = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(img)
    for y in range(4, size[1] - 4, 8):
        draw.line([(6, y), (size[0] - 6, y)], fill=(20, 20, 20), width=2)
    draw.rectangle([2, 2, size[0] - 3, size[1] - 3], outline=(200, 0, 0))
    return img


@pytest.fixture(autouse=True)
def reset_duplicate_registry():
    """Keep near-duplicate registry state from leaking between tests."""
    sick.reset_duplicate_registry()
    yield
    sick.reset_duplicate_registry()


class TestPerceptualHash:
    """64-bit dHash generation and near-duplicate detection."""

    def test_hamming_distance_basics(self):
        zeros = "0" * 16
        assert sick.hamming_distance(zeros, zeros) == 0
        assert sick.hamming_distance(zeros, "f" * 16) == 64
        assert sick.hamming_distance(zeros, "0" * 15 + "1") == 1

    def test_perceptual_hash_is_stable_16_hex(self, tmp_path):
        path = make_png_file(tmp_path / "a.png")
        with Image.open(path) as first, Image.open(path) as second:
            hash_a, hash_b = sick.perceptual_hash(first), sick.perceptual_hash(second)
        assert hash_a == hash_b
        assert len(hash_a) == 16
        int(hash_a, 16)  # parses as hex

    def test_perceptual_hash_survives_resize_and_brightness(self):
        """Resized + brightened copies must stay within the dup threshold."""
        base = make_poster()
        bright = Image.eval(base.convert("L"), lambda v: min(255, int(v * 1.3))).convert("RGB")
        big = bright.resize((128, 128))
        distance = sick.hamming_distance(sick.perceptual_hash(base), sick.perceptual_hash(big))
        assert distance <= 10

    def test_perceptual_hash_separates_different_images(self):
        rng = np.random.default_rng(1)
        noise = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")
        distance = sick.hamming_distance(
            sick.perceptual_hash(make_poster()), sick.perceptual_hash(noise))
        assert distance > 20

    def _process(self, tmp_path, img, name):
        path = tmp_path / name
        img.save(path)
        with mock.patch("sick.initialize_nude_detector"), \
                mock.patch("sick.get_dominant_color", return_value=(1, 2, 3)), \
                mock.patch("sick.detect_faces", return_value=False), \
                mock.patch("sick.detect_steganography", return_value=(False, 0.0, None)), \
                mock.patch("sick.detect_age", return_value=None), \
                mock.patch("sick.detect_ai_generated", return_value=(False, 0.1)), \
                mock.patch("sick.detect_potential_child_exploitation", return_value=(False, None)), \
                mock.patch("sick.detect_nudity", return_value={"result": [], "score": 0}), \
                mock.patch("sick.extract_text_from_image", return_value=""), \
                mock.patch("sick.extract_exif_data", return_value={}):
            return sick.process_image(str(path))

    def test_near_duplicate_flagged_in_row(self, tmp_path):
        first = self._process(tmp_path, make_poster(), "one.png")
        resized = make_poster().resize((128, 128))
        second = self._process(tmp_path, resized, "two.png")
        assert first is not None and second is not None
        assert len(first) == 16 and len(second) == 16
        assert "near-duplicate" not in first[-1]
        assert "near-duplicate" in second[-1]
        assert "one.png" in second[-1]

    def test_distinct_images_not_flagged(self, tmp_path):
        rng = np.random.default_rng(2)
        noise = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")
        first = self._process(tmp_path, make_poster(), "poster.png")
        second = self._process(tmp_path, noise, "noise.png")
        assert first is not None and second is not None
        assert "near-duplicate" not in first[-1]
        assert "near-duplicate" not in second[-1]


    def test_solid_images_not_flagged_as_duplicates(self, tmp_path):
        """Flat images hash identically; that must not read as 'duplicate'."""
        red = self._process(tmp_path, Image.new("RGB", (32, 32), (255, 0, 0)), "red.png")
        blue = self._process(tmp_path, Image.new("RGB", (48, 48), (0, 0, 255)), "blue.png")
        assert red is not None and blue is not None
        assert "near-duplicate" not in red[-1]
        assert "near-duplicate" not in blue[-1]

    def test_scan_entry_points_reset_registry(self, tmp_path):
        """crawl() and process_local_folder() start from a fresh registry."""
        poster_hash = sick.perceptual_hash(make_poster())
        sick.find_or_register_duplicate(poster_hash, "stale-entry")
        folder = tmp_path / "images"
        folder.mkdir()
        pages = {"http://site.test/a": "<html></html>"}
        get = fake_requests_get(pages)
        with mock.patch("sick.requests.get", get), \
                mock.patch("sick.process_image", return_value=None):
            sick.crawl("http://site.test/a", depth=1)
        # without a reset the stale entry would match at distance 0
        assert sick.find_or_register_duplicate(poster_hash, "probe") is None
        sick.find_or_register_duplicate(poster_hash, "stale-again")
        with mock.patch("sick.process_image", return_value=None):
            sick.process_local_folder(str(folder))
        assert sick.find_or_register_duplicate(poster_hash, "probe2") is None


def jpeg_with_gps(path, lat_ref="N", lon_ref="W", altitude_ref=None, makernote=None):
    """Write a JPEG carrying GPS coordinates and an Exif IFD."""
    from PIL.TiffImagePlugin import IFDRational

    def dms(parts):
        return tuple(tuple(IFDRational(n, d) for n, d in parts))

    exif = Image.Exif()
    gps = {
        1: lat_ref,
        2: dms(((51, 1), (30, 1), (741, 100))),
        3: lon_ref,
        4: dms(((0, 1), (12, 1), (0, 1))),
        6: IFDRational(25, 2),
    }
    if altitude_ref is not None:
        gps[5] = altitude_ref
    exif[34853] = gps
    sub = exif.get_ifd(0x8769)
    sub[36867] = "2024:01:01 10:00:00"  # DateTimeOriginal
    if makernote:
        sub[37500] = makernote
    Image.new("RGB", (16, 16), (10, 20, 30)).save(path, "JPEG", exif=exif.tobytes())
    return path


class TestExifExtraction:
    """EXIF decoding: GPS coordinates and Exif-IFD tags must surface."""

    def test_gps_decoded_to_decimal_degrees(self, tmp_path):
        path = jpeg_with_gps(tmp_path / "geo.jpg")
        with Image.open(path) as img:
            data = sick.extract_exif_data(img)
        assert "GPSPosition" in data
        assert list(data)[0] == "GPSPosition"  # first: survives 300-char truncation
        lat_text, lon_text = data["GPSPosition"].split(", ")
        assert abs(float(lat_text) - 51.502058) < 1e-4  # 51 deg 30' 7.41"
        assert abs(float(lon_text) - -0.2) < 1e-4       # 0 deg 12' W
        assert data.get("GPSAltitude") == "12.5m"

    def test_gps_south_east_refs(self, tmp_path):
        path = jpeg_with_gps(tmp_path / "geo.jpg", lat_ref="S", lon_ref="E")
        with Image.open(path) as img:
            data = sick.extract_exif_data(img)
        lat_text, lon_text = data["GPSPosition"].split(", ")
        assert float(lat_text) < 0 < float(lon_text)

    def test_exif_ifd_tags_surfaced(self, tmp_path):
        path = jpeg_with_gps(tmp_path / "geo.jpg")
        with Image.open(path) as img:
            data = sick.extract_exif_data(img)
        assert data.get("DateTimeOriginal") == "2024:01:01 10:00:00"
        assert "GPSInfo" not in data and "ExifOffset" not in data  # no raw pointers

    def test_malformed_gps_returns_no_position(self):
        # pylint: disable=protected-access; unit-testing the decoder directly
        assert sick._decode_gps({1: "N", 2: "garbage", 3: "W", 4: None}) == {}

    def test_altitude_ref_bytes_do_not_kill_position(self, tmp_path):
        """Real Pillow round-trips GPSAltitudeRef as BYTES; position must survive."""
        path = jpeg_with_gps(tmp_path / "geo.jpg", altitude_ref=1)
        with Image.open(path) as img:
            data = sick.extract_exif_data(img)
        assert "GPSPosition" in data
        assert data.get("GPSAltitude") == "-12.5m"  # below sea level

    def test_bad_altitude_keeps_position(self):
        # pylint: disable=protected-access; unit-testing the decoder directly
        decoded = sick._decode_gps({
            1: "N", 2: ((51, 1), (30, 1), (0, 1)),
            3: "W", 4: ((0, 1), (12, 1), (0, 1)),
            6: "garbage",
        })
        assert "GPSPosition" in decoded
        assert "GPSAltitude" not in decoded

    def test_bytes_hemisphere_ref(self):
        # pylint: disable=protected-access; unit-testing the decoder directly
        decoded = sick._decode_gps({
            1: b"S", 2: ((10, 1), (0, 1), (0, 1)),
            3: b"E", 4: ((20, 1), (0, 1), (0, 1)),
        })
        assert decoded["GPSPosition"] == "-10.000000, 20.000000"

    def test_unrecognizable_hemisphere_fails_closed(self):
        # pylint: disable=protected-access; wrong-hemisphere is worse than none
        assert sick._decode_gps({
            1: "X", 2: ((10, 1), (0, 1), (0, 1)),
            3: "E", 4: ((20, 1), (0, 1), (0, 1)),
        }) == {}

    def test_makernote_in_subifd_is_abbreviated(self, tmp_path):
        path = jpeg_with_gps(tmp_path / "geo.jpg", makernote="M" * 800)
        with Image.open(path) as img:
            data = sick.extract_exif_data(img)
        makernote_value = str(data.get("MakerNote", ""))
        assert "MMMM" not in makernote_value or len(makernote_value) <= 40

    def test_zero_coordinates_render_without_negative_zero(self):
        # pylint: disable=protected-access; unit-testing the decoder directly
        decoded = sick._decode_gps({
            1: "S", 2: ((0, 1), (0, 1), (0, 1)),
            3: "W", 4: ((0, 1), (0, 1), (0, 1)),
        })
        assert decoded["GPSPosition"] == "0.000000, 0.000000"

    def test_nul_padded_ref_still_recognized(self):
        # pylint: disable=protected-access; unit-testing the decoder directly
        decoded = sick._decode_gps({
            1: b"N\x00", 2: ((10, 1), (0, 1), (0, 1)),
            3: b"E\x00", 4: ((20, 1), (0, 1), (0, 1)),
        })
        assert decoded["GPSPosition"] == "10.000000, 20.000000"

    def test_row_contains_decoded_gps(self, tmp_path):
        path = jpeg_with_gps(tmp_path / "geo.jpg")
        with mock.patch("sick.initialize_nude_detector"), \
                mock.patch("sick.get_dominant_color", return_value=(1, 2, 3)), \
                mock.patch("sick.detect_faces", return_value=False), \
                mock.patch("sick.detect_steganography", return_value=(False, 0.0, None)), \
                mock.patch("sick.detect_age", return_value=None), \
                mock.patch("sick.detect_ai_generated", return_value=(False, 0.1)), \
                mock.patch("sick.detect_potential_child_exploitation", return_value=(False, None)), \
                mock.patch("sick.detect_nudity", return_value={"result": [], "score": 0}), \
                mock.patch("sick.extract_text_from_image", return_value=""):
            row = sick.process_image(str(path))
        assert row is not None
        assert "GPSPosition" in row[-2]


def small_watermark_image():
    """Small, low-contrast text (like a watermark) on a ~92x26 image."""
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (92, 26), (225, 225, 225))
    draw = ImageDraw.Draw(img)
    draw.text((3, 8), "SECRET MARKER 42", fill=(140, 140, 140),
              font=ImageFont.load_default(size=9))
    return img


class TestOcrExtraction:
    """OCR preprocessing and word-confidence gating."""

    def test_preprocess_upscales_small_images(self):
        processed = sick.preprocess_for_ocr(small_watermark_image())
        assert processed.mode == "L"
        assert min(processed.size) >= 64          # 26 -> 78 via 3x upscale
        big = sick.preprocess_for_ocr(Image.new("RGB", (300, 300), (10, 10, 10)))
        assert big.size == (300, 300)             # large images untouched

    def test_ocr_reads_small_low_contrast_text(self):
        """The watermark fixture the unpreprocessed pipeline misses."""
        text = sick.extract_text_from_image(small_watermark_image())
        assert "SECRET" in text and "MARKER" in text

    def test_ocr_confidence_gate_filters_junk(self):
        """Only words with confidence >= 60 survive; blanks and -1 rows drop."""
        fake_data = {
            "text": ["GOOD", "BAD", "  ", "SKIPROW"],
            "conf": [90, 30, 95, -1],
        }
        with mock.patch("sick.pytesseract.image_to_data", return_value=fake_data):
            text = sick.extract_text_from_image(small_watermark_image())
        assert text == "GOOD"


class TestImageHelpers:
    """Characterization tests for the pure image-analysis helpers."""

    def test_get_image_hash_is_stable(self, tmp_path):
        path = make_png_file(tmp_path / "img.png")
        with Image.open(path) as first, Image.open(path) as second:
            hash_a, hash_b = sick.get_image_hash(first), sick.get_image_hash(second)
        assert hash_a == hash_b
        assert len(hash_a) == 32  # md5 hex digest

    def test_extract_exif_data_empty_for_plain_png(self, tmp_path):
        path = make_png_file(tmp_path / "img.png")
        with Image.open(path) as img:
            assert sick.extract_exif_data(img) == {}

    def test_rs_analysis_returns_payload_estimate(self):
        """RS analysis must return a float payload estimate in [0, 1]."""
        rng = np.random.default_rng(42)
        clean = gradient_array()[:, :, 0]
        full = lsb_embed(gradient_array(), rng, 1.0)[:, :, 0]
        half = lsb_embed(gradient_array(), rng, 0.5)[:, :, 0]
        for channel in (clean, full, half):
            assert isinstance(sick.rs_analysis(channel), float)
            assert 0.0 <= sick.rs_analysis(channel) <= 1.0
        # must not fire on clean content (RS is unreliable at very high
        # payloads, so only the clean side is asserted -- see Ker 2008)
        assert sick.rs_analysis(clean) < 0.1

    def test_ws_analysis_estimates_payload(self):
        """WS (Ker & Bohme 2008) must quantitatively estimate LSB payload."""
        rng = np.random.default_rng(42)
        clean = gradient_array()[:, :, 0]
        full = lsb_embed(gradient_array(), rng, 1.0)[:, :, 0]
        half = lsb_embed(gradient_array(), rng, 0.5)[:, :, 0]
        assert sick.ws_analysis(clean) < 0.1
        assert sick.ws_analysis(full) > 0.8
        assert 0.25 <= sick.ws_analysis(half) <= 0.75

    def test_rs_analysis_handles_degenerate_flat_channel(self):
        flat = np.full((64, 64), 128, dtype=np.uint8)
        assert sick.rs_analysis(flat) == 0.0

    def test_lsb_plane_compression_ratio_separates_structured_and_random(self):
        rng = np.random.default_rng(42)
        structured = sick.lsb_plane_compression_ratio(gradient_array())
        random_plane = sick.lsb_plane_compression_ratio(lsb_embed(gradient_array(), rng, 1.0))
        assert structured < 0.5
        assert random_plane > 0.9

    def test_detect_steganography_flags_fully_embedded_image(self, tmp_path):
        """A fully LSB-embedded image must be flagged as steganographic."""
        arr = lsb_embed(gradient_array(), np.random.default_rng(7), 1.0)
        path = tmp_path / "stego.png"
        Image.fromarray(arr, "RGB").save(path)
        detected, score, _ = sick.detect_steganography(str(path))
        assert detected
        assert score >= 0.5

    def test_detect_steganography_clean_gradient_not_flagged(self, tmp_path):
        path = tmp_path / "clean.png"
        Image.fromarray(gradient_array(), "RGB").save(path)
        detected, score, _ = sick.detect_steganography(str(path))
        assert not detected
        assert score < 0.5

    def test_detect_steganography_score_is_bounded(self, tmp_path):
        rng = np.random.default_rng(3)
        noise = rng.integers(0, 256, size=(64, 256, 3), dtype=np.uint8)
        path = tmp_path / "noise.png"
        Image.fromarray(noise, "RGB").save(path)
        detected, score, _ = sick.detect_steganography(str(path))
        assert isinstance(detected, (bool, np.bool_))
        assert 0.0 <= score <= 1.0

    def test_detect_hidden_text_finds_embedded_word(self):
        """LSB-embed a phrase exactly the way detect_hidden_text reads it."""
        data = b"TESTPAYLOAD"
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        padded = np.concatenate([bits, np.zeros(3 - len(bits) % 3, dtype=np.uint8)])
        triplets = padded.reshape(-1, 3)
        np_img = np.zeros((len(triplets), 1, 3), dtype=np.uint8)
        np_img[:, 0, 0] = triplets[:, 0]
        np_img[:, 0, 1] = triplets[:, 1]
        np_img[:, 0, 2] = triplets[:, 2]
        result = sick.detect_hidden_text(np_img)
        assert result is not None
        assert "TESTPAYLOAD" in result

    def test_download_image_accepts_valid_data_url(self):
        png = make_png_bytes()
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        path = sick.download_image(url)
        try:
            assert path is not None
            assert path.endswith(".png")
            assert Path(path).read_bytes() == png
        finally:
            if path:
                Path(path).unlink(missing_ok=True)

    def test_download_image_rejects_non_image_data_url(self):
        url = "data:text/plain;base64," + base64.b64encode(b"hello").decode()
        assert sick.download_image(url) is None

    def test_process_image_rejects_non_image_file(self, tmp_path):
        """A non-image file must be rejected without loading any ML models."""
        path = tmp_path / "fake.png"
        path.write_text("this is not an image")
        with mock.patch("sick.initialize_nude_detector") as init:
            assert sick.process_image(str(path)) is None
        init.assert_called_once()

    def test_detect_steganography_clean_solid_image(self, tmp_path):
        path = make_png_file(tmp_path / "clean.png", color=(128, 128, 128))
        detected, score, hidden_text = sick.detect_steganography(str(path))
        assert not detected
        assert score < 0.6
        assert hidden_text is None

    def test_detect_ai_generated_returns_bounded_score(self, tmp_path):
        path = make_png_file(tmp_path / "grad.png")
        is_ai, score = sick.detect_ai_generated(str(path))
        assert isinstance(is_ai, (bool, np.bool_))
        assert 0.0 <= score <= 1.0

    def test_get_dominant_color_of_solid_red(self, tmp_path):
        path = make_png_file(tmp_path / "red.png", color=(255, 0, 0))
        color = sick.get_dominant_color(str(path))
        assert color is not None
        red, green, blue = color
        assert red > 200 and green < 50 and blue < 50

    def test_detect_faces_on_solid_image(self):
        img = Image.new("RGB", (64, 64), (0, 128, 255))
        assert sick.detect_faces(img) is False

    @pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not installed")
    def test_extract_text_from_image_returns_string(self, tmp_path):
        path = make_png_file(tmp_path / "blank.png")
        with Image.open(path) as img:
            assert isinstance(sick.extract_text_from_image(img), str)
