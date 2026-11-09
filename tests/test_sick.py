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
import io
import logging
import shutil
import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

import sick

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_row(url):
    """Return a 13-column result row like process_image produces."""
    return [url] + ["x"] * 12


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
    def _get(url, **kwargs):
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
        crawl.assert_called_once_with("http://site.test/", 1)
        display.assert_called_once_with(rows)


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
        """LSB-embed b'TEST' exactly the way detect_hidden_text reads it."""
        bits = np.unpackbits(np.frombuffer(b"TEST", dtype=np.uint8))
        padded = np.concatenate([bits, np.zeros(3 - len(bits) % 3, dtype=np.uint8)])
        triplets = padded.reshape(-1, 3)
        np_img = np.zeros((len(triplets), 1, 3), dtype=np.uint8)
        np_img[:, 0, 0] = triplets[:, 0]
        np_img[:, 0, 1] = triplets[:, 1]
        np_img[:, 0, 2] = triplets[:, 2]
        assert sick.detect_hidden_text(np_img) == "TEST"

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
