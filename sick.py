"""SiCK -- Suspicious Image Collection Kit: triage images from the web or
local folders for steganography, AI generation, nudity and forensic metadata."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from PIL import ExifTags, UnidentifiedImageError, ImageOps
from fake_useragent import UserAgent
from rich.progress import Progress
from colorthief import ColorThief
from nudenet import NudeDetector
from urllib.parse import urljoin, urlparse
import io
from rich.console import Console
from  bs4 import BeautifulSoup
from rich.table import Table
from deepface import DeepFace
import pytesseract
from pytesseract import Output
import numpy as np
import tempfile
import requests
import argparse
import hashlib
import re
import zlib
import threading
import magic
from PIL import Image
import json
import cv2
import csv
import os
import logging
import base64
from pathlib import Path

# Initialize Globals
console = Console()
user_agent = UserAgent()
nude_detector = None
_nude_detector_lock = threading.Lock()

def initialize_nude_detector():
    global nude_detector
    if nude_detector is None:
        with _nude_detector_lock:
            if nude_detector is None:
                logger.info("Initializing NudeDetector. This may take a moment...")
                nude_detector = NudeDetector()
                logger.info("NudeDetector initialized successfully.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mask used by RS analysis: which pixels of each 4-pixel group get flipped
RS_MASK = np.array([1, 0, 0, 1], dtype=bool)


def _flip_lsb(values):
    """F1: the LSB flipping permutation 2i <-> 2i+1."""
    return values ^ 1


def _flip_lsb_negative(values):
    """F-1: the other LSB pairing 2i-1 <-> 2i."""
    v = values.astype(np.int16)
    return np.clip(np.where(v % 2 == 0, v - 1, v + 1), 0, 255)


def rs_analysis(channel):
    """Estimate the LSB-replacement payload rate with RS analysis.

    Follows Fridrich, Goljan & Du, "Reliable Detection of LSB Steganography
    in Color and Gray-Scale Images" (IEEE Multimedia, 2001): classify groups
    of 4 adjacent pixels as regular/singular for a mask and its dual, repeat
    on the fully-LSB-flipped channel, and solve the RS quadratic for the
    crossing point x; the payload estimate is |x / (x - 0.5)|.  The estimator
    is known to become unreliable as the rate approaches 1 (the curves
    converge), so treat very high readings with caution.

    Returns a float in [0, 1]; degenerate inputs yield 0.0.
    """
    channel = np.asarray(channel, dtype=np.uint8)
    if channel.size < 8 or float(channel.max()) == float(channel.min()):
        return 0.0  # degenerate: too small or perfectly flat

    def regular_singular_counts(groups):
        """Return (R, S) fractions for F1 and (R-, S-) for F-1 under RS_MASK."""
        base = np.abs(np.diff(groups.astype(np.int16), axis=1)).sum(axis=1)
        pos = groups.copy()
        pos[:, RS_MASK] = _flip_lsb(pos[:, RS_MASK])
        neg = groups.copy()
        neg[:, RS_MASK] = _flip_lsb_negative(neg[:, RS_MASK])
        var_pos = np.abs(np.diff(pos.astype(np.int16), axis=1)).sum(axis=1)
        var_neg = np.abs(np.diff(neg.astype(np.int16), axis=1)).sum(axis=1)
        return (np.mean(var_pos > base), np.mean(var_pos < base),
                np.mean(var_neg > base), np.mean(var_neg < base))

    n_groups = channel.size // 4
    groups = channel.reshape(-1, 4)[:n_groups]
    r, s, r_neg, s_neg = regular_singular_counts(groups)
    r_f, s_f, r_neg_f, s_neg_f = regular_singular_counts(_flip_lsb(groups))

    d0 = r - s                 # R_M(p/2) - S_M(p/2)
    d_neg0 = r_neg - s_neg     # R_-M(p/2) - S_-M(p/2)
    d1 = r_f - s_f             # R_M(1-p/2) - S_M(1-p/2)
    d_neg1 = r_neg_f - s_neg_f # R_-M(1-p/2) - S_-M(1-p/2)

    # 2(d1 + d0)x^2 + (d-0 - d-1 - d1 - 3d0)x + (d0 - d-0) = 0
    a = 2.0 * (d1 + d0)
    b = d_neg0 - d_neg1 - d1 - 3.0 * d0
    c = d0 - d_neg0

    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return 0.0
        x = -c / b
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return 0.0
        root = np.sqrt(discriminant)
        candidates = ((-b + root) / (2 * a), (-b - root) / (2 * a))
        x = min(candidates, key=abs)  # smallest-magnitude root, per the paper
    if abs(x - 0.5) < 1e-12:
        return 1.0
    rate = abs(x / (x - 0.5))
    return float(np.clip(rate, 0.0, 1.0))


def ws_analysis(channel):
    """Estimate the LSB-replacement payload rate with WS (weighted stego).

    Follows Fridrich's WS method as revised by Ker & Bohme, "Revisiting
    Weighted Stego-Image Steganalysis" (SPIE, 2008): predict each cover
    pixel from its 4 neighbours, and compute the weighted mean residual
    r_i = (s_i - flip(s_i)) * (s_i - predicted_i).  Twice the mean residual
    is an unbiased estimator of the payload rate (a replaced LSB only flips
    the pixel half the time, hence the factor 2).  Weights follow the
    moderated scheme 1/(5 + local variance).

    Returns a float in [0, 1].
    """
    channel = np.asarray(channel, dtype=np.uint8)
    if channel.shape[0] < 3 or channel.shape[1] < 3:
        return 0.0

    s = channel[1:-1, 1:-1].astype(np.float64)
    flipped = (_flip_lsb(channel[1:-1, 1:-1])).astype(np.float64)
    up = channel[:-2, 1:-1].astype(np.float64)
    down = channel[2:, 1:-1].astype(np.float64)
    left = channel[1:-1, :-2].astype(np.float64)
    right = channel[1:-1, 2:].astype(np.float64)
    predicted = (up + down + left + right) / 4.0

    residual = (s - flipped) * (s - predicted)

    local_var = ((up - predicted) ** 2 + (down - predicted) ** 2 +
                 (left - predicted) ** 2 + (right - predicted) ** 2) / 4.0
    weights = 1.0 / (5.0 + local_var)

    rate = 2.0 * float(np.sum(weights * residual) / np.sum(weights))
    return float(np.clip(rate, 0.0, 1.0))


def lsb_plane_compression_ratio(np_img):
    """Measure how incompressible the LSB plane is.

    Natural, structured content leaves correlations in the LSB plane that
    deflate compresses well; encrypted payload bits look random and stay
    near the 1:1 size ratio.  Returns compressed/raw size in [0, 1+].
    """
    plane = np.packbits(np.bitwise_and(np_img, 1).reshape(-1))
    raw = plane.tobytes()
    if not raw:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)

# NudeNet v3 detector classes that indicate exposed nudity.  The v2-era
# prefix names previously used here (EXPOSED_BREAST, EXPOSED_ANUS, ...)
# match nothing on nudenet >= 3.0 -- breast/buttocks/anus nudity was
# silently reported as score 0.
NUDITY_CLASSES = frozenset({
    'ANUS_EXPOSED', 'BUTTOCKS_EXPOSED', 'FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_EXPOSED', 'MALE_BREAST_EXPOSED', 'MALE_GENITALIA_EXPOSED',
})
NUDITY_DETECTION_THRESHOLD = 0.5  # precision gate: nudenet's own floor is 0.2;
# 0.2-0.5 detections are dropped -- raise recall by lowering if triage needs it


# Function to detect nudity in an image
def detect_nudity(image_path):
    """Score exposed nudity in an image.

    Returns {'result': <list of detection dicts on success, a plain
    string message on unsupported/failed paths>, 'score': <max kept
    detection score, 0 when nothing survives the filter>}.
    """
    try:
        # Check if the image is a GIF
        if image_path.lower().endswith('.gif'):
            return {"result": "GIF images are not supported for nudity detection", "score": 0}

        detections = nude_detector.detect(image_path)
        nudity_detections = [
            detection for detection in detections
            if detection.get('class') in NUDITY_CLASSES
            and detection.get('score', 0) >= NUDITY_DETECTION_THRESHOLD
        ]

        # Calculate overall nudity score
        if nudity_detections:
            nudity_score = max(detection.get('score', 0) for detection in nudity_detections)
        else:
            nudity_score = 0

        return {
            "result": nudity_detections,
            "score": nudity_score
        }
    except Exception as e:
        logger.error(f"Error during nudity detection in image: {image_path}. Error: {e}")
        return {"result": [], "score": 0}

# Near-duplicate threshold: dHash distance at/below which two images are
# flagged as copies. Calibration corpora (libphash) show <=7 bits catches
# 95% of edited photograph copies with ~4 false pairs per 19,900 distinct
# pairs; 10 is the common recall-leaning starting point.
PERCEPTUAL_HASH_DUPLICATE_THRESHOLD = 10

_duplicate_registry = {}  # dhash hex -> identifier of first occurrence
_duplicate_registry_lock = threading.Lock()


def reset_duplicate_registry():
    """Clear the near-duplicate registry (fresh state per scan run)."""
    with _duplicate_registry_lock:
        _duplicate_registry.clear()


def perceptual_hash(pil_image):
    """Return a 64-bit dHash (16 hex chars) of a PIL image.

    The difference hash survives resize, recompression and moderate
    brightness/contrast edits, but NOT crops, mirrors or rotations.
    Animated GIFs are hashed from their first frame.  Flat, low-detail
    images produce similar hashes -- treat matches on such content as
    weak evidence.
    """
    small = pil_image.convert('L').resize((9, 8))
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).astype(np.uint8)  # 8x8 = 64 bits
    return np.packbits(bits).tobytes().hex()


def hamming_distance(hash_a, hash_b):
    """Number of differing bits between two hex hashes."""
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def find_or_register_duplicate(image_hash, identifier):
    """Register a hash; return (first_identifier, distance) if a hash
    within PERCEPTUAL_HASH_DUPLICATE_THRESHOLD was already seen, else None.
    The first occurrence stays canonical; duplicates are not re-registered.
    """
    with _duplicate_registry_lock:
        # Flat images (solid colours, blank spacers) all hash to all-zeros
        # or all-ones; that carries no structure and must never read as a
        # 'near-duplicate' match.
        if image_hash in ('0' * 16, 'f' * 16):
            return None
        for existing_hash, existing_id in _duplicate_registry.items():
            distance = hamming_distance(image_hash, existing_hash)
            if distance <= PERCEPTUAL_HASH_DUPLICATE_THRESHOLD:
                return existing_id, distance
        _duplicate_registry[image_hash] = identifier
        return None


# Function to generate a hash for an image
def get_image_hash(pil_image):
    image_bytes = pil_image.tobytes()
    md5_hash = hashlib.md5(image_bytes)
    return md5_hash.hexdigest()

# Word-confidence gate: tesseract calibration corpora show conf < 60 catches
# most wrong reads (69%) at the cost of ~11% of correct ones; -1 rows are
# structural (page/block/line), never words.
OCR_MIN_CONFIDENCE = 60

# Never upscale beyond this many pixels on a side, even for tiny strips.
OCR_MAX_UPSCALED_DIMENSION = 12000


def preprocess_for_ocr(pil_image):
    """Prepare an image for OCR: grayscale, 3x upscale when small, autocontrast.

    Tesseract's defaults assume document-scale text; small web text
    (watermarks, captions, usernames) is the number one cause of missed
    reads.  Binarization is deliberately avoided -- it destroys text
    overlaid on photographic content -- and autocontrast uses a small
    cutoff so a single specular pixel cannot flatten the histogram.
    """
    gray = pil_image.convert('L')
    if min(gray.size) < 64:
        factor = min(3.0, OCR_MAX_UPSCALED_DIMENSION / max(gray.size))
        if factor > 1.0:
            gray = gray.resize((round(gray.width * factor), round(gray.height * factor)),
                               Image.Resampling.LANCZOS)
    return ImageOps.autocontrast(gray, cutoff=1)


# Function to extract text from an image using OCR
def extract_text_from_image(pil_image):
    """OCR an image with preprocessing and per-word confidence gating.

    Words scoring below OCR_MIN_CONFIDENCE (and structural -1 rows and
    blank-text rows) are dropped; the gate is a tradeoff, not a cliff.
    """
    data = pytesseract.image_to_data(
        preprocess_for_ocr(pil_image), config='--psm 3', output_type=Output.DICT)
    words = [word.strip() for word, conf in zip(data['text'], data['conf'])
             if word.strip() and float(conf) >= OCR_MIN_CONFIDENCE]
    return ' '.join(words)

def _ratio_to_float(value):
    """Convert an EXIF rational (IFDRational, Fraction or (num, den)) to float."""
    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            num, den = value
            return float(num) / float(den) if den else 0.0
        raise ValueError(f"unexpected rational shape: {value!r}")
    return float(value)


def _gps_ref_char(value):
    """Normalize a GPS hemisphere ref to 'N'/'S'/'E'/'W'; None if unreadable."""
    if isinstance(value, (bytes, bytearray)):
        value = value.decode('ascii', errors='replace') if value else ''
    text = str(value).strip(' \t\r\n\x00').upper()
    return text[:1] if text in ('N', 'S', 'E', 'W') else None


def _decode_gps(gps_info):
    """Decode a GPSInfo IFD dict into analyst-readable entries.

    Returns {} when coordinates are absent or the hemisphere refs are
    missing-or-unrecognizable (a wrong hemisphere is a >100 km error --
    fail closed rather than guess).  Latitude and longitude convert from
    degrees/minutes/seconds rationals to signed decimal degrees (S/W
    negative); altitude is best-effort and never costs the position.
    """
    decoded = {}
    try:
        def dms_to_decimal(value):
            parts = [_ratio_to_float(part) for part in value]
            while len(parts) < 3:
                parts.append(0.0)
            return parts[0] + parts[1] / 60.0 + parts[2] / 3600.0

        lat_ref = _gps_ref_char(gps_info.get(1))
        lon_ref = _gps_ref_char(gps_info.get(3))
        if lat_ref not in ('N', 'S') or lon_ref not in ('E', 'W'):
            return {}
        latitude = dms_to_decimal(gps_info[2])       # GPSLatitude
        if lat_ref == 'S':
            latitude = -latitude
        longitude = dms_to_decimal(gps_info[4])      # GPSLongitude
        if lon_ref == 'W':
            longitude = -longitude
        decoded['GPSPosition'] = f"{latitude + 0.0:.6f}, {longitude + 0.0:.6f}"
    except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError):
        return {}

    try:
        if 6 in gps_info:                            # GPSAltitude
            altitude = _ratio_to_float(gps_info[6])
            ref = gps_info.get(5, 0)                 # GPSAltitudeRef (often bytes)
            if isinstance(ref, (bytes, bytearray)):
                ref = ref[0] if ref else 0
            if int(ref) == 1:                        # 1 = below sea level
                altitude = -altitude
            decoded['GPSAltitude'] = f"{altitude:.1f}m"
    except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError):
        pass  # altitude is optional; keep the position
    return decoded


# Function to extract EXIF data from an image
def extract_exif_data(pil_image):
    """Extract analyst-readable EXIF metadata from a PIL image.

    GPS coordinates are decoded to signed decimal degrees and placed FIRST
    in the result (so they survive the report cell's truncation), Exif-IFD
    sub-tags (DateTimeOriginal, ExposureTime, ...) are merged in, raw IFD
    offsets are never emitted, and MakerNote blobs are abbreviated.
    """
    exif_data = {}
    if hasattr(pil_image, 'getexif'):
        exif_info = pil_image.getexif()
        if exif_info is not None:
            for tag, value in exif_info.items():
                if tag in (34665, 34853):  # ExifIFD / GPSInfo pointer offsets
                    continue               # expanded below
                decoded = ExifTags.TAGS.get(tag, tag)
                if tag == 37500:           # MakerNote: binary blob
                    exif_data[decoded] = str(value)[:40]
                    continue
                exif_data[decoded] = value
            try:
                for tag, value in dict(exif_info.get_ifd(0x8769)).items():
                    if tag in (40965, 34853):  # nested IFD pointers
                        continue
                    if tag == 37500:           # MakerNote usually lives HERE
                        exif_data.setdefault('MakerNote', str(value)[:40])
                        continue
                    # setdefault: top-level IFD0 wins on name collisions
                    exif_data.setdefault(ExifTags.TAGS.get(tag, str(tag)), value)
            except Exception as e:
                logger.debug(f"Exif IFD expansion failed: {e}")
            gps = _decode_gps(dict(exif_info.get_ifd(34853) or {}))
            if gps:
                # position first so the 300-char report cell keeps it
                exif_data = {**gps, **exif_data}
    return exif_data

_face_cascades = {}
_face_cascade_lock = threading.Lock()


def _load_face_cascade(name):
    """Load (and cache) one of OpenCV's bundled haar cascades."""
    with _face_cascade_lock:
        if name not in _face_cascades:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
            if cascade.empty():
                logger.error(f"Error loading face cascade classifier: {name}")
            _face_cascades[name] = cascade
        return _face_cascades[name]


def _iou(box_a, box_b):
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


# Function to detect faces in an image
def detect_faces(pil_image):
    """Count faces across frontal and BOTH profile orientations.

    The frontal cascade is one-sided: profile faces are detected with the
    profile cascade, and the opposite profile by running it again on the
    horizontally flipped image (boxes translated back).  Detections from
    the three passes overlapping by IoU >= 0.5 count as one face.
    Returns an integer count (0 on error).
    """
    try:
        gray = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2GRAY)
        width = gray.shape[1]
        detect_kwargs = {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)}
        frontal = _load_face_cascade('haarcascade_frontalface_default.xml')
        profile = _load_face_cascade('haarcascade_profileface.xml')

        detections = list(frontal.detectMultiScale(gray, **detect_kwargs))
        detections.extend(profile.detectMultiScale(gray, **detect_kwargs))
        flipped = cv2.flip(gray, 1)
        for (x, y, w, h) in profile.detectMultiScale(flipped, **detect_kwargs):
            detections.append((width - int(x) - int(w), int(y), int(w), int(h)))

        counted = []
        for box in detections:
            if all(_iou(box, other) < 0.5 for other in counted):
                counted.append(box)
        return len(counted)
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return 0

# Function to download an image with a random user agent
def download_image(url):
    try:
        if url.startswith('data:'):
            # Handle data URLs
            _, encoded = url.split(",", 1)
            data = base64.b64decode(encoded)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as f:
                f.write(data)
                f.flush()
                real_mime_type = magic.from_file(f.name, mime=True)
                if not real_mime_type.startswith('image/'):
                    logger.warning(f"Content from data URL is not a valid image (MIME type: {real_mime_type}).")
                    return None
                correct_extension = real_mime_type.split('/')[1].lower()
                new_filename = f'{f.name}.{correct_extension}'
                os.rename(f.name, new_filename)
                return new_filename
        else:
            headers = {'User-Agent': user_agent.random}
            response = requests.get(url, timeout=10, headers=headers, stream=True)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()
            if not content_type.startswith('image/'):
                logger.warning(f"Content at {url} does not appear to be a valid image (Content-Type: {content_type}).")
                return None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                f.flush()
                real_mime_type = magic.from_file(f.name, mime=True)
                if not real_mime_type.startswith('image/'):
                    logger.warning(f"Downloaded content is not a valid image (MIME type: {real_mime_type}).")
                    return None
                correct_extension = real_mime_type.split('/')[1].lower()
                new_filename = f'{f.name}.{correct_extension}'
                os.rename(f.name, new_filename)
                return new_filename
    except requests.RequestException as e:
        logger.error(f"Error downloading image from {url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error downloading image from {url}: {e}")
    return None

# Generator names that appear in Software/CreatorTool-style metadata tags
GENERATOR_SOFTWARE_HINTS = (
    'midjourney', 'dall-e', 'dalle', 'stable diffusion', 'stablediffusion',
    'sdxl', 'comfyui', 'automatic1111', 'invokeai', 'novelai', 'fooocus',
    'firefly', 'imagen', 'gpt-image', 'chatgpt', 'gemini', 'flux',
    'leonardo', 'nightcafe', 'craiyon', 'runwayml', 'seedream',
)

# PNG tEXt keys that are generator metadata by themselves
GENERATOR_TEXT_CHUNK_KEYS = {
    'sd-metadata': 'Stable Diffusion metadata chunk',
    'dream': 'Stable Diffusion (dream) metadata chunk',
    'invokeai_metadata': 'InvokeAI metadata chunk',
    'fooocus_scheme': 'Fooocus metadata chunk',
}

# Metadata keys whose VALUES are scanned for generator names
SOFTWARE_VALUE_KEYS = ('software', 'creatortool', 'creator_tool')


def scan_generator_metadata(image_path):
    """Search an image's bytes and metadata for generator provenance.

    Looks for Stable Diffusion / ComfyUI / NovelAI parameter chunks, known
    generator names in software tags, IPTC trainedAlgorithmicMedia source
    declarations, and C2PA content-credential manifests.  Returns a list of
    human-readable indicators.  An empty list proves nothing: re-encoding,
    screenshots and most social platforms strip metadata entirely.
    """
    indicators = []

    raw = Path(image_path).read_bytes()
    # C2PA manifests need BOTH a manifest box (PNG caBX chunk / JPEG APP11
    # JUMBF) and a claim label -- bare "c2pa" bytes occur by chance in
    # compressed image data.
    has_manifest_box = b'caBX' in raw or b'JUMBF' in raw
    has_claim = (b'c2pa.claim' in raw or b'c2pa.signature' in raw
                 or b'c2pa.actions' in raw)
    if has_manifest_box and has_claim:
        indicators.append('C2PA content credentials manifest')
    if b'trainedAlgorithmicMedia' in raw:
        indicators.append('IPTC digital source type marks trained algorithmic media')

    with Image.open(image_path) as img:
        for key, value in (getattr(img, 'text', {}) or {}).items():
            key_l = key.lower()
            value_s = str(value)
            value_l = value_s.lower()
            if key_l == 'parameters' and 'steps:' in value_l \
                    and ('sampler' in value_l or 'seed:' in value_l):
                indicators.append('Stable Diffusion parameters chunk')
            if key_l in GENERATOR_TEXT_CHUNK_KEYS:
                indicators.append(GENERATOR_TEXT_CHUNK_KEYS[key_l])
            if key_l in ('prompt', 'workflow') \
                    and ('class_type' in value_l or '"nodes"' in value_l):
                indicators.append('ComfyUI workflow graph chunk')
            if key_l == 'comment' and '"uc"' in value_l:
                indicators.append('NovelAI generation metadata (Comment chunk)')
            if key_l in SOFTWARE_VALUE_KEYS:
                for hint in GENERATOR_SOFTWARE_HINTS:
                    if hint in value_l:
                        indicators.append(f'Generator software tag: {value_s.strip()[:60]}')
                        break

        if hasattr(img, 'getexif'):
            exif = img.getexif() or {}
            for tag in (11, 270, 305, 37510):
                # ProcessingSoftware, ImageDescription, Software, UserComment
                value = exif.get(tag)
                if value is None:
                    continue
                value_l = str(value).lower()
                if 'steps:' in value_l and 'sampler' in value_l:
                    indicators.append('Stable Diffusion parameters in EXIF UserComment')
                for hint in GENERATOR_SOFTWARE_HINTS:
                    if hint in value_l:
                        indicators.append(f'Generator software tag: {str(value)[:60]}')
                        break

    return indicators


def spectral_ai_analysis(gray):
    """Return (gan_grid_score, high_frequency_ratio) for a grayscale image.

    GAN up-samplers leave periodic grid peaks in the Fourier spectrum
    (Frank et al. 2020); diffusion models underestimate high frequencies
    (Durall & Keuper 2020; F2C 2025).  The spectrum is computed on the
    high-pass residual (image minus its median-filtered copy), the standard
    preprocessing, and reduced to an azimuthally-averaged radial profile.
    """
    gray = np.asarray(gray, dtype=np.float64)
    if gray.size < 256:
        return 0.0, 0.0
    blurred = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float64)
    residual = gray - blurred
    if float(np.sum(residual * residual)) < 1e-9:
        return 0.0, 0.0  # perfectly flat

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual))) ** 2
    h, w = spectrum.shape
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((y - h // 2) ** 2 + (x - w // 2) ** 2).astype(int)
    max_r = min(h, w) // 2
    counts = np.bincount(radius.ravel(), minlength=radius.max() + 1)
    power = np.bincount(radius.ravel(), weights=spectrum.ravel(),
                        minlength=radius.max() + 1)
    profile = power / np.maximum(counts, 1)

    band = profile[int(0.25 * max_r):int(0.95 * max_r) + 1]
    smoothed = np.convolve(band, np.ones(3) / 3, mode='same')
    peak_ratio = float(np.max(smoothed) / (np.median(smoothed) + 1e-12))
    gan_grid = float(np.clip((peak_ratio - 4.0) / 6.0, 0.0, 1.0))

    total = float(np.sum(profile[1:]))
    hf_ratio = float(np.sum(profile[int(0.7 * max_r):]) / (total + 1e-12))
    return gan_grid, hf_ratio


def detect_ai_generated(image_path):
    """Heuristic AI-generated image detection.

    Layer 1, generator provenance metadata (Stable Diffusion / ComfyUI /
    NovelAI parameter chunks, generator Software tags, IPTC
    trainedAlgorithmicMedia, C2PA manifests), is near-deterministic when
    present and short-circuits with a high score.  Layer 2, Fourier-domain
    heuristics, scores GAN up-sampling grid artifacts and a diffusion-style
    high-frequency deficit; the smoothness component alone cannot cross the
    detection threshold because naturally smooth images share that property.
    Returns (detected, score).
    """
    try:
        indicators = scan_generator_metadata(image_path)
        if indicators:
            for indicator in indicators:
                logger.info(f"AI-generation provenance in {image_path}: {indicator}")
            return True, 0.95

        with Image.open(image_path) as img:
            gray = np.asarray(img.convert('L'), dtype=np.float64)

        gan_grid, hf_ratio = spectral_ai_analysis(gray)
        smooth_component = float(np.clip((0.35 - hf_ratio) / 0.35, 0.0, 1.0))
        score = 0.6 * gan_grid + 0.4 * smooth_component
        return bool(score >= 0.5), float(np.clip(score, 0.0, 1.0))

    except Exception as e:
        logger.error(f"Error during AI generation detection in image: {image_path}. Error: {e}")
        return False, None

# Function to get the dominant color of an image
def get_dominant_color(image_path):
    try:
        with Image.open(image_path) as img:
            # ColorThief accepts a filename or a (binary) file object, not an
            # open PIL Image, so re-encode the RGB-normalized pixels in memory.
            buffer = io.BytesIO()
            img.convert('RGB').save(buffer, format='PNG')
            buffer.seek(0)
            color_thief = ColorThief(buffer)
            dominant_color = color_thief.get_color(quality=1)
        return dominant_color
    except OSError as e:
        logger.error(f"OS error when getting dominant color for {image_path}: {e}")
    except ValueError as e:
        logger.error(f"Value error when getting dominant color for {image_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error when getting dominant color for {image_path}: {e}")
    return None

# Function to detect steganography in an image
def detect_steganography(image_path):
    """Heuristic LSB-replacement steganography detector.

    Combines two published quantitative payload estimators -- RS analysis
    (Fridrich, Goljan & Du 2001) and WS analysis (Ker & Bohme 2008), the
    latter recommended as the first-choice method for LSB replacement --
    with an LSB-plane incompressibility signal.  The two estimator rates are
    averaged per channel and across channels, then blended with the
    compression signal.  Returns (detected, score, hidden_text); a score of
    0.5 or higher is reported as suspected steganography.
    """
    try:
        with Image.open(image_path) as img:
            # Handle PNG transparency and convert to RGB
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                img = bg
            if img.mode != 'RGB':
                img = img.convert('RGB')

            np_img = np.array(img).astype(np.uint8)

            channels = [np_img[:, :, c] for c in range(np_img.shape[2])]
            ws_rate = float(np.mean([ws_analysis(ch) for ch in channels]))
            rs_rate = float(np.mean([rs_analysis(ch) for ch in channels]))
            estimator_rate = 0.65 * ws_rate + 0.35 * rs_rate

            # Random-looking LSB planes do not compress; structured ones do.
            compression_ratio = lsb_plane_compression_ratio(np_img)
            compression_component = min(max((compression_ratio - 0.7) / 0.3, 0.0), 1.0)

            stego_score = 0.75 * estimator_rate + 0.25 * compression_component

            # Detect hidden text
            hidden_text = detect_hidden_text(np_img)

            return stego_score >= 0.5, float(stego_score), hidden_text

    except Exception as e:
        logger.error(f"Error during steganography detection in image: {image_path}. Error: {e}")
        return False, None, None

# Byte signatures of common file types stego tools embed as payloads.
# Every signature is >= 5 bytes or structurally validated: at realistic
# resolutions the 64 extraction plans scan tens of MB of candidate bytes,
# and shorter or unvalidated magics (bare PK\x03\x04, \xff\xd8\xff, BZh,
# \x1f\x8b\x08) match by chance on most clean images.
EMBEDDED_FILE_SIGNATURES = (
    (b'PK\x03\x04\x14\x00', 'ZIP archive'),
    (b'\x89PNG\r\n\x1a\n', 'PNG image'),
    (b'\xff\xd8\xff\xe0\x00\x10', 'JPEG image (JFIF)'),
    (b'GIF87a', 'GIF image'),
    (b'GIF89a', 'GIF image'),
    (b'Rar!\x1a\x07\x00', 'RAR archive'),
    (b"7z\xbc\xaf'\x1c", '7-Zip archive'),
    (b'%PDF-', 'PDF document'),
    (b'-----BEGIN ', 'PEM/ASCII-armored data'),
)

# Structurally-validated signatures that need variable bytes matched:
# - JPEG APP1/Exif: SOI + APP1 marker + 2-byte length + 'Exif\0'
# - bzip2: 'BZh' + compression level digit + block magic (pi digits)
# - gzip: magic + CM=deflate + FLG without reserved bits + 4-byte MTIME
#   + XFL byte as written by standard implementations
EMBEDDED_FILE_SIGNATURE_PATTERNS = (
    (rb'\xff\xd8\xff\xe1.{2}Exif\x00', 'JPEG image (EXIF)'),
    (rb'BZh[1-9]\x31\x41\x59\x26\x53\x59', 'bzip2 stream'),
    (rb'\x1f\x8b\x08[\x00-\x1f].{4}[\x00\x02\x04\x08]', 'gzip stream'),
)


def detect_hidden_text(np_img):
    """Recover hidden payloads from LSB planes across many embedding orders.

    Real embedding tools vary channel (R/G/B/interleaved), traversal
    (row/column-major), bit order (MSB/LSB-first) and bit plane (1 or 2),
    so we scan a catalog of 32 extraction plans -- the approach taken by
    tools like zsteg.  Two finding classes are reported:

    * embedded file signatures (ZIP/PNG/JPEG/RAR/7z/PDF/gzip/bzip2/PEM),
      which are searched in every plan's raw bytes;
    * printable text runs, gated on length >= 8, >= 5 distinct characters
      and >= 70% alphanumeric content so structured-but-clean planes
      (e.g. repeated 0x55 from a smooth gradient) do not false-positive.

    Returns the strongest finding as a string, or None.
    """
    try:
        np_img = np.asarray(np_img)
        if np_img.ndim == 2:
            np_img = np.stack((np_img,) * 3, axis=-1)

        def iter_candidates():
            for plane_bit, plane_name in ((1, 'bit1'), (2, 'bit2')):
                for transposed, trav_name in ((False, 'row'), (True, 'col')):
                    img = np_img.transpose(1, 0, 2) if transposed else np_img
                    planes = [((img[:, :, c] & plane_bit) >> (plane_bit - 1)).astype(np.uint8)
                              for c in range(3)]
                    streams = [
                        ('R', planes[0]), ('G', planes[1]), ('B', planes[2]),
                        ('RGB-interleaved', np.stack(planes, axis=-1)),
                    ]
                    for channel_name, stream in streams:
                        bits = stream.reshape(-1)
                        for bitorder, order_name in (('big', 'msb'), ('little', 'lsb')):
                            packed = np.packbits(bits, bitorder=bitorder)
                            yield (f"{channel_name}/{trav_name}/{order_name}/{plane_name}",
                                   packed.tobytes())

        def find_signature(data):
            for signature, label in EMBEDDED_FILE_SIGNATURES:
                offset = data.find(signature)
                if offset != -1:
                    return label, offset
            for pattern, label in EMBEDDED_FILE_SIGNATURE_PATTERNS:
                match = re.search(pattern, data, re.DOTALL)
                if match:
                    return label, match.start()
            return None, -1

        best = ("", "")
        for plan, data in iter_candidates():
            label, offset = find_signature(data)
            if label is not None:
                preview = data[offset:offset + 24].decode('ascii', errors='replace')
                return f"{label} embedded via LSB ({plan}): {preview!r}"
            for run_bytes in re.findall(rb'[ -~]{8,}', data):
                run = run_bytes.decode('ascii')
                distinct = len(set(run))
                alnum = sum(c.isalnum() or c == ' ' for c in run) / len(run)
                if distinct >= 5 and alnum >= 0.7 and len(run) > len(best[0]):
                    best = (run, plan)
        if best[0]:
            return f"{best[0][:120]} (via LSB {best[1]})"
        return None

    except Exception as e:
        logger.error(f"Error during hidden text detection: {e}")
        return None

def detect_age(image_path):
    """Return the youngest detected age in an image, or None.

    DeepFace lists one entry per detected face in arbitrary order; the
    YOUNGEST is the one the child-exploitation screen must see (taking the
    first entry could silently ignore a child standing beside an adult).
    Entries with face_confidence == 0 are DeepFace's whole-image fallback
    (an 'age' of the scenery under enforce_detection=False) and are
    discarded.
    """
    try:
        result = DeepFace.analyze(image_path, actions=['age'],
                                  enforce_detection=False, silent=True)
        faces = result if isinstance(result, list) else [result]
        ages = []
        for face in faces:
            if float(face.get('face_confidence', 1.0) or 0.0) <= 0.0:
                continue
            try:
                # truncation (not rounding) is deliberate: conservative for
                # an age < 18 screen (17.9 estimates as 17)
                ages.append(int(float(face.get('age'))))
            except (TypeError, ValueError):
                continue
        if ages:
            return min(ages)
        logger.warning(f"No face detected for age estimation in image: {image_path}")
        return None
    except ValueError as e:
        if "Face could not be detected" in str(e):
            logger.warning(f"No face detected for age estimation in image: {image_path}")
            return None
        logger.error(f"Error during age detection in image: {image_path}. Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during age detection in image: {image_path}. Error: {e}")
        return None

def detect_potential_child_exploitation(image_path, faces_detected):
    if not faces_detected:
        return False, None

    try:
        age = detect_age(image_path)
        if age is not None and age < 18:
            nudity_result = detect_nudity(image_path)
            if nudity_result['score'] > 0.7:  # High confidence of nudity
                return True, f"Potential child exploitation detected. Estimated age: {age}, Nudity score: {nudity_result['score']}"
        return False, None
    except Exception as e:
        logger.error(f"Error during child exploitation detection in image: {image_path}. Error: {e}")
        return False, None

def extract_gif_frames(image_path, max_frames=4):
    """Extract up to *max_frames* GIF frames as temporary PNG files.

    Frames are sampled at evenly spaced indices spanning the whole
    animation (first and last frame always included), so content appearing
    late in the loop is still analyzed.  On a corrupt or truncated GIF the
    already-extracted files are removed before the error propagates.
    Caller must remove the returned files.
    """
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    frame_paths = []
    try:
        with Image.open(image_path) as gif:
            total = getattr(gif, "n_frames", 1)
            if total <= max_frames:
                indices = list(range(total))
            elif max_frames == 1:
                indices = [(total - 1) // 2]
            else:
                indices = sorted({round(i * (total - 1) / (max_frames - 1))
                                  for i in range(max_frames)})
            for index in indices:
                gif.seek(index)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as frame_file:
                    # register before decoding: a decode failure must still
                    # clean up the file this very iteration created
                    frame_paths.append(frame_file.name)
                    gif.convert("RGB").save(frame_file.name, format="PNG")
    except Exception:
        for frame_path in frame_paths:
            Path(frame_path).unlink(missing_ok=True)
        raise
    return frame_paths


# Function to process an image and extract information
def process_image(image_path):
    downloaded_file = None
    original_url = image_path if not os.path.isfile(image_path) else None
    try:
        initialize_nude_detector()
        if not os.path.isfile(image_path):
            logger.info(f"Downloading image from URL: {image_path}")
            downloaded_file = download_image(image_path)
            if not downloaded_file:
                logger.warning(f"Failed to download image from URL: {image_path}")
                return None
            image_path = downloaded_file
            logger.info(f"Image downloaded successfully: {image_path}")

        real_mime_type = magic.from_file(image_path, mime=True)
        logger.info(f"Detected MIME type: {real_mime_type}")
        if not real_mime_type.startswith('image/'):
            logger.warning(f"Unsupported file type {real_mime_type} encountered at: {image_path}")
            return None

        if real_mime_type == 'image/svg+xml':
            logger.warning(f"SVG file encountered at: {image_path}. SVG processing is not supported.")
            return None

        with Image.open(image_path) as img_obj:
            logger.info(f"Processing image: {image_path}")
            image_hash = get_image_hash(img_obj)
            try:
                phash = perceptual_hash(img_obj)
            except Exception as e:
                logger.warning(f"Perceptual hashing failed for {image_path}: {e}")
                phash = None
            try:
                exif_info = extract_exif_data(img_obj)
            except Exception as e:
                logger.warning(f"EXIF extraction failed for {image_path}: {e}")
                exif_info = {}
            try:
                detected_text = extract_text_from_image(img_obj)
            except Exception as e:
                logger.warning(f"OCR extraction failed for {image_path}: {e}")
                detected_text = ""

            if real_mime_type == 'image/gif':
                logger.info("Processing GIF image")
                # GIFs are palette-based: LSB stego/AI heuristics would be
                # misleading on decoded rasters, so those stay N/A, but the
                # content detectors run on frames sampled across the animation.
                frame_paths = []
                try:
                    frame_paths = extract_gif_frames(image_path, max_frames=4)
                    worst_nudity = {"result": [], "score": 0}
                    for frame_path in frame_paths:
                        frame_nudity = detect_nudity(frame_path)
                        if frame_nudity['score'] > worst_nudity['score']:
                            worst_nudity = frame_nudity
                    nudity_results = worst_nudity

                    first_frame = frame_paths[0] if frame_paths else None
                    if first_frame:
                        dominant_color = get_dominant_color(first_frame)
                        with Image.open(first_frame) as frame_img:
                            faces_found = detect_faces(frame_img)
                    else:
                        dominant_color = 'N/A'
                        faces_found = 0
                    frame_age = detect_age(first_frame) if faces_found and first_frame else None
                    age_estimation = frame_age if frame_age is not None else 'N/A'
                    stego_detected, stego_score = False, None
                    # Consult the worst-frame nudity evidence (thresholds mirror
                    # detect_potential_child_exploitation) rather than re-running
                    # the detector on the first frame only.
                    if faces_found and frame_age is not None and frame_age < 18 \
                            and nudity_results['score'] > 0.7:
                        child_exploitation_detected = True
                        child_exploitation_details = (
                            f"Potential child exploitation detected. Estimated age: {frame_age}, "
                            f"Nudity score: {nudity_results['score']}")
                    else:
                        child_exploitation_detected, child_exploitation_details = False, None
                    # img_obj is the still-open handle from the top of
                    # process_image; reuse it rather than reopening the file
                    frame_count = getattr(img_obj, 'n_frames', 1)
                    # info['duration'] is the per-frame delay in ms, not the
                    # total animation time
                    duration = img_obj.info.get('duration', 'N/A')
                    ai_generated, ai_score = False, None
                    hidden_text = None
                except Exception as e:
                    logger.warning(f"GIF frame analysis failed for {image_path}: {e}")
                    nudity_results = {"result": "GIF frame extraction failed", "score": 0.0}
                    dominant_color = 'N/A'
                    faces_found = 0
                    age_estimation = 'N/A'
                    stego_detected, stego_score = False, None
                    hidden_text = None
                    child_exploitation_detected, child_exploitation_details = False, None
                    frame_count = 'N/A'
                    duration = 'N/A'
                    ai_generated, ai_score = False, None
                finally:
                    for frame_path in frame_paths:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
            else:
                logger.info("Processing non-GIF image")
                dominant_color = get_dominant_color(image_path)
                faces_found = detect_faces(img_obj)
                stego_detected, stego_score, hidden_text = detect_steganography(image_path)
                age_estimation = detect_age(image_path) if faces_found else 'N/A'
                frame_count = 'N/A'
                duration = 'N/A'
                ai_generated, ai_score = detect_ai_generated(image_path)
                child_exploitation_detected, child_exploitation_details = detect_potential_child_exploitation(image_path, faces_found)
                nudity_results = detect_nudity(image_path)

            # Surface the OCR and EXIF results in the output row
            ocr_text = ' '.join(detected_text.split())[:200] if detected_text and detected_text.strip() else "None detected"
            exif_summary = json.dumps(exif_info, default=str)[:300] if exif_info else "None"

            # Compact single-line nudity cell (the old json.dumps(indent=2)
            # spilled multiline JSON across the table and CSV rows)
            raw_detections = nudity_results['result']
            if isinstance(raw_detections, str):
                nudity_cell = raw_detections
            elif raw_detections:
                summary = ", ".join(
                    f"{d.get('class')}:{d.get('score', 0):.2f}"
                    if isinstance(d, dict) else str(d)
                    for d in raw_detections[:4])
                nudity_cell = f"{summary} (Score: {nudity_results['score']:.2f})"
            else:
                nudity_cell = f"None detected (Score: {nudity_results['score']:.2f})"

            # Near-duplicate detection against every image seen this scan
            phash_cell = phash if phash else "N/A"
            if phash:
                duplicate = find_or_register_duplicate(phash, original_url or image_path)
                if duplicate:
                    phash_cell = f"{phash} (near-duplicate of {duplicate[0]}, distance {duplicate[1]})"

            # Create a table row with the extracted information
            row = [
                original_url or image_path,
                image_hash,
                str(dominant_color),
                str(faces_found),
                nudity_cell,
                f"{'Yes' if stego_detected else 'No'} (Score: {stego_score:.2f})" if stego_score is not None else "N/A",
                str(age_estimation),
                hidden_text if hidden_text else "None detected",
                str(frame_count),
                str(duration),
                f"{'Yes' if ai_generated else 'No'} (Score: {ai_score:.2f})" if ai_score is not None else "N/A",
                "Yes" if child_exploitation_detected else "No",
                child_exploitation_details or "N/A",
                ocr_text,
                exif_summary,
                phash_cell
            ]
            logger.info(f"Successfully processed image: {image_path}")
            return row
    except (UnidentifiedImageError, OSError) as error:
        logger.error(f"Error processing image at {image_path}: {error}")
    except Exception as e:
        logger.error(f"Unexpected error processing image at {image_path}: {e}")
    finally:
        if downloaded_file and os.path.exists(downloaded_file):
            os.remove(downloaded_file)
            logger.info(f"Removed temporary file: {downloaded_file}")
    return None

def _same_host(url_a, url_b):
    """True when both URLs name the same host, ignoring case and
    scheme-default ports ('http://x:80' == 'http://x').

    A port that is NOT the scheme default stays significant, so
    'http://x:443' does not match 'https://x'.  IPv6 literals and URLs
    with unparseable ports compare conservatively (no match).
    """
    defaults = {'http': '80', 'https': '443'}
    ports = []
    hosts = []
    for url in (url_a, url_b):
        parsed = urlparse(url)
        try:
            port = parsed.port
        except ValueError:
            return False
        if port == int(defaults.get(parsed.scheme, -1)):
            port = None
        hosts.append((parsed.hostname or '').lower())
        ports.append(port)
    return hosts[0] == hosts[1] and ports[0] == ports[1] and hosts[0] != ''


# Function to crawl a website and process images
def crawl(url, depth, max_pages=100):
    """Crawl pages up to *depth*, processing images found on each page.

    Iterative breadth-first worklist bounded by *max_pages* total fetch
    ATTEMPTS (error pages consume budget).  Tracks visited pages (so link
    cycles terminate) and already-processed image URLs.  Links and images
    resolve against the final (post-redirect) URL, but the crawl's SCOPE
    stays anchored to the operator's start URL: images on a redirected
    landing are processed, while links to any other host -- including the
    redirect target's own pages, via open redirects -- are never followed.
    One failing image cannot abort a page's results.
    """
    reset_duplicate_registry()
    start_url = url.split('#', 1)[0]
    visited = set()
    seen_images = set()
    results = []
    queue = deque([(start_url, depth)])
    pending = {start_url}

    while queue and len(visited) < max_pages:
        page_url, page_depth = queue.popleft()
        pending.discard(page_url)
        page_url = page_url.split('#', 1)[0]  # fragments never change the page
        if page_depth <= 0 or page_url in visited:
            continue
        visited.add(page_url)

        try:
            headers = {'User-Agent': user_agent.random}
            response = requests.get(page_url, timeout=10, headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error crawling {page_url}: {e}")
            continue

        # Resolve links and images against the FINAL url after redirects
        base_url = response.url if isinstance(getattr(response, "url", None), str) else page_url
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract image URLs, skipping any already processed on an earlier page
        image_urls = []
        for img_tag in soup.select('img[src]'):
            image_url = urljoin(base_url, img_tag['src'])
            if image_url not in seen_images:
                seen_images.add(image_url)
                image_urls.append(image_url)

        # Process images in parallel; one failure must not abort the page
        with ThreadPoolExecutor() as executor:
            with Progress() as progress:
                task = progress.add_task("[green]Processing images...", total=len(image_urls))
                futures = [executor.submit(process_image, image_url) for image_url in image_urls]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")
                        result = None
                    if result:
                        results.append(result)
                    progress.update(task, advance=1)

        # Follow links only within the START url's host: an open redirect
        # must not be able to steer the crawl onto a foreign site
        for link in soup.select('a[href]'):
            next_url = urljoin(base_url, link['href']).split('#', 1)[0]
            parsed = urlparse(next_url)
            if parsed.scheme not in ('http', 'https'):
                continue
            if not _same_host(next_url, start_url):
                continue
            if next_url in visited or next_url in pending:
                continue
            queue.append((next_url, page_depth - 1))
            pending.add(next_url)

    if len(visited) >= max_pages and queue:
        logger.info(f"Page cap reached ({max_pages} pages); stopping crawl.")
    return results

def process_local_folder(folder_path):
    reset_duplicate_registry()
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    results = []
    files = [f for f in Path(folder_path).rglob('*') if f.suffix.lower() in image_extensions]

    with Progress() as progress:
        task = progress.add_task("[green]Processing local images...", total=len(files))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, str(file_path)) for file_path in files]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                finally:
                    progress.update(task, advance=1)

    return results

# Leading characters spreadsheet applications interpret as formula starters
CSV_FORMULA_PREFIXES = ('=', '+', '-', '@', '\t', '\r')


def sanitize_csv_cell(value):
    """Neutralize CSV formula injection (CWE-1236) in one output cell.

    Cell content in this tool is attacker-controlled (OCR text, EXIF,
    extracted stego payloads, URLs), and cells starting with =, +, -, @ or
    whitespace execute as formulas when the report is opened in Excel or
    LibreOffice.  Prefixing with an apostrophe makes the cell display as
    text; the console table keeps showing the raw value.
    """
    if isinstance(value, str) and value[:1] in CSV_FORMULA_PREFIXES:
        return "'" + value
    return value


def display_results(results, output_path="results.csv"):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Image Path/URL", style="cyan", overflow="fold")
    table.add_column("Hash", style="yellow", overflow="fold")
    table.add_column("Dominant Color", style="green")
    table.add_column("Faces Detected", style="bright_white")
    table.add_column("Nudity Assessment", style="red", no_wrap=True)
    table.add_column("Steganography Detected", style="bright_white", overflow="fold")
    table.add_column("Estimated Age", style="bright_blue")
    table.add_column("Hidden Text", style="bright_yellow", overflow="fold")
    table.add_column("Frame Count", style="bright_magenta")
    table.add_column("Duration", style="bright_cyan")
    table.add_column("AI Generated", style="bright_green")
    table.add_column("Child Exploitation", style="red")
    table.add_column("Child Exploitation Details", style="red", overflow="fold")
    table.add_column("Extracted Text (OCR)", style="bright_white", overflow="fold")
    table.add_column("EXIF Data", style="bright_white", overflow="fold")
    table.add_column("Perceptual Hash", style="yellow", overflow="fold")
    for row in results:
        table.add_row(*row)
    console.print("\nProcessed Image Information:")
    console.print(table)

    # Save results to CSV with improved headers
    csv_headers = [
        "Image URL",
        "Hash",
        "Dominant Color",
        "Faces Detected",
        "Nudity Assessment",
        "Steganography Detected",
        "Estimated Age",
        "Hidden Text",
        "Frame Count",
        "Duration",
        "AI Generated",
        "Child Exploitation",
        "Child Exploitation Details",
        "Extracted Text (OCR)",
        "EXIF Data",
        "Perceptual Hash"
    ]
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)  # Write header row
        writer.writerows([[sanitize_csv_cell(cell) for cell in row] for row in results])

    logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Suspicious Image Collection Kit -- SiCK v0.21")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--url", help="The URL to start crawling")
    group.add_argument("-f", "--folder", help="Local folder to scan for images")
    parser.add_argument("-d", "--depth", type=int, default=1, help="Depth of crawling (for URL mode)")
    parser.add_argument("-m", "--max-pages", type=int, default=100,
                        help="Maximum number of pages to fetch while crawling (default 100)")
    parser.add_argument("-o", "--output", default="results.csv",
                        help="Path for the CSV report (default: results.csv in the current directory)")
    parser.add_argument("-v", "--verbose", action='store_true', help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    reset_duplicate_registry()

    if args.url:
        results = crawl(args.url, args.depth, args.max_pages)
    else:
        results = process_local_folder(args.folder)

    if results:
        display_results(results, args.output)

if __name__ == "__main__":
    main()
