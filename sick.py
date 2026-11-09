from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import ExifTags, UnidentifiedImageError, ImageChops
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
import numpy as np
import tempfile
import requests
import argparse
import hashlib
import zlib
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

def initialize_nude_detector():
    global nude_detector
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

# Function to detect nudity in an image
def detect_nudity(image_path):
    try:
        # Check if the image is a GIF
        if image_path.lower().endswith('.gif'):
            return {"result": "GIF images are not supported for nudity detection", "score": 0}
        
        detections = nude_detector.detect(image_path)
        nudity_related_classes = {
            'EXPOSED_ANUS', 'EXPOSED_BUTTOCKS', 'EXPOSED_BREAST',
            'EXPOSED_GENITALIA', 'MALE_GENITALIA_EXPOSED', 'FEMALE_GENITALIA_EXPOSED'
        }
        nudity_detections = [
            detection for detection in detections
            if detection.get('class') in nudity_related_classes and detection.get('score', 0) >= 0.5
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

# Function to generate a hash for an image
def get_image_hash(pil_image):
    image_bytes = pil_image.tobytes()
    md5_hash = hashlib.md5(image_bytes)
    return md5_hash.hexdigest()

# Function to extract text from an image using OCR
def extract_text_from_image(pil_image):
    text = pytesseract.image_to_string(pil_image)
    return text

# Function to extract EXIF data from an image
def extract_exif_data(pil_image):
    exif_data = {}
    if hasattr(pil_image, 'getexif'):
        exif_info = pil_image.getexif()
        if exif_info is not None:
            for tag, value in exif_info.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif_data[decoded] = value
    return exif_data

# Function to detect faces in an image
def detect_faces(pil_image):
    try:
        cv_image = np.array(pil_image)
        if len(cv_image.shape) == 2:
            gray_image = cv_image
        elif len(cv_image.shape) == 3:
            if cv_image.shape[2] == 4:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        else:
            logger.warning(f"Unexpected image shape: {cv_image.shape}")
            return False
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            logger.error("Error loading face cascade classifier")
            return False
        
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(faces) > 0
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return False

# Function to download an image with a random user agent
def download_image(url):
    try:
        if url.startswith('data:'):
            # Handle data URLs
            header, encoded = url.split(",", 1)
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

def detect_hidden_text(np_img):
    try:
        # Extract LSB from each color channel
        lsb_r = np.bitwise_and(np_img[:,:,0], 1).flatten()
        lsb_g = np.bitwise_and(np_img[:,:,1], 1).flatten()
        lsb_b = np.bitwise_and(np_img[:,:,2], 1).flatten()

        # Combine LSBs to form bytes
        lsb_combined = np.packbits(np.column_stack((lsb_r, lsb_g, lsb_b)).flatten())

        # Convert bytes to string
        hidden_text = lsb_combined.tobytes().decode('utf-8', errors='ignore')

        # Remove non-printable characters and filter out short words
        words = ''.join(filter(lambda x: x.isprintable(), hidden_text)).split()
        meaningful_words = [word for word in words if len(word) >= 4]  # Only keep words with 4 or more characters

        # Join meaningful words and return
        result = ' '.join(meaningful_words)
        return result[:100] if result else None  # Return first 100 characters if any meaningful text is found

    except Exception as e:
        logger.error(f"Error during hidden text detection: {e}")
        return None

def detect_age(image_path):
    try:
        result = DeepFace.analyze(image_path, actions=['age'], enforce_detection=False)
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('age')
        else:
            logger.warning(f"No face detected for age estimation in image: {image_path}")
            return None
    except ValueError as e:
        if "Face could not be detected" in str(e):
            logger.warning(f"No face detected for age estimation in image: {image_path}")
            return None
        else:
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
                dominant_color = 'N/A'
                faces_found = 'N/A'
                stego_detected, stego_score, hidden_text = False, None, None
                age_estimation = 'N/A'
                frame_count = img_obj.n_frames
                duration = img_obj.info.get('duration', 'N/A')
                ai_generated, ai_score = False, None
                child_exploitation_detected, child_exploitation_details = False, None
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

            # Create a table row with the extracted information
            row = [
                original_url or image_path,
                image_hash,
                str(dominant_color),
                "Yes" if faces_found == True else "No" if faces_found == False else faces_found,
                f"{json.dumps(nudity_results['result'], indent=2)} (Score: {nudity_results['score']:.2f})",
                f"{'Yes' if stego_detected else 'No'} (Score: {stego_score:.2f})" if stego_score is not None else "N/A",
                str(age_estimation),
                hidden_text if hidden_text else "None detected",
                str(frame_count),
                str(duration),
                f"{'Yes' if ai_generated else 'No'} (Score: {ai_score:.2f})" if ai_score is not None else "N/A",
                "Yes" if child_exploitation_detected else "No",
                child_exploitation_details or "N/A",
                ocr_text,
                exif_summary
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

# Function to crawl a website and process images
def crawl(url, depth, visited=None, seen_images=None, results=None):
    """Crawl pages up to *depth*, processing images found on each page.

    Tracks visited pages (so link cycles terminate) and already-processed
    image URLs, stays on the starting page's host, and accumulates result
    rows across every page so the caller can report them once at the end.
    """
    if visited is None:
        visited = set()
    if seen_images is None:
        seen_images = set()
    if results is None:
        results = []

    url = url.split('#', 1)[0]  # fragments never change the page
    if depth <= 0 or url in visited:
        return results
    visited.add(url)

    try:
        headers = {'User-Agent': user_agent.random}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
    except requests.RequestException as e:
        logger.error(f"Error crawling {url}: {e}")
        return results

    # Extract image URLs, skipping any already processed on an earlier page
    image_urls = []
    for img_tag in soup.select('img[src]'):
        image_url = urljoin(url, img_tag['src'])
        if image_url not in seen_images:
            seen_images.add(image_url)
            image_urls.append(image_url)

    # Process images in parallel
    with ThreadPoolExecutor() as executor:
        with Progress() as progress:
            task = progress.add_task("[green]Processing images...", total=len(image_urls))
            futures = [executor.submit(process_image, image_url) for image_url in image_urls]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                progress.update(task, advance=1)

    # Follow same-host http(s) links only
    start_host = urlparse(url).netloc
    for link in soup.select('a[href]'):
        next_url = urljoin(url, link['href']).split('#', 1)[0]
        parsed = urlparse(next_url)
        if parsed.scheme not in ('http', 'https') or parsed.netloc != start_host:
            continue
        crawl(next_url, depth - 1, visited, seen_images, results)

    return results

def process_local_folder(folder_path):
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

def display_results(results):
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
        "EXIF Data"
    ]
    with open("results.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)  # Write header row
        writer.writerows(results)
    
    logger.info(f"Results saved to results.csv")

def main():
    parser = argparse.ArgumentParser(description="Suspicious Image Collection Kit -- SiCK v0.21")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--url", help="The URL to start crawling")
    group.add_argument("-f", "--folder", help="Local folder to scan for images")
    parser.add_argument("-d", "--depth", type=int, default=1, help="Depth of crawling (for URL mode)")
    parser.add_argument("-v", "--verbose", action='store_true', help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.url:
        results = crawl(args.url, args.depth)
    else:
        results = process_local_folder(args.folder)

    if results:
        display_results(results)

if __name__ == "__main__":
    main()
