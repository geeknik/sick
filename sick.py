from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import ExifTags, UnidentifiedImageError
from fake_useragent import UserAgent
from rich.progress import Progress
from scipy.stats import binomtest
from colorthief import ColorThief
from nudenet import NudeDetector
from urllib.parse import urljoin
from rich.console import Console
from  bs4 import BeautifulSoup
from rich.table import Table
from scipy.fft import dctn
from scipy import stats
from PIL import Image
from deepface import DeepFace
import pytesseract
import numpy as np
import tempfile
import requests
import argparse
import hashlib
import magic
import json
import cv2
import csv
import os
import logging
import base64
from pathlib import Path
from scipy.fft import dctn
from scipy import stats

# Initialize Globals
console = Console()
user_agent = UserAgent()
nude_detector = NudeDetector()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rs_analysis(lsb_array):
    lsb_array = lsb_array.astype(np.int64) # Cast to larger data type
    # Calculate RS values for adjacent pixel pairs
    rs_values = [abs(lsb_array[i] - lsb_array[i+1]) for i in range(len(lsb_array) - 1)]
    # Analyze the frequency distribution of RS values
    # You can use chi-square test or other statistical methods here
    # For simplicity, we'll just calculate the mean
    mean_rs = np.mean(rs_values)
    threshold = 0.45  # Adjust based on experimentation
    return mean_rs > threshold

# Function to detect nudity in an image
def detect_nudity(image_path):
    try:
        # Check if the image is a GIF
        if image_path.lower().endswith('.gif'):
            return ["GIF images are not supported for nudity detection"]
        
        detections = nude_detector.detect(image_path)
        nudity_related_classes = {
            'EXPOSED_ANUS', 'EXPOSED_BUTTOCKS', 'EXPOSED_BREAST',
            'EXPOSED_GENITALIA', 'MALE_GENITALIA_EXPOSED', 'FEMALE_GENITALIA_EXPOSED'
        }
        nudity_detections = [
            detection for detection in detections
            if detection.get('class') in nudity_related_classes and detection.get('score', 0) >= 0.5
        ]
        return nudity_detections
    except Exception as e:
        logger.error(f"Error during nudity detection in image: {image_path}. Error: {e}")
        return []

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
    cv_image = np.array(pil_image)
    if len(cv_image.shape) == 2:
        gray_image = cv_image
    elif cv_image.shape[2] == 4:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    return len(faces) > 0

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
                correct_extension = real_mime_type.split('/')[1].lower()
                new_filename = f'{f.name}.{correct_extension}'
                os.rename(f.name, new_filename)
                return new_filename
        else:
            headers = {'User-Agent': user_agent.random}
            response = requests.head(url, headers=headers)  # Check content type first

            if response.status_code == 200:  # Check for successful response
                content_type = response.headers.get('Content-Type')
                if content_type and content_type.startswith('image/'):
                    response = requests.get(url, timeout=10, headers=headers)
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as f:
                        f.write(response.content)
                        f.flush()
                        real_mime_type = magic.from_file(f.name, mime=True)
                        correct_extension = real_mime_type.split('/')[1].lower()
                        new_filename = f'{f.name}.{correct_extension}'
                        os.rename(f.name, new_filename)
                        return new_filename
                else:
                    logger.warning(f"Content at {url} does not appear to be a valid image (missing or invalid Content-Type).")
                    return None
            else:
                logger.warning(f"Error accessing URL {url}: Status code {response.status_code}")
                return None

    except requests.RequestException as e:
        logger.error(f"Error downloading image: {e}")
        return None

# Function to get the dominant color of an image
def get_dominant_color(image_path):
    try:
        color_thief = ColorThief(image_path)
        dominant_color = color_thief.get_color(quality=1)
        return dominant_color
    except Exception as e:
        print(f"Error getting dominant color: {image_path}. Error: {e}")
    return None

# Function to detect steganography in an image
def detect_steganography(image_path):
    try:
        img = Image.open(image_path)

        # Handle PNG transparency and convert to RGB
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
            img = bg

        np_img = np.array(img).astype(np.uint8)

        # Ensure the image is 3D (for RGB channels)
        if len(np_img.shape) == 2:
            np_img = np.stack((np_img,) * 3, axis=-1)

        # LSB Analysis
        lsb_array = np.bitwise_and(np_img.flatten(), 1)
        lsb_variance = np.var(lsb_array)

        # Chi-Square Test
        observed_freq = np.bincount(lsb_array, minlength=2)
        expected_freq = np.sum(observed_freq) / 2
        chi_square_stat = np.sum((observed_freq - expected_freq)**2 / expected_freq)
        chi_square_p_value = 1 - stats.chi2.cdf(chi_square_stat, 1)

        # Sample Pair Analysis
        sample_pairs = np.column_stack((lsb_array[::2], lsb_array[1::2]))
        sp_chi_square_stat = np.sum((np.bincount(sample_pairs.dot([1, 2])) - len(sample_pairs)/4)**2) / (len(sample_pairs)/4)
        sp_chi_square_p_value = 1 - stats.chi2.cdf(sp_chi_square_stat, 3)

        # RS Analysis
        rs_result = rs_analysis(lsb_array)

        # DCT Analysis
        dct_coeffs = dctn(np_img[:,:,0], norm='ortho')  # Analyze first channel
        dct_lsb = np.bitwise_and(dct_coeffs.flatten().astype(np.int64), 1)
        dct_chi_square_stat = np.sum((np.bincount(dct_lsb, minlength=2) - len(dct_lsb)/2)**2) / (len(dct_lsb)/2)
        dct_chi_square_p_value = 1 - stats.chi2.cdf(dct_chi_square_stat, 1)

        # Combine results
        stego_score = (
            (lsb_variance > 0.25) +
            (chi_square_p_value < 0.05) +
            (sp_chi_square_p_value < 0.05) +
            rs_result +
            (dct_chi_square_p_value < 0.05)
        )

        # Detect hidden text
        hidden_text = detect_hidden_text(np_img)

        return stego_score >= 2, stego_score, hidden_text  # Consider it suspicious if 2 or more tests indicate steganography

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

        # Remove non-printable characters
        hidden_text = ''.join(filter(lambda x: x.isprintable(), hidden_text))

        # Return first 100 characters if any text is found
        return hidden_text[:100] if hidden_text.strip() else None

    except Exception as e:
        logger.error(f"Error during hidden text detection: {e}")
        return None

def detect_age(image_path):
    try:
        result = DeepFace.analyze(image_path, actions=['age'])
        return result[0]['age']
    except Exception as e:
        logger.error(f"Error during age detection in image: {image_path}. Error: {e}")
        return None

# Function to process an image and extract information
def process_image(image_path, verbose):
    downloaded_file = None
    original_url = image_path if not os.path.isfile(image_path) else None
    try:
        if not os.path.isfile(image_path):
            downloaded_file = download_image(image_path)
            if not downloaded_file:
                return None
            image_path = downloaded_file

        real_mime_type = magic.from_file(image_path, mime=True)
        if not real_mime_type.startswith('image/'):
            logger.warning(f"Unsupported file type {real_mime_type} encountered at: {image_path}")
            return None

        if real_mime_type == 'image/svg+xml':
            logger.warning(f"SVG file encountered at: {image_path}. SVG processing is not supported.")
            return None

        with Image.open(image_path) as img_obj:
            image_hash = get_image_hash(img_obj)
            exif_info = extract_exif_data(img_obj)
            detected_text = extract_text_from_image(img_obj)
            dominant_color = get_dominant_color(image_path) if real_mime_type != 'image/gif' else 'N/A'
            faces_found = detect_faces(img_obj) if real_mime_type != 'image/gif' else 'N/A'
            nudity_results = detect_nudity(image_path)
            stego_detected, stego_score, hidden_text = detect_steganography(image_path) if real_mime_type != 'image/gif' else (False, None, None)
            age_estimation = detect_age(image_path) if real_mime_type != 'image/gif' and faces_found else 'N/A'

            # Create a table row with the extracted information
            row = [
                original_url or image_path,
                image_hash,
                str(dominant_color),
                "Yes" if faces_found else "No",
                json.dumps(nudity_results, indent=2),
                f"{'Yes' if stego_detected else 'No'} (Score: {stego_score})" if stego_score is not None else "N/A",
                str(age_estimation),
                hidden_text if hidden_text else "None detected"
            ]
            return row
    except (UnidentifiedImageError, OSError) as error:
        if verbose:
            logger.error(f"Error processing image at {image_path}: {error}")
    finally:
        if downloaded_file and os.path.exists(downloaded_file):
            os.remove(downloaded_file)
    return None

# Function to crawl a website and process images
def crawl(url, depth, verbose):
    if depth <= 0:
        return

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract image URLs
        image_urls = [urljoin(url, img_tag['src']) for img_tag in soup.select('img[src]')]

        # Process images in parallel
        with ThreadPoolExecutor() as executor:
            with Progress() as progress:
                task = progress.add_task("[green]Processing images...", total=len(image_urls))
                futures = [executor.submit(process_image, image_url, verbose) for image_url in image_urls]
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                    progress.update(task, advance=1)

        # Display and save results
        if results:
            display_results(results)

        # Extract links for further crawling
        next_urls = [urljoin(url, link['href']) for link in soup.select('a[href]')]

        # Crawl next URLs recursively
        for next_url in next_urls:
            crawl(next_url, depth - 1, verbose)

    except requests.RequestException as e:
        print(f"Error crawling {url}: {e}")

def process_local_folder(folder_path, verbose):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    results = []
    
    with Progress() as progress:
        files = list(Path(folder_path).rglob('*'))
        task = progress.add_task("[green]Processing local images...", total=len(files))
        
        for file_path in files:
            if file_path.suffix.lower() in image_extensions:
                try:
                    result = process_image(str(file_path), verbose)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
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
        "Hidden Text"
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
    parser.add_argument("-v", "--verbose", action='store_true', help="Increase output verbosity (show errors)")
    args = parser.parse_args()

    if args.url:
        crawl(args.url, args.depth, args.verbose)
    elif args.folder:
        results = process_local_folder(args.folder, args.verbose)
        if results:
            display_results(results)

if __name__ == "__main__":
    main()
def display_results(results):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Image Path/URL", style="cyan", overflow="fold")
    table.add_column("Hash", style="yellow", overflow="fold")
    table.add_column("Dominant Color", style="green")
    table.add_column("Faces Detected", style="bright_white")
    table.add_column("Nudity Assessment", style="red", no_wrap=True)
    table.add_column("Steganography Detected", style="bright_white", overflow="fold")
    for row in results:
        table.add_row(*row)
    console.print("\nProcessed Image Information:")
    console.print(table)

    # Save results to CSV
    with open("results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(table.columns)  # Write header row
        writer.writerows(results)
    
    logger.info(f"Results saved to results.csv")
