from fake_useragent import UserAgent
from colorthief import ColorThief
from nudenet import NudeDetector
from urllib.parse import urljoin
from rich.console import Console
from bs4 import BeautifulSoup
from rich.table import Table
from PIL import ExifTags
from rich import print
from PIL import Image
import pytesseract
import numpy as np
import tempfile
import requests
import argparse
import hashlib
import magic
import json
import cv2
import os

# Intialize Globals
console = Console()
user_agent = UserAgent()
nude_detector = NudeDetector()

# Example on how to get a random user-agent string
random_user_agent = user_agent.random # can substitute user_agent.chrome; user_agent.firefox; user_agent.safari

def detect_nudity(image_path):
    try:
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
        print(f"Error during nudity detection in image: {image_path}. Error: {e}")
        return []

def get_image_hash(pil_image):
    # Generate a hash for an image
    image_bytes = pil_image.tobytes()
    md5_hash = hashlib.md5(image_bytes)
    return md5_hash.hexdigest()

def extract_text_from_image(pil_image):
    # Use OCR to extract text from an image
    text = pytesseract.image_to_string(pil_image)
    return text

def extract_exif_data(pil_image):
    # Extract EXIF data from an image if available
    exif_data = {}
    if hasattr(pil_image, 'getexif'):  # Check if getexif method is available for the image
        exif_info = pil_image.getexif()
        if exif_info is not None:
            for tag, value in exif_info.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif_data[decoded] = value
    return exif_data

def detect_faces(pil_image):
    # Convert PIL Image to appropriate format for OpenCV
    cv_image = np.array(pil_image)
    if len(cv_image.shape) == 2:
        gray_image = cv_image
    elif cv_image.shape[2] == 4:
        # Convert RGBA to RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    return len(faces) > 0

def download_image(url):
    # Download images with random user agent
    try:
        headers = {'User-Agent': user_agent.random}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as f:
            f.write(response.content)
            f.flush()  # Ensure all data is written to disk
            real_mime_type = magic.from_file(f.name, mime=True)
            if real_mime_type.startswith('image/'):
                correct_extension = real_mime_type.split('/')[1].lower()
                new_filename = f'{f.name}.{correct_extension}'
                os.rename(f.name, new_filename)
                return new_filename
            else:
                print(f"Content at {url} does not appear to be a valid image based on MIME type: {real_mime_type}")
                return None
    except (requests.RequestException, IOError) as e:
        print(f"Error downloading image: {e}")
        return None

def get_dominant_color(image_path):
    # Identify the domminant color in the image
    try:
        color_thief = ColorThief(image_path)
        dominant_color = color_thief.get_color(quality=1)
        return dominant_color
    except Exception as e:
        print(f"Error getting dominant color. It may not be possible to determine dominant color for a 1x1 or overly simplistic image: {image_path}. Error: {e}")
    return None

def crawl(url, depth, verbose):
    # Crawling the website
    if depth <= 0:
        return
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for img_tag in soup.select('img[src]'):
            image_url = urljoin(url, img_tag['src'])
            image_path = download_image(image_url)
            if image_path:
                real_mime_type = magic.from_file(image_path, mime=True)
                if real_mime_type.startswith('image/'):
                    # Create a fancy table for console log
                    try:
                        table = Table(show_header=True, header_style="bold magenta")
                        table.add_column("Image URL", no_wrap=True)
                        table.add_column("Hash")
                        table.add_column("Dominant Color")
                        table.add_column("Faces Detected")
                        table.add_column("Nudity", no_wrap=True)
                        with Image.open(image_path) as img_obj:
                            image_hash = get_image_hash(img_obj)
                            exif_data = extract_exif_data(img_obj)
                            text = extract_text_from_image(img_obj)
                            dominant_color = get_dominant_color(image_path) if real_mime_type != 'image/gif' else 'N/A'
                            faces_detected = detect_faces(img_obj) if real_mime_type != 'image/gif' else 'N/A'
                            nudity_detections = detect_nudity(image_path)
                            # Output the details including nudity detection results
                            table.add_row(
                              image_url,
                              image_hash,
                              str(dominant_color),
                              "Yes" if faces_detected else "No",
                              json.dumps(nudity_detections)
                            )
                        console.print(table)
                    except (Image.UnidentifiedImageError, OSError) as e:
                        if verbose:
                            console.print(f"Cannot process image at {image_url}: {e}")
                    finally:
                        os.remove(image_path)  # Clean up temp image file
                else:
                    print(f"Unsupported image MIME type {real_mime_type} at URL {image_url}")
                    os.remove(image_path)
        for link in soup.select('a[href]'):
            next_url = urljoin(url, link['href'])
            crawl(next_url, depth - 1)
    except requests.RequestException as e:
        print(f"Error crawling {url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Suspicious Image Collection Kit -- SiCK v0.14")
    parser.add_argument("url", help="The URL to start crawling")
    parser.add_argument("-d", "--depth", type=int, default=1, help="Depth of crawling")
    parser.add_argument("-v", "--verbose", action='store_true', help="Increase output verbosity (show errors)")
    args = parser.parse_args()

    crawl(args.url, args.depth, args.verbose)

if __name__ == "__main__":
    main()
