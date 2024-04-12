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
import csv
import os

# Initialize Globals
console = Console()
user_agent = UserAgent()
nude_detector = NudeDetector()

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
                print(f"Content at {url} does not appear to be a valid image (missing or invalid Content-Type).")
                return None
        else:
            print(f"Error accessing URL {url}: Status code {response.status_code}")
            return None

    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
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

        # Handle PNG transparency
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        np_img = np.array(img)

        # LSB Variance Analysis
        lsb_array = np.bitwise_and(np_img.flatten(), 1)
        variance = np.var(lsb_array)

        # Adaptive Thresholding (example based on image size)
        image_size = img.size
        if image_size[0] * image_size[1] > 1000000:
            lsb_threshold = 0.27  # Adjust thresholds as needed
        else:
            lsb_threshold = 0.25

        # DCT Analysis (optional)
        dct_coeffs = dctn(np_img, norm='ortho')
        # ... (implement your DCT analysis and return a boolean result) ...
        dct_analysis_result = False  # Placeholder for DCT analysis result

        # Combine results
        if variance > lsb_threshold or dct_analysis_result:
            return True, variance
        else:
            return False, 0

    except Exception as e:
        print(f"Error during steganography detection in image: {image_path}. Error: {e}")
        return False, None

# Function to process an image and extract information
def process_image(image_url, verbose):
    image_path = download_image(image_url)
    if image_path:
        real_mime_type = magic.from_file(image_path, mime=True)
        if real_mime_type.startswith('image/'):
            try:
                with Image.open(image_path) as img_obj:
                    image_hash = get_image_hash(img_obj)
                    exif_info = extract_exif_data(img_obj)
                    detected_text = extract_text_from_image(img_obj)
                    dominant_color = get_dominant_color(image_path) if real_mime_type != 'image/gif' else 'N/A'
                    faces_found = detect_faces(img_obj) if real_mime_type != 'image/gif' else 'N/A'
                    nudity_results = detect_nudity(image_path)
                    stego_detected, variance = detect_steganography(image_path)

                    # Create a table row with the extracted information
                    row = [
                        image_url,
                        image_hash,
                        str(dominant_color),
                        "✅" if faces_found else "❌",
                        json.dumps(nudity_results, indent=2),
                        "✅" if stego_detected else "❌"
                    ]
                    return row
            except (UnidentifiedImageError, OSError) as error:
                if verbose:
                    console.print(f"[bold red]Error processing image at {image_url}[/bold red]: {error}", highlight=True)
            finally:
                os.remove(image_path)
        else:
            console.print(f"[bold red]Unsupported image MIME type {real_mime_type} encountered at URL:[/bold red] {image_url}", highlight=True)
            os.remove(image_path)
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

        # Create and display the table
        if results:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Image URL", style="cyan", overflow="fold")
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
                writer.writerow(table.columns) # Write header row
                writer.writerows(results)

        # Extract links for further crawling
        next_urls = [urljoin(url, link['href']) for link in soup.select('a[href]')]

        # Crawl next URLs recursively
        for next_url in next_urls:
            crawl(next_url, depth - 1, verbose)

    except requests.RequestException as e:
        print(f"Error crawling {url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Suspicious Image Collection Kit -- SiCK v0.20")
    parser.add_argument("url", help="The URL to start crawling")
    parser.add_argument("-d", "--depth", type=int, default=1, help="Depth of crawling")
    parser.add_argument("-v", "--verbose", action='store_true', help="Increase output verbosity (show errors)")
    args = parser.parse_args()

    crawl(args.url, args.depth, args.verbose)

if __name__ == "__main__":
    main()
