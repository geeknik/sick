from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import ExifTags, UnidentifiedImageError, ImageChops
import imagehash
from fake_useragent import UserAgent
from rich.progress import Progress
from scipy.stats import binomtest
from colorthief import ColorThief
from nudenet import NudeDetector
from urllib.parse import urljoin
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

def detect_ai_generated(image_path):
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale
            img_gray = img.convert('L')
            
            # Compute the difference hash
            dhash = imagehash.dhash(img_gray)
            
            # Compute the wavelet hash
            whash = imagehash.whash(img_gray)
            
            # Compute the perceptual hash
            phash = imagehash.phash(img_gray)
            
            # Check for unnaturally smooth gradients
            edges = img_gray.filter(ImageFilter.FIND_EDGES)
            edge_ratio = np.sum(np.array(edges) > 0) / (img_gray.width * img_gray.height)
            
            # Combine metrics (you may need to adjust these thresholds based on experimentation)
            ai_score = (dhash.hash.sum() / 256 + whash.hash.sum() / 256 + phash.hash.sum() / 256) / 3
            ai_score = (ai_score + (1 - edge_ratio)) / 2  # Incorporate edge information
            
            return ai_score > 0.6, ai_score  # Threshold can be adjusted

    except Exception as e:
        logger.error(f"Error during AI generation detection in image: {image_path}. Error: {e}")
        return False, None

# Function to get the dominant color of an image
def get_dominant_color(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            color_thief = ColorThief(img)
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
    try:
        with Image.open(image_path) as img:
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

            # Entropy Analysis
            entropy = stats.entropy(np.bincount(np_img.flatten()))
            max_entropy = np.log2(256)  # Maximum possible entropy for 8-bit image
            normalized_entropy = entropy / max_entropy

            # Combine results with adjusted thresholds and weights
            stego_score = (
                0.2 * (lsb_variance > 0.3) +
                0.2 * (chi_square_p_value < 0.01) +
                0.2 * (sp_chi_square_p_value < 0.01) +
                0.2 * rs_result +
                0.1 * (dct_chi_square_p_value < 0.01) +
                0.1 * (normalized_entropy > 0.9)  # High entropy might indicate hidden data
            )

            # Detect hidden text
            hidden_text = detect_hidden_text(np_img)

            return stego_score >= 0.6, stego_score, hidden_text  # Adjusted threshold

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
        return result[0]['age']
    except Exception as e:
        logger.error(f"Error during age detection in image: {image_path}. Error: {e}")
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
def process_image(image_path, verbose):
    downloaded_file = None
    original_url = image_path if not os.path.isfile(image_path) else None
    try:
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
            exif_info = extract_exif_data(img_obj)
            detected_text = extract_text_from_image(img_obj)
            
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
                child_exploitation_details or "N/A"
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
        logger.error(f"Error crawling {url}: {e}")

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
    table.add_column("Frame Count", style="bright_magenta")
    table.add_column("Duration", style="bright_cyan")
    table.add_column("AI Generated", style="bright_green")
    table.add_column("Child Exploitation", style="red")
    table.add_column("Child Exploitation Details", style="red", overflow="fold")
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
        "Child Exploitation Details"
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
