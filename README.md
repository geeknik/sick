# SiCK - Suspicious Image Collection Kit

![image](https://github.com/geeknik/sick/assets/466878/5fafb28d-2d2e-45d5-a98c-c44b1db4c1d6)


## Introduction
SiCK (Suspicious Image Collection Kit) is a Python tool designed for analyzing images scraped from the web for suspicious content. It automates the process of downloading images, extracting text (OCR), reading EXIF data, detecting faces & nudity, and identifying dominant colors.

## Features
- Crawls a user-defined URL & depth with a page cap, visited-page tracking, and image de-duplication across pages; crawl scope is anchored to the start URL's host (redirected landings have their images analyzed, but foreign links are never followed)
- Image hashing: exact-content MD5 plus 64-bit dHash perceptual hashing with near-duplicate detection across a scan (resize/recompress/brightness-tolerant; crops, mirrors and rotations are not matched, GIFs hash from the first frame, flat images are excluded)
- OCR text extraction with pytesseract: grayscale + autocontrast + 3x upscaling of small images (watermark-class text the raw pipeline misses) and per-word confidence gating (conf >= 60)
- EXIF data extraction with GPS coordinates decoded to signed decimal degrees (first in the report cell, so they survive truncation), Exif-IFD sub-tags surfaced (DateTimeOriginal &c.) and MakerNote blobs abbreviated
- Face detection with OpenCV: frontal and BOTH profile orientations (flip-and-retranslate) with IoU overlap de-duplication; the Faces Detected column reports the count
- Nudity detection with NudeNet
- Per-frame analysis of GIF animations (nudity on up to 4 frames sampled across the animation, plus faces, age estimation and dominant color on the first frame; the Duration column reports the per-frame delay in ms, not total animation time)
- Age estimation with DeepFace: reports the youngest age across all detected faces (the screen a child standing beside an adult must not miss); whole-image fallback "ages" from faceless frames are discarded
- Determines dominant color
- Detects LSB steganography via RS analysis (Fridrich et al. 2001) and WS analysis (Ker & Böhme 2008) with quantitative payload-rate estimation
- Hidden payload extraction: scans 32 LSB embedding orders (R/G/B/interleaved planes, row/column traversal, MSB/LSB bit order, bit planes 1-2) for text and embedded file signatures (ZIP/PNG/JPEG/RAR/7z/PDF/gzip/bzip2/PEM)
- AI-generated image detection: generator provenance scanning (Stable Diffusion / ComfyUI / NovelAI parameter chunks, generator software tags, IPTC trainedAlgorithmicMedia, C2PA content credentials) plus Fourier-domain heuristics (GAN grid artifacts, diffusion-style high-frequency deficit)

## Installation
**Step 1: Install Python 3.10+**
If you don't have Python 3.10 or newer installed, you need to install it first. You can download it from the [official Python website](https://www.python.org/downloads/) or use a version manager like `pyenv`.

**Step 2: Clone the repo**
To continue installing SiCK, clone the repository and install the required dependencies:
```bash
git clone https://github.com/geeknik/sick
cd sick
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Testing
Run the unit test suite with:
```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

## Usage
Run SiCK against a URL or a local folder:
```bash
python sick.py -u [URL] (-d [DEPTH]) (-m [MAX_PAGES]) (-o [OUTPUT]) (-v)
python sick.py -f [FOLDER] (-o [OUTPUT]) (-v)
```
- `URL`: The URL to start crawling (required, unless `-f` is used)
- `FOLDER`: Local folder to scan for images (required, unless `-u` is used)
- `DEPTH`: Depth for web crawling (optional, default 1)
- `MAX_PAGES`: Maximum number of pages to fetch while crawling (optional, default 100)
- `OUTPUT`: path for the CSV report (optional, default `results.csv` in the current directory)
- `VERBOSE`: enable debug logging (optional)

CSV cells that could execute as formulas when opened in a spreadsheet (leading `=`, `+`, `-`, `@`) are neutralized with an apostrophe prefix.

## Contributing

Contributions to SiCK are welcome.

## License

SiCK is released under the MIT License.

## Acknowledgments

Thanks to all the contributors and supporters of the SiCK project:
* that one guy who did this one thing that one time
* those other two guys who will always remain nameless
* the faceless shadow outside my window
* the heavy breathing and clicking on phone
* hi!~
