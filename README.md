# SiCK - Suspicious Image Collection Kit

![image](https://github.com/geeknik/sick/assets/466878/5fafb28d-2d2e-45d5-a98c-c44b1db4c1d6)


## Introduction
SiCK (Suspicious Image Collection Kit) is a Python tool designed for analyzing images scraped from the web for suspicious content. It automates the process of downloading images, extracting text (OCR), reading EXIF data, detecting faces & nudity, and identifying dominant colors.

## Features
- Crawls a user-defined URL & depth
- Image Hash Generation
- OCR text extraction with pytesseract
- EXIF data extraction
- Face detection with OpenCV
- Nudity detection with NudeNet
- Age estimation with DeepFace
- Determines dominant color
- Detects LSB steganography via RS analysis (Fridrich et al. 2001) and WS analysis (Ker & Böhme 2008) with quantitative payload-rate estimation
- AI-generated image heuristics

## Installation
**Step 1: Install Python 3.8**
If you don't have Python 3.8 installed, you need to install it first. You can download it from the [official Python website](https://www.python.org/downloads/release/python-380/) or use a version manager like `pyenv`.

**Step 2: Clone the repo**
To continue installing SiCK, clone the repository and install the required dependencies:
```bash
git clone https://github.com/geeknik/sick
cd sick
python3.8 -m venv env
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
python sick.py -u [URL] (-d [DEPTH]) (-v)
python sick.py -f [FOLDER] (-v)
```
- `URL`: The URL to start crawling (required, unless `-f` is used)
- `FOLDER`: Local folder to scan for images (required, unless `-u` is used)
- `DEPTH`: Depth for web crawling (optional, default 1)
- `VERBOSE`: enable debug logging (optional)

![Screenshot_09:34:17_12-04-2024](https://github.com/geeknik/sick/assets/466878/ae52d61a-f37a-47f1-8b33-76368fba3105)

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
