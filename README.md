# SiCK - Suspicious Image Collection Kit

![image](https://github.com/geeknik/sick/assets/466878/5fafb28d-2d2e-45d5-a98c-c44b1db4c1d6)


## Introduction
SiCK (Suspicious Image Collection Kit) is a Python tool designed for web scraping and analyzing images for suspicious content. It automates the process of downloading images, extracting text (OCR), reading EXIF data, detecting faces, and identifying dominant colors.

## Features
- Image downloading from given URLs.
- Generates a hash of the image.
- Optical Character Recognition (OCR) using pytesseract.
- EXIF data extraction from images.
- Face detection in images.
- Determining the dominant color of images.

## Installation
To install SiCK, clone the repository and install the required dependencies:
```bash
git clone https://github.com/geeknik/sick
cd sick
pip install -r requirements.txt
```

## Usage
Run SiCK using the following command:
```bash
python sick.py -d [DEPTH] [URL]
```
- `DEPTH`: Depth for web crawling.
- `URL`: The URL to start crawling.

## Dependencies
* BeautifulSoup
* PIL
* pytesseract
* numpy
* requests
* argparse
* hashlib
* magic
* cv2

## Contributing

Contributions to SiCK are welcome.

## License

SiCK is released under the GPLv3 License.

## Contact/Support

For support or other queries, contact [geeknik](https://x.com/geeknik) on X.

## Acknowledgments

Thanks to all the contributors and supporters of the SiCK project:
* that one guy who did that one thing that one time
