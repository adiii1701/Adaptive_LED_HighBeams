import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / 'models'
PROTOTXT_URL = 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt'
CAFFEMODEL_URL = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel'

def download(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Exists: {dest}")
        return
    print(f"Downloading {url} -> {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"Saved: {dest}")

def main():
    prototxt_path = MODELS_DIR / 'deploy.prototxt'
    caffemodel_path = MODELS_DIR / 'mobilenet_iter_73000.caffemodel'
    download(PROTOTXT_URL, prototxt_path)
    download(CAFFEMODEL_URL, caffemodel_path)
    print("Model files are ready.")

if __name__ == '__main__':
    main()


