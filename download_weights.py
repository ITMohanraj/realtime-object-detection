# download_weights.py
import os
import sys
import urllib.request

# Configuration URLs
YOLO_TINY_CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
YOLO_TINY_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3-tiny.weights"
YOLO_FULL_CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
YOLO_FULL_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = min(100.0, readsofar * 100.0 / totalsize)
        sys.stdout.write(f"\rDownloading... {percent:3.1f}% ({readsofar // (1024*1024)}MB / {totalsize // (1024*1024)}MB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rRead {readsofar} bytes")
        sys.stdout.flush()

def download_file(url, filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"'{filename}' already exists and is not empty. Skipping download.")
        return True
    
    print(f"Downloading {url} -> {filename}...")
    try:
        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(filename)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
            
        urllib.request.urlretrieve(url, filename, reporthook)
        print(f"\nSuccessfully downloaded '{filename}'.")
        return True
    except Exception as e:
        print(f"\nError downloading '{filename}' from {url}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def main():
    model_type = os.environ.get("MODEL_TYPE", "yolov3-tiny").lower()
    download_full = os.environ.get("DOWNLOAD_FULL_YOLO", "false").lower() == "true" or model_type == "yolov3"
    
    print(f"Starting weights downloader. Target Model: {model_type}")
    
    success = True
    
    # 1. Download COCO Names (required by both)
    if not download_file(COCO_NAMES_URL, "coco.names"):
        success = False
        
    # 2. Download YOLOv3-Tiny Files (always download as backup/default)
    if not download_file(YOLO_TINY_CFG_URL, "yolov3-tiny.cfg"):
        success = False
    if not download_file(YOLO_TINY_WEIGHTS_URL, "yolov3-tiny.weights"):
        success = False
        
    # 3. Download Full YOLOv3 Files (if requested)
    if download_full:
        print("Downloading standard YOLOv3 model files (large)...")
        if not download_file(YOLO_FULL_CFG_URL, "yolov3.cfg"):
            success = False
        if not download_file(YOLO_FULL_WEIGHTS_URL, "yolov3.weights"):
            success = False
            
    if not success:
        print("\nError: Some downloads failed.")
        sys.exit(1)
    else:
        print("\nAll required model files are ready.")

if __name__ == "__main__":
    main()
