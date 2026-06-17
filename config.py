# config.py
import os

# YOLO Configuration
MODEL_TYPE = os.environ.get("MODEL_TYPE", "yolov8-onnx").lower()
MODEL_NAME = os.environ.get("MODEL_NAME", "yolov8s.onnx")

if MODEL_TYPE == "yolov3":
    YOLO_CFG = os.environ.get("YOLO_CFG", "yolov3.cfg")
    YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov3.weights")
elif MODEL_TYPE == "yolov3-tiny":
    YOLO_CFG = os.environ.get("YOLO_CFG", "yolov3-tiny.cfg")
    YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov3-tiny.weights")
elif MODEL_TYPE == "yolov8-onnx":
    YOLO_CFG = ""
    YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8s.onnx")
else:
    # Use empty/None config paths for non-Darknet models
    YOLO_CFG = os.environ.get("YOLO_CFG", "")
    YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "")

COCO_NAMES = os.environ.get("COCO_NAMES", "coco.names")

# Model Settings
INPUT_SIZE = (416, 416)
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.35))
NMS_THRESHOLD = float(os.environ.get("NMS_THRESHOLD", 0.45))
ENABLE_PREPROCESS = os.getenv('ENABLE_PREPROCESS', 'false').lower() == 'true'
API_KEY = os.getenv('API_KEY', '')

# Color Detection HSV Ranges
COLOR_RANGES = {
    'red': [(0, 100, 100), (10, 255, 255)],
    'orange': [(11, 100, 100), (25, 255, 255)],
    'yellow': [(26, 100, 100), (35, 255, 255)],
    'green': [(36, 100, 100), (85, 255, 255)],
    'blue': [(86, 100, 100), (125, 255, 255)],
    'purple': [(126, 100, 100), (150, 255, 255)],
    'pink': [(151, 100, 100), (170, 255, 255)],
    'brown': [(10, 100, 20), (20, 255, 200)],
    'black': [(0, 0, 0), (180, 255, 40)],
    'white': [(0, 0, 200), (180, 30, 255)],
    'gray': [(0, 0, 50), (180, 50, 200)]
}

COLOR_NAMES = {
    'red': 'red',
    'orange': 'orange',
    'yellow': 'yellow',
    'green': 'green',
    'blue': 'blue',
    'purple': 'purple',
    'pink': 'pink',
    'brown': 'brown',
    'black': 'black',
    'white': 'white',
    'gray': 'gray'
}

# Streamlit UI Config
PAGE_TITLE = "👁️ Vision Assistant for Visually Impaired Users"
PAGE_LAYOUT = "wide"

# TTS Settings
TTS_LANG = 'en'
TTS_SLOW = False

# Auto-description interval (seconds)
DESCRIPTION_INTERVAL = 10