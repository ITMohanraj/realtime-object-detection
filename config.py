# config.py
import numpy as np
import os

# YOLO Configuration
MODEL_TYPE = os.environ.get("MODEL_TYPE", "yolov3-tiny").lower()

if MODEL_TYPE == "yolov3":
    YOLO_CFG = os.environ.get("YOLO_CFG", "yolov3.cfg")
    YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov3.weights")
else:
    YOLO_CFG = os.environ.get("YOLO_CFG", "yolov3-tiny.cfg")
    YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov3-tiny.weights")

COCO_NAMES = os.environ.get("COCO_NAMES", "coco.names")

# Model Settings
INPUT_SIZE = (320, 320)
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5))
NMS_THRESHOLD = float(os.environ.get("NMS_THRESHOLD", 0.4))

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