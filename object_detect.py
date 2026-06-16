# object_detector.py
import cv2
import numpy as np
import config
import torch
from config import YOLO_CFG, YOLO_WEIGHTS, COCO_NAMES, INPUT_SIZE, MODEL_NAME

class ObjectDetector:
    def __init__(self):
        """Initialize YOLO object detector"""
        if MODEL_NAME.startswith("yolov5") or MODEL_NAME.startswith("yolov8"):
            # Load Ultralytics YOLO model (PyTorch)
            self.model = torch.hub.load('ultralytics/yolov5', MODEL_NAME, pretrained=True)
            self.model.conf = 0.25  # default confidence
            self.model.iou = 0.45   # default NMS IoU
            self.device = torch.device('cpu')
            self.model.to(self.device)
        else:
            # Legacy OpenCV DNN loading
            self.net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load COCO classes
        with open(COCO_NAMES, "r") as f:
            self.classes = f.read().strip().split("\n")
        
        # Random colors for bounding boxes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Get output layer names
        if hasattr(self, 'net'):
            self.output_layers = self.net.getUnconnectedOutLayersNames()
        else:
            self.output_layers = []
    
    def detect_objects(self, image, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Detect objects in an image using YOLO
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
        
        Returns:
            Tuple of (boxes, confidences, class_ids, indexes)
        """
        height, width = image.shape[:2]
        
        # Prepare input blob
        # Optional preprocessing: apply CLAHE to improve visibility in low‑contrast scenes
        if getattr(config, "ENABLE_PREPROCESS", False):
            # Convert to YCrCb, apply CLAHE on the luminance channel, then back to BGR
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y_eq = clahe.apply(y)
            ycrcb_eq = cv2.merge((y_eq, cr, cb))
            image = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        
        if hasattr(self, 'model'):
            # Ultralytics model inference
            results = self.model(image)
            # results.xyxy[0] -> (x1, y1, x2, y2, conf, cls)
            boxes, confidences, class_ids = [], [], []
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(conf))
                class_ids.append(int(cls))
            indexes = list(range(len(boxes)))
        else:
            # Existing OpenCV DNN pipeline (unchanged)
            blob = cv2.dnn.blobFromImage(
                image,
                1/255.0,
                INPUT_SIZE,
                swapRB=True,
                crop=False
            )
            self.net.setInput(blob)
            layer_outputs = self.net.forward(self.output_layers)
            boxes = []
            confidences = []
            class_ids = []
            height, width = image.shape[:2]
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            if boxes:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
                indexes = indexes.flatten() if len(indexes) > 0 else []
            else:
                indexes = []
        
        return boxes, confidences, class_ids, indexes
    
    def get_class_name(self, class_id):
        """Get class name from class ID"""
        return self.classes[class_id] if class_id < len(self.classes) else "unknown"
    
    def get_class_color(self, class_id):
        """Get color for bounding box from class ID"""
        return self.colors[class_id] if class_id < len(self.colors) else (0, 255, 0)