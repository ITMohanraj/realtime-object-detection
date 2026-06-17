# object_detector.py
import cv2
import numpy as np
import config

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class ObjectDetector:
    def __init__(self):
        """Initialize YOLO object detector"""
        if HAS_TORCH:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        
        # Check if using modern Ultralytics YOLO models (v8, v11, etc.)
        is_ultralytics = (
            config.MODEL_TYPE in ["yolov8", "yolov11", "yolo11", "ultralytics"] or
            "yolov8" in config.MODEL_NAME or
            "yolo11" in config.MODEL_NAME
        ) and config.MODEL_TYPE != "yolov8-onnx"
        
        if config.MODEL_TYPE == "yolov8-onnx":
            self.net = cv2.dnn.readNetFromONNX(config.YOLO_WEIGHTS)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif is_ultralytics:
            if not HAS_TORCH:
                raise RuntimeError("PyTorch (torch) is not installed. Cannot load Ultralytics YOLO model.")
            try:
                from ultralytics import YOLO
                model_file = config.MODEL_NAME
                if not model_file.endswith('.pt'):
                    model_file += '.pt'
                self.model = YOLO(model_file)
                self.model.to(self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load Ultralytics YOLO model: {e}")
        elif config.MODEL_NAME.startswith("yolov5"):
            if not HAS_TORCH:
                raise RuntimeError("PyTorch (torch) is not installed. Cannot load YOLOv5 model.")
            try:
                # Load Ultralytics YOLOv5 model (PyTorch Hub fallback)
                self.model = torch.hub.load('ultralytics/yolov5', config.MODEL_NAME, pretrained=True)
                self.model.to(self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLOv5 via PyTorch Hub: {e}")
        else:
            # Legacy OpenCV DNN loading
            self.net = cv2.dnn.readNetFromDarknet(config.YOLO_CFG, config.YOLO_WEIGHTS)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load classes (prioritize model classes if available)
        if hasattr(self, 'model'):
            if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
                self.classes = [self.model.names[i] for i in sorted(self.model.names.keys())]
            else:
                with open(config.COCO_NAMES, "r") as f:
                    self.classes = f.read().strip().split("\n")
        else:
            with open(config.COCO_NAMES, "r") as f:
                self.classes = f.read().strip().split("\n")
        
        # Random colors for bounding boxes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Get output layer names for OpenCV DNN
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
        
        if config.MODEL_TYPE == "yolov8-onnx":
            blob = cv2.dnn.blobFromImage(
                image,
                1/255.0,
                (640, 640),
                swapRB=True,
                crop=False
            )
            self.net.setInput(blob)
            outputs = self.net.forward()
            
            output = outputs[0]
            if len(output.shape) == 3:
                output = output[0]
            output = output.T # Transpose to (8400, 84)
            
            # Vectorized NumPy filtering for massive speedup
            boxes_raw = output[:, 0:4]
            scores_raw = output[:, 4:]
            
            class_ids_arr = np.argmax(scores_raw, axis=1)
            confidences_arr = np.max(scores_raw, axis=1)
            
            mask = confidences_arr > confidence_threshold
            filtered_boxes = boxes_raw[mask]
            filtered_confidences = confidences_arr[mask]
            filtered_class_ids = class_ids_arr[mask]
            
            boxes = []
            confidences = []
            class_ids = []
            
            x_factor = width / 640.0
            y_factor = height / 640.0
            
            for box, conf, cls_id in zip(filtered_boxes, filtered_confidences, filtered_class_ids):
                x_center, y_center, w, h = box
                left = int((x_center - w / 2) * x_factor)
                top = int((y_center - h / 2) * y_factor)
                box_width = int(w * x_factor)
                box_height = int(h * y_factor)
                
                boxes.append([left, top, box_width, box_height])
                confidences.append(float(conf))
                class_ids.append(int(cls_id))
                    
            if boxes:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
                indexes = indexes.flatten() if len(indexes) > 0 else []
            else:
                indexes = []
        elif hasattr(self, 'model'):
            # Check if it is an Ultralytics model (e.g. YOLOv8 / YOLOv11)
            if hasattr(self.model, 'predict'):
                results = self.model(image, conf=confidence_threshold, iou=nms_threshold, verbose=False)
                result = results[0]
                boxes, confidences, class_ids = [], [], []
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().item())
                    cls = int(box.cls[0].cpu().item())
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(conf)
                    class_ids.append(cls)
                indexes = list(range(len(boxes)))
            else:
                # YOLOv5 PyTorch Hub model
                self.model.conf = confidence_threshold
                self.model.iou = nms_threshold
                results = self.model(image)
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
                config.INPUT_SIZE,
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