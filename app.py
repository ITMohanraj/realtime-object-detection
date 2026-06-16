# app.py
import base64
import logging
import os
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from collections import defaultdict

# Import local modules
import config
from object_detect import ObjectDetector
from color_detector import ColorDetector
from scene_descriptor import SceneDescriptor
from text_to_speech import TextToSpeech

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("VisionAssistant")

# Validate weights and config files are present
def verify_model_files():
    cfg_exists = os.path.exists(config.YOLO_CFG)
    weights_exists = os.path.exists(config.YOLO_WEIGHTS)
    coco_exists = os.path.exists(config.COCO_NAMES)
    
    if not (cfg_exists and weights_exists and coco_exists):
        logger.warning(
            f"Missing YOLO model files. CFG: {config.YOLO_CFG} ({cfg_exists}), "
            f"WEIGHTS: {config.YOLO_WEIGHTS} ({weights_exists}), "
            f"COCO: {config.COCO_NAMES} ({coco_exists}). "
            "Attempting to download files..."
        )
        try:
            from download_weights import main as download_main
            download_main()
        except Exception as e:
            logger.error(f"Automatic weight download failed: {e}")

verify_model_files()

# Initialize API Application
app = FastAPI(
    title="Vision Assistant API",
    description="Backend API for object detection, color analysis, and audio scene descriptions for visually impaired users.",
    version="1.0.0"
)

# Enable CORS for cross-origin requests (e.g. mobile apps, external frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Engine Models (lazy loaded or preloaded)
try:
    logger.info(f"Loading YOLO model using: CFG={config.YOLO_CFG}, WEIGHTS={config.YOLO_WEIGHTS}")
    detector = ObjectDetector()
    color_detector = ColorDetector()
    descriptor = SceneDescriptor(detector)
    tts_engine = TextToSpeech()
    logger.info("Successfully loaded all model components and TTS engine.")
except Exception as e:
    logger.error(f"Critical error loading models: {e}")
    # We do not crash the script, allowing health checks to run and reporting errors over HTTP

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the single page web UI"""
    template_path = os.path.join("templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Web interface HTML file not found.")
    
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Error reading index.html: {e}")
        raise HTTPException(status_code=500, detail="Error serving the web interface.")

@app.get("/api/health")
async def health_check():
    """Service health and model metadata check"""
    try:
        # Verify model loading state
        model_loaded = 'detector' in globals() and detector.net is not None
        status = "healthy" if model_loaded else "degraded"
        
        return {
            "status": status,
            "timestamp": time.time(),
            "configuration": {
                "model_type": config.MODEL_TYPE,
                "cfg_file": config.YOLO_CFG,
                "weights_file": config.YOLO_WEIGHTS,
                "confidence_threshold": config.CONFIDENCE_THRESHOLD,
                "nms_threshold": config.NMS_THRESHOLD
            },
            "models_loaded": model_loaded
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Query(None, description="Confidence threshold override (0.0 to 1.0)"),
    generate_audio: bool = Query(True, description="Generate and return Base64 MP3 audio description")
):
    """
    Perform real-time object detection, color identification, and generate an audio announcement.
    
    - **file**: Uploaded image file (JPEG/PNG)
    - **conf_threshold**: Optional confidence score limit override
    - **generate_audio**: If true, synthesizes and returns base64 MP3 audio data.
    """
    # 1. Input validation
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    # Check if models were loaded
    if 'detector' not in globals() or detector.net is None:
        logger.error("Detections requested but model components are not loaded.")
        raise HTTPException(status_code=503, detail="Detection service is currently unavailable (model loading failed).")
        
    try:
        # 2. Read file bytes
        start_time = time.perf_counter()
        image_bytes = await file.read()
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Uploaded image bytes could not be decoded by OpenCV.")
            raise HTTPException(status_code=400, detail="Uploaded image is corrupted or invalid.")
        
        height, width, _ = img.shape
        
        # 3. Detect objects
        c_threshold = conf_threshold if conf_threshold is not None else config.CONFIDENCE_THRESHOLD
        boxes, confidences, class_ids, indexes = detector.detect_objects(
            img, 
            confidence_threshold=c_threshold, 
            nms_threshold=config.NMS_THRESHOLD
        )
        
        # 4. Analyze colors & map detections
        detected_ids = []
        color_info = defaultdict(list)
        detections_list = []
        
        for i in indexes:
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            label = detector.get_class_name(class_id)
            confidence = confidences[i]
            
            # Extract Region of Interest (ROI) for color detection
            roi = img[max(0, y):min(y+h, height), max(0, x):min(x+w, width)]
            detected_color = color_detector.detect_dominant_color(roi)
            
            detected_ids.append(class_id)
            color_info[class_id].append(detected_color)
            
            # Save object data (using bounding box coordinates relative to original image size)
            detections_list.append({
                "label": label,
                "confidence": round(float(confidence), 3),
                "color": detected_color,
                "box": [int(x), int(y), int(w), int(h)]
            })
            
        # 5. Generate scene description
        counts = {}
        for item in detected_ids:
            counts[item] = counts.get(item, 0) + 1
            
        description = descriptor.generate_description(counts, color_info)
        
        # 6. Generate text-to-speech audio if requested
        audio_base64 = None
        if generate_audio and len(detected_ids) > 0:
            audio_bytes = tts_engine.get_speech_bytes(description)
            if audio_bytes:
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        logger.info(f"Processed image in {inference_time_ms}ms. Detected {len(detections_list)} objects. Description: '{description}'")
        
        return {
            "success": True,
            "description": description,
            "audio": audio_base64,
            "detections": detections_list,
            "metrics": {
                "inference_time_ms": inference_time_ms,
                "model_used": config.MODEL_TYPE,
                "image_width": width,
                "image_height": height
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Inference pipeline crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error running inference.")

if __name__ == "__main__":
    import uvicorn
    # Allow running local debug server with python app.py
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
