import cv2
import numpy as np
import streamlit as st
import time
from collections import defaultdict

# Initialize local modules
from object_detect import ObjectDetector
from color_detector import ColorDetector
from scene_descriptor import SceneDescriptor
from text_to_speech import TextToSpeech

# Cache resource loading to prevent re-initializing YOLO on every Streamlit rerun
@st.cache_resource
def load_detector():
    return ObjectDetector()

@st.cache_resource
def load_color_detector():
    return ColorDetector()

@st.cache_resource
def load_tts_engine():
    return TextToSpeech()

detector = load_detector()
color_detector = load_color_detector()
descriptor = SceneDescriptor(detector)
tts_engine = load_tts_engine()

# Map properties from detector
classes = detector.classes
colors = detector.colors

# -------------------- Wrappers for Compatibility --------------------
def detect_color(roi):
    """Detect dominant color in ROI using modular ColorDetector"""
    return color_detector.detect_dominant_color(roi)

def text_to_speech(text, lang='en'):
    """Speak text using modular TextToSpeech"""
    # Use speak method from tts_engine
    tts_engine.speak(text, lang=lang)

def generate_scene_description(counts, color_info):
    """Generate scene description using modular SceneDescriptor"""
    return descriptor.generate_description(counts, color_info)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Vision Assistant for Visually Impaired", layout="wide")

st.title("👁️ Vision Assistant for Visually Impaired Users")
st.markdown("### A simple, cost-effective system to describe scenes using AI")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    
    # Input selection
    input_mode = st.radio("Select Input Mode:", ["Webcam", "Upload Image"])
    
    confidence_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
    
    enable_tts = st.checkbox("Enable Text-to-Speech", value=True)
    
    auto_describe = st.checkbox("Auto-describe scene", value=True)
    
    if st.button("Describe Scene Now"):
        describe_now = True
    else:
        describe_now = False
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Select input mode (Webcam or Upload)
    2. Adjust confidence threshold
    3. Enable TTS for audio descriptions
    4. Click 'Describe Scene Now' for immediate audio
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Detection")
    
    if input_mode == "Webcam":
        # Webcam feed
        run_camera = st.checkbox("Start Camera", value=True)
        
        if run_camera:
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            
            last_description_time = time.time()
            description_interval = 10  # seconds
            
            while run_camera:
                ret, img = cap.read()
                if not ret:
                    st.error("Failed to capture from camera")
                    break
                
                height, width, _ = img.shape
                
                # Detect objects using modular detector
                boxes, confidences, class_ids, indexes = detector.detect_objects(
                    img, 
                    confidence_threshold=confidence_threshold, 
                    nms_threshold=0.4
                )
                
                detected_ids = []
                color_info = defaultdict(list)
                
                if len(indexes) > 0:
                    for i in indexes:
                        x, y, w, h = boxes[i]
                        label = classes[class_ids[i]]
                        color = colors[class_ids[i]]
                        
                        # Extract ROI for color detection
                        roi = img[max(0, y):min(y+h, height), max(0, x):min(x+w, width)]
                        detected_color = detect_color(roi)
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        
                        # Create label with color information
                        label_text = f"{detected_color} {label} {confidences[i]:.2f}"
                        cv2.putText(img, label_text, (x, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        
                        detected_ids.append(class_ids[i])
                        color_info[class_ids[i]].append(detected_color)
                
                # Count objects
                counts = {}
                for item in detected_ids:
                    counts[item] = counts.get(item, 0) + 1
                
                # Generate description
                description = generate_scene_description(counts, color_info)
                
                # Add description to image
                overlay = img.copy()
                cv2.rectangle(overlay, (10, 10), (400, 150), (0,0,0), -1)
                alpha = 0.7
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                
                # Display description on image
                y_offset = 40
                lines = description.split('. ')
                for line in lines:
                    if line:
                        cv2.putText(img, line, (20, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        y_offset += 25
                
                # Display counts
                cv2.rectangle(img, (width-250, 10), (width-10, 150), (0,0,0), -1)
                img = cv2.addWeighted(img, 1, img, 0, 0)
                
                x_offset, y_offset = width-230, 40
                for class_id, count in counts.items():
                    cv2.putText(img, f"{classes[class_id]}: {count}", (x_offset, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    y_offset += 25
                
                # Convert to RGB for Streamlit
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(img_rgb, channels="RGB", use_column_width=True)
                
                # Auto-describe at intervals
                current_time = time.time()
                if (auto_describe and enable_tts and 
                    current_time - last_description_time > description_interval and 
                    len(detected_ids) > 0):
                    text_to_speech(description)
                    last_description_time = current_time
                
                # Manual describe
                if describe_now and enable_tts and len(detected_ids) > 0:
                    text_to_speech(description)
                    describe_now = False
                    st.rerun()
                
                # Check for stop
                if not st.session_state.get('run_camera', True):
                    break
            
            cap.release()
    
    else:  # Upload Image mode
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            height, width, _ = img.shape
            
            # Detect objects using modular detector
            boxes, confidences, class_ids, indexes = detector.detect_objects(
                img, 
                confidence_threshold=confidence_threshold, 
                nms_threshold=0.4
            )
            
            detected_ids = []
            color_info = defaultdict(list)
            result_img = img.copy()
            
            if len(indexes) > 0:
                for i in indexes:
                    x, y, w, h = boxes[i]
                    label = classes[class_ids[i]]
                    
                    # Extract ROI for color detection
                    roi = result_img[max(0, y):min(y+h, height), max(0, x):min(x+w, width)]
                    detected_color = detect_color(roi)
                    
                    # Draw bounding box
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0,255,0), 2)
                    
                    # Create label with color information
                    label_text = f"{detected_color} {label} {confidences[i]:.2f}"
                    cv2.putText(result_img, label_text, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    
                    detected_ids.append(class_ids[i])
                    color_info[class_ids[i]].append(detected_color)
            
            # Display result
            st.image(result_img, caption="Processed Image", use_column_width=True)
            
            # Generate and display description
            counts = {}
            for item in detected_ids:
                counts[item] = counts.get(item, 0) + 1
            
            description = generate_scene_description(counts, color_info)
            
            with col2:
                st.header("Scene Description")
                st.write(description)
                
                if enable_tts and st.button("🔊 Hear Description"):
                    text_to_speech(description)

with col2:
    st.header("Detection Summary")
    
    if 'counts' in locals():
        if counts:
            for class_id, count in counts.items():
                obj_name = classes[class_id]
                color_list = color_info.get(class_id, [])
                
                if color_list:
                    color_counts = defaultdict(int)
                    for color in color_list:
                        color_counts[color] += 1
                    
                    color_text = ", ".join([f"{num} {color}" for color, num in color_counts.items()])
                    st.write(f"**{obj_name.capitalize()}**: {count} total ({color_text})")
                else:
                    st.write(f"**{obj_name.capitalize()}**: {count}")
        else:
            st.write("No objects detected")
    
    st.markdown("---")
    st.header("About")
    st.markdown("""
    This system assists visually impaired users by:
    
    • **Object Detection**: Identifying objects using YOLOv3
    • **Color Detection**: Determining object colors
    • **Audio Description**: Converting scene to speech
    
    **Features:**
    - Real-time webcam processing
    - Image upload support
    - Adjustable confidence levels
    - Natural language descriptions
    - Text-to-speech output
    """)

# Cleanup
cv2.destroyAllWindows()