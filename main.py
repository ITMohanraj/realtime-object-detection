import cv2
import numpy as np
import streamlit as st
import tempfile
import time
from gtts import gTTS
import pygame
import os
from collections import defaultdict

# Initialize pygame for audio playback
pygame.mixer.init()

# -------------------- Load YOLO --------------------
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load COCO classes
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Random colors for bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# -------------------- Color Detection Function --------------------
def detect_color(roi):
    """Detect dominant color in ROI"""
    if roi.size == 0:
        return "unknown"
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate average color
    avg_color = cv2.mean(hsv)[:3]
    
    # Define color ranges in HSV
    color_ranges = {
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
    
    color_names = {
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
    
    h, s, v = avg_color
    
    # Check each color range
    for color_name, (lower, upper) in color_ranges.items():
        if (lower[0] <= h <= upper[0] and 
            lower[1] <= s <= upper[1] and 
            lower[2] <= v <= upper[2]):
            return color_names[color_name]
    
    return "colored"

# -------------------- Text-to-Speech Function --------------------
def text_to_speech(text, lang='en'):
    """Convert text to speech and play it"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
            tts.save(temp_file)
        
        # Play audio
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        # Clean up
        os.remove(temp_file)
        
    except Exception as e:
        print(f"TTS Error: {e}")

# -------------------- Generate Scene Description --------------------
def generate_scene_description(counts, color_info):
    """Generate natural language description of the scene"""
    if not counts:
        return "No objects detected in the scene."
    
    descriptions = []
    
    # Group objects by type and color
    for class_id, count in counts.items():
        obj_name = classes[class_id]
        color_list = color_info[class_id]
        
        if count == 1:
            if color_list:
                descriptions.append(f"One {color_list[0]} {obj_name}")
            else:
                descriptions.append(f"One {obj_name}")
        else:
            if color_list:
                # Count colors
                color_counts = defaultdict(int)
                for color in color_list:
                    color_counts[color] += 1
                
                color_desc = []
                for color, num in color_counts.items():
                    if num == 1:
                        color_desc.append(f"one {color}")
                    else:
                        color_desc.append(f"{num} {color}")
                
                color_str = " and ".join(color_desc)
                descriptions.append(f"{count} {obj_name}s: {color_str}")
            else:
                descriptions.append(f"{count} {obj_name}s")
    
    # Combine descriptions
    if len(descriptions) == 1:
        return f"The scene contains {descriptions[0]}."
    elif len(descriptions) == 2:
        return f"The scene contains {descriptions[0]} and {descriptions[1]}."
    else:
        return f"The scene contains {', '.join(descriptions[:-1])}, and {descriptions[-1]}."

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
                
                # Prepare input blob
                blob = cv2.dnn.blobFromImage(img, 1/255.0, (320, 320), swapRB=True, crop=False)
                net.setInput(blob)
                
                # Forward pass
                output_layers_names = net.getUnconnectedOutLayersNames()
                layer_outputs = net.forward(output_layers_names)
                
                boxes = []
                confidences = []
                class_ids = []
                
                # Extract bounding boxes
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
                
                # Non-maximum suppression
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
                
                detected_ids = []
                color_info = defaultdict(list)
                
                if len(indexes) > 0:
                    for i in indexes.flatten():
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
            
            # Process image
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (320, 320), swapRB=True, crop=False)
            net.setInput(blob)
            
            output_layers_names = net.getUnconnectedOutLayersNames()
            layer_outputs = net.forward(output_layers_names)
            
            boxes = []
            confidences = []
            class_ids = []
            
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
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
            
            detected_ids = []
            color_info = defaultdict(list)
            result_img = img.copy()
            
            if len(indexes) > 0:
                for i in indexes.flatten():
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