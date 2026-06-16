# utils.py
import cv2
import numpy as np
from collections import defaultdict

def count_occurrences(lst):
    """Return a dictionary with count of each item in list."""
    counts = {}
    for item in lst:
        counts[item] = counts.get(item, 0) + 1
    return counts

def draw_detections(image, boxes, confidences, class_ids, indexes, 
                    object_detector, color_info, show_confidence=True):
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: Input image
        boxes: List of bounding boxes
        confidences: List of confidence scores
        class_ids: List of class IDs
        indexes: List of selected indexes after NMS
        object_detector: ObjectDetector instance
        color_info: Dictionary of class_id -> list of colors
        show_confidence: Whether to show confidence score
    
    Returns:
        Image with drawn detections
    """
    result = image.copy()
    
    if len(indexes) > 0:
        for i in indexes:
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            
            # Get label and color
            label = object_detector.get_class_name(class_id)
            box_color = object_detector.get_class_color(class_id)
            
            # Get detected color for this object
            detected_color = color_info.get(class_id, ["unknown"])[0]
            
            # Create label text
            if show_confidence:
                label_text = f"{detected_color} {label} {confidences[i]:.2f}"
            else:
                label_text = f"{detected_color} {label}"
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), box_color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result, (x, y - 25), (x + label_size[0], y), box_color, -1)
            
            # Draw label text
            cv2.putText(result, label_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result

def add_text_overlay(image, text, position=(10, 30), 
                     bg_color=(0, 0, 0), text_color=(255, 255, 255),
                     alpha=0.7, font_scale=0.6, thickness=2):
    """
    Add semi-transparent text overlay to image
    
    Args:
        image: Input image
        text: Text to add
        position: (x, y) position for text
        bg_color: Background color (BGR)
        text_color: Text color (BGR)
        alpha: Transparency of background
        font_scale: Font scale
        thickness: Font thickness
    
    Returns:
        Image with text overlay
    """
    result = image.copy()
    
    # Split text into lines
    lines = text.split('\n')
    
    # Calculate total text block size
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 25
    max_width = 0
    
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, text_width)
    
    total_height = len(lines) * line_height
    
    # Create semi-transparent background
    x, y = position
    overlay = result.copy()
    cv2.rectangle(overlay, (x - 5, y - 20), 
                  (x + max_width + 5, y + total_height), 
                  bg_color, -1)
    
    # Apply transparency
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    
    # Add text lines
    y_offset = y
    for line in lines:
        cv2.putText(result, line, (x, y_offset), 
                   font, font_scale, text_color, thickness)
        y_offset += line_height
    
    return result

def create_stats_panel(image, counts, color_info, object_detector, 
                       position='right', width=250):
    """
    Create statistics panel overlay
    
    Args:
        image: Input image
        counts: Dictionary of class_id -> count
        color_info: Dictionary of class_id -> list of colors
        object_detector: ObjectDetector instance
        position: 'left' or 'right'
        width: Width of panel
    
    Returns:
        Image with stats panel
    """
    result = image.copy()
    img_height, img_width = image.shape[:2]
    
    # Determine panel position
    if position == 'right':
        x_start = img_width - width
        x_end = img_width
    else:
        x_start = 0
        x_end = width
    
    # Create semi-transparent panel
    overlay = result.copy()
    cv2.rectangle(overlay, (x_start, 10), (x_end - 10, 200), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)
    
    # Add title
    cv2.putText(result, "Detection Summary:", (x_start + 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add counts
    y_offset = 70
    for class_id, count in counts.items():
        obj_name = object_detector.get_class_name(class_id)
        
        if class_id in color_info:
            # Count colors
            from collections import defaultdict
            color_counts = defaultdict(int)
            for color in color_info[class_id]:
                color_counts[color] += 1
            
            color_text = ", ".join([f"{num} {color}" for color, num in color_counts.items()])
            text = f"{obj_name}: {count} ({color_text})"
        else:
            text = f"{obj_name}: {count}"
        
        cv2.putText(result, text, (x_start + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
    
    return result