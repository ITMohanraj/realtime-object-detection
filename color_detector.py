# color_detector.py
import cv2
import numpy as np
from config import COLOR_RANGES, COLOR_NAMES

class ColorDetector:
    def __init__(self):
        """Initialize color detector"""
        pass
    
    def detect_dominant_color(self, roi):
        """
        Detect dominant color in Region of Interest
        
        Args:
            roi: Region of Interest (BGR image)
        
        Returns:
            Color name as string
        """
        if roi.size == 0:
            return "unknown"
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate average color
        avg_color = cv2.mean(hsv)[:3]
        h, s, v = avg_color
        
        # Check each color range
        for color_name, (lower, upper) in COLOR_RANGES.items():
            if (lower[0] <= h <= upper[0] and 
                lower[1] <= s <= upper[1] and 
                lower[2] <= v <= upper[2]):
                return COLOR_NAMES[color_name]
        
        return "colored"
    
    def detect_multiple_colors(self, roi, num_colors=3):
        """
        Detect multiple dominant colors in ROI using clustering
        
        Args:
            roi: Region of Interest (BGR image)
            num_colors: Number of dominant colors to detect
        
        Returns:
            List of color names
        """
        if roi.size == 0:
            return ["unknown"]
        
        # Reshape the image to be a list of pixels
        pixels = roi.reshape(-1, 3)
        
        # Convert to float32 for k-means
        pixels = np.float32(pixels)
        
        # Define criteria and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to uint8
        centers = np.uint8(centers)
        
        # Convert centers to HSV and detect colors
        detected_colors = []
        for center in centers:
            # Convert single pixel to HSV
            center_hsv = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = center_hsv
            
            # Find matching color
            color_found = False
            for color_name, (lower, upper) in COLOR_RANGES.items():
                if (lower[0] <= h <= upper[0] and 
                    lower[1] <= s <= upper[1] and 
                    lower[2] <= v <= upper[2]):
                    detected_colors.append(COLOR_NAMES[color_name])
                    color_found = True
                    break
            
            if not color_found:
                detected_colors.append("colored")
        
        return list(set(detected_colors))  # Return unique colors