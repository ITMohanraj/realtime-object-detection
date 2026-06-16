# scene_descriptor.py
from collections import defaultdict

class SceneDescriptor:
    def __init__(self, object_detector):
        """
        Initialize scene descriptor
        
        Args:
            object_detector: ObjectDetector instance
        """
        self.object_detector = object_detector
    
    def generate_description(self, counts, color_info):
        """
        Generate natural language description of the scene
        
        Args:
            counts: Dictionary of class_id -> count
            color_info: Dictionary of class_id -> list of colors
        
        Returns:
            Natural language description string
        """
        if not counts:
            return "No objects detected in the scene."
        
        descriptions = []
        
        # Group objects by type and color
        for class_id, count in counts.items():
            obj_name = self.object_detector.get_class_name(class_id)
            color_list = color_info.get(class_id, [])
            
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
        elif len(descriptions) > 0:
            return f"The scene contains {', '.join(descriptions[:-1])}, and {descriptions[-1]}."
        else:
            return "The scene appears to be empty."
    
    def generate_detailed_description(self, boxes, confidences, class_ids, indexes, color_info):
        """
        Generate detailed description with locations
        
        Args:
            boxes: List of bounding boxes
            confidences: List of confidence scores
            class_ids: List of class IDs
            indexes: List of selected indexes after NMS
            color_info: Dictionary of class_id -> list of colors
        
        Returns:
            Detailed description string
        """
        if len(indexes) == 0:
            return "No objects detected."
        
        descriptions = []
        
        for i in indexes:
            class_id = class_ids[i]
            obj_name = self.object_detector.get_class_name(class_id)
            confidence = confidences[i]
            color = color_info.get(class_id, ["unknown"])[0] if class_id in color_info else "unknown"
            
            # Get position (simplified)
            x, y, w, h = boxes[i]
            position = self._get_position_description(x, y, w, h)
            
            description = f"A {color} {obj_name} with {confidence:.0%} confidence {position}"
            descriptions.append(description)
        
        return " ".join(descriptions)
    
    def _get_position_description(self, x, y, w, h, img_width=640, img_height=480):
        """
        Get position description relative to image
        
        Args:
            x, y, w, h: Bounding box coordinates
            img_width, img_height: Image dimensions
        
        Returns:
            Position description string
        """
        center_x = x + w/2
        center_y = y + h/2
        
        # Horizontal position
        if center_x < img_width/3:
            horizontal = "on the left"
        elif center_x < 2*img_width/3:
            horizontal = "in the center"
        else:
            horizontal = "on the right"
        
        # Vertical position
        if center_y < img_height/3:
            vertical = "at the top"
        elif center_y < 2*img_height/3:
            vertical = "in the middle"
        else:
            vertical = "at the bottom"
        
        return f"located {horizontal} and {vertical}"