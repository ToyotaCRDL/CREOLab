"""
Reference image generator for creating object-annotated reference images.
Generates reference images with numbered overlays for object identification.
"""

import cv2
import numpy as np
import os
import random
import json
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from src.core.base_models import CaptionData, ObjectInfo
from src.video.frame_extractor import FrameExtractor


class ReferenceImageGenerator:
    """
    Generates reference images with object position overlays.
    """
    
    def __init__(self, font_scale: float = 1.0, font_thickness: int = 2):
        """
        Initialize reference image generator.
        
        Args:
            font_scale: Scale factor for overlay text
            font_thickness: Thickness of overlay text
        """
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.frame_extractor = FrameExtractor()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Color palette for object overlays (BGR format)
        self.colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green  
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        
        # Text colors for each background color (BGR format)
        # Use black text for bright colors, white text for dark colors
        self.text_colors = [
            (255, 255, 255),  # White text for Red
            (0, 0, 0),        # Black text for Green
            (255, 255, 255),  # White text for Blue
            (0, 0, 0),        # Black text for Yellow
            (255, 255, 255),  # White text for Magenta
            (0, 0, 0),        # Black text for Cyan
            (255, 255, 255),  # White text for Purple
            (0, 0, 0),        # Black text for Orange
        ]
    
    
    def generate_random_object_numbers(self, num_objects: int) -> List[int]:
        """
        Generate random numbers for objects (1 to number of objects).
        
        Args:
            num_objects: Number of objects
            
        Returns:
            List of random numbers from 1 to num_objects
        """
        random_numbers = list(range(1, num_objects + 1))
        random.shuffle(random_numbers)
        return random_numbers

    def create_reference_image_with_overlay(self, 
                                          video_path: str,
                                          caption_data: CaptionData = None,
                                          output_path: str = None,
                                          objects: List[ObjectInfo] = None,
                                          use_object_ids: bool = False) -> Tuple[bool, List[int]]:
        """
        Create reference image with object position overlays.
        Extracts first frame from video and adds object overlays.
        
        Args:
            video_path: Path to source video
            caption_data: Object information for overlay (optional if objects provided)
            output_path: Path to save the reference image
            objects: List of ObjectInfo objects (optional if caption_data provided)
            use_object_ids: If True, use object.id for overlay text; if False, use random numbers
            
        Returns:
            Tuple of (success, random_numbers_used)
        """
        # Extract first frame from video using FrameExtractor
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract frame directly to the target directory
        output_dir = os.path.dirname(output_path)
        # Use a temporary filename for the raw frame (will be overwritten with overlay version)
        temp_filename = f"temp_frame_{os.path.basename(output_path)}"
        frame_path = self.frame_extractor.extract_first_frame(video_path, output_dir, temp_filename)
        frame = cv2.imread(frame_path)
        if frame is None:
            raise RuntimeError(f"Failed to load extracted frame: {frame_path}")
        # Clean up temporary frame file
        os.remove(frame_path)

        # Determine object list
        if objects:
            object_list = objects
        elif caption_data:
            object_list = caption_data.objects
        else:
            print("Error: Must provide either caption_data or objects")
            return False, []
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay_frame = frame.copy()
        
        # Generate random numbers for objects (1 to number of objects)
        num_objects = len(object_list)
        random_numbers = self.generate_random_object_numbers(num_objects)
        
        # Add object overlays
        for i, obj in enumerate(object_list):
            # Convert normalized coordinates to pixel coordinates
            x_pixel = int(obj.position[0] * width)
            y_pixel = int(obj.position[1] * height)
            
            # Get color for this object (cycle through colors if more objects than colors)
            color_index = i % len(self.colors)
            color = self.colors[color_index]
            text_color = self.text_colors[color_index]
            
            # Choose overlay text
            if use_object_ids and hasattr(obj, 'shuffled_id') and obj.shuffled_id is not None:
                # Use shuffled_id if available
                overlay_text = str(obj.shuffled_id)
            elif use_object_ids:
                # Fallback to regular id
                overlay_text = str(obj.id)
            else:
                # Use random numbers (legacy behavior)
                overlay_text = str(random_numbers[i])
            
            # Calculate text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                overlay_text, self.font, self.font_scale, self.font_thickness
            )
            
            # Draw background rectangle
            rect_x1 = x_pixel - 5
            rect_y1 = y_pixel - text_height - 5
            rect_x2 = x_pixel + text_width + 5
            rect_y2 = y_pixel + baseline + 5
            
            cv2.rectangle(overlay_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), 
                         color, -1)
            
            # Draw text with appropriate color for visibility
            cv2.putText(overlay_frame, overlay_text, (x_pixel, y_pixel),
                       self.font, self.font_scale, text_color, self.font_thickness)
            
            # Draw a small circle at the exact position
            cv2.circle(overlay_frame, (x_pixel, y_pixel), 3, color, -1)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the reference image
        success = cv2.imwrite(output_path, overlay_frame)
        if success:
            print(f"Reference image with overlay saved: {output_path}")
        else:
            print(f"Failed to save reference image: {output_path}")
        
        return success, random_numbers
    
    
    def generate_object_legend_json(self, 
                                  caption_data: CaptionData,
                                  output_path: str,
                                  random_numbers: List[int] = None) -> List[Dict]:
        """
        Generate a JSON legend for the overlaid objects.
        
        Args:
            caption_data: Object information
            output_path: Path to save the legend JSON
            random_numbers: Optional list of random numbers to use (should match overlay)
            
        Returns:
            List of object dictionaries with shuffled IDs
        """
        # Use provided random numbers or generate new ones
        if random_numbers is None:
            num_objects = len(caption_data.objects)
            random_numbers = self.generate_random_object_numbers(num_objects)
        
        legend_objects = []
        for i, obj in enumerate(caption_data.objects):
            shuffled_id = random_numbers[i]
            x, y = obj.position
            
            legend_objects.append({
                "shuffled_id": shuffled_id,
                "description": obj.description,
                "x": x,
                "y": y
            })
        
        # Sort by shuffled_id for consistent ordering
        legend_objects.sort(key=lambda x: x["shuffled_id"])
        
        # Save legend to JSON file
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(legend_objects, f, indent=2, ensure_ascii=False)
            print(f"Object legend JSON saved: {output_path}")
        
        return legend_objects
