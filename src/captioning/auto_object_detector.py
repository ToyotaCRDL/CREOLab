"""
Auto object detector that automatically detects objects from reference images
and generates overlay images with object annotations.
Focused on object detection, reference image creation, position adjustment, and mapping creation.
Does not handle segment captioning - delegates that to segment_captioner.py.
"""

import os
import json
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import tempfile

from src.core.openai_client import OpenAIVisionClient
from src.core.base_models import ExtractedFrame, ProcedureStep, SegmentProcedure, ObjectInfo, CaptionData, create_step_id
from src.utils.prompt_loader import prompt_loader


class AutoObjectDetector:
    """
    Auto object detector that automatically detects objects from reference images and generates
    overlay images with object annotations for enhanced captioning.
    Focused on detection, reference image creation, position adjustment, and mapping creation.
    """
    
    def __init__(self, openai_client: OpenAIVisionClient, logger: logging.Logger = None):
        """
        Initialize auto object detector.
        
        Args:
            openai_client: OpenAI Vision API client
            logger: Logger instance (defaults to module logger if not provided)
        """
        self.client = openai_client
        self.prompt_loader = prompt_loader
        self.logger = logger or logging.getLogger(__name__)
        # Only multi-frame detection is used
        self._detected_objects_cache = {}
        self._overlay_cache = {}
    

    def _get_clean_base_name(self, file_path: str, suffixes_to_remove: list = None) -> str:
        """
        Extract clean base name from file path by removing common suffixes.
        
        Args:
            file_path: Full path or filename
            suffixes_to_remove: List of suffixes to remove (default: standard reference image suffixes)
            
        Returns:
            Clean base name without extension and suffixes
            
        Example:
            'scenario01_decoy0_reference_no_overlay.jpg' -> 'scenario01_decoy0'
            'scenario01_decoy0_reference_with_overlay.jpg' -> 'scenario01_decoy0'
        """
        # Default suffixes if not specified
        if suffixes_to_remove is None:
            suffixes_to_remove = [
                '_reference_no_overlay',
                '_reference_with_overlay',
                '_reference_auto_detection_overlay',
                '_reference_auto_detection_overlay_adj'
            ]
        
        # Extract base name without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Remove matching suffixes
        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break  # Only remove one suffix
        
        return base_name
    
    def detect_objects_from_multi_frame_context(self, original_video_path: str, video_id: str = None, output_dir: str = None) -> List[ObjectInfo]:
        """
        Detect objects using first frame and multiple context frames from original video.
        Extracts first frame as reference and 7 evenly distributed frames from the original video 
        to provide better context for object identification and naming.
        
        Args:
            original_video_path: Path to the original video file
            video_id: Optional video identifier for caching
            output_dir: Optional output directory for context frames
            
        Returns:
            List of detected ObjectInfo instances with improved naming
        """
        # Check cache first
        cache_key = f"multi_frame_{video_id or original_video_path}"
        if cache_key in self._detected_objects_cache:
            return self._detected_objects_cache[cache_key]
        
        # Import FrameExtractor
        from src.video.frame_extractor import FrameExtractor
        
        self.logger.info(f"  Enhanced object detection using multi-frame context from: {original_video_path}")
        
        # Create context frames directory within output directory
        if output_dir:
            context_frames_dir = os.path.join(output_dir, "context_frames")
        else:
            # Fallback to a context_frames directory in current working directory
            context_frames_dir = "context_frames"
        
        os.makedirs(context_frames_dir, exist_ok=True)
        
        # Extract first frame as reference image
        extractor = FrameExtractor()
        reference_image_path = extractor.extract_first_frame(original_video_path, context_frames_dir)
        if not reference_image_path:
            error_msg = f"CRITICAL ERROR: Failed to extract first frame from video: {original_video_path}"
            self.logger.error(f"  {error_msg}")
            raise RuntimeError(error_msg)
        
        self.logger.info(f"  Extracted reference frame: {reference_image_path}")
        
        # Extract 7 evenly distributed context frames
        context_frames = extractor.extract_context_frames_from_video(
            original_video_path, context_frames_dir, num_frames=7
        )
        
        self.logger.info(f"  Extracted {len(context_frames)} context frames for enhanced detection")
        
        # Prepare multi-frame prompt
        multi_frame_prompt = self.prompt_loader.get_prompt("common_captioning_prompts.json", "multi_frame_detection")
        
        # Prepare image list: reference image first, then context frames
        image_paths = [reference_image_path] + context_frames
        
        # Use multi-frame analysis
        response = self.client.analyze_multiple_frames(
            frame_paths=image_paths,
            prompt=multi_frame_prompt
        )
        
        self.logger.info(f"  Multi-frame detection response length: {len(response)} characters")
        
        # Parse JSON response using unified parser
        objects_data = self._parse_gpt_json_response(response, expected_type='dict')
        
        objects = []
        
        for obj_data in objects_data.get("objects", []):
            obj_info = ObjectInfo(
                id=obj_data["id"],
                description=obj_data["description"],
                position=tuple(obj_data["position"]) if obj_data.get("position") else None
            )
            objects.append(obj_info)
        
        # Apply unique naming for duplicate object descriptions
        objects = self._apply_unique_naming(objects)
        
        # Assign shuffled IDs for display consistency
        objects = self._assign_shuffled_ids(objects)
        
        self.logger.info(f"  Enhanced detection found {len(objects)} objects:")
        for obj in objects:
            self.logger.info(f"    {obj.id}. {obj.description} (x={obj.position[0]:.2f}, y={obj.position[1]:.2f})")
        
        # Cache the results
        self._detected_objects_cache[cache_key] = objects
        return objects
    
    def adjust_object_coordinates(self, overlay_image_path: str, objects: List[ObjectInfo], video_id: str = None) -> List[ObjectInfo]:
        """
        Adjust object coordinates using GPT-5 by analyzing the overlay image.
        
        Args:
            overlay_image_path: Path to the overlay image with numbered objects
            objects: List of original ObjectInfo instances
            video_id: Optional video identifier for caching
            
        Returns:
            List of ObjectInfo instances with adjusted coordinates
        """
        if not os.path.exists(overlay_image_path):
            raise FileNotFoundError(f"Overlay image not found for adjustment: {overlay_image_path}")
        
        self.logger.info(f"  Adjusting object coordinates using overlay image: {overlay_image_path}")
        
        # Format current objects for prompt using shuffled_id if available
        current_objects_text = ""
        for obj in objects:
            display_id = obj.shuffled_id if obj.shuffled_id is not None else obj.id
            if obj.position:
                current_objects_text += f"{display_id}. {obj.description} (x={obj.position[0]:.2f}, y={obj.position[1]:.2f})\n"
            else:
                current_objects_text += f"{display_id}. {obj.description}\n"
        
        # Create adjustment prompt
        adjustment_prompt = self.prompt_loader.get_prompt("common_captioning_prompts.json", "coordinate_adjustment").format(
            current_objects=current_objects_text.strip()
        )
        
        # Use GPT-5 to adjust coordinates
        response = self.client.analyze_single_frame(
            frame_path=overlay_image_path,
            prompt=adjustment_prompt
        )
        
        # Clean and parse JSON response
        adjusted_data = self._parse_gpt_json_response(response, expected_type='dict')
        self.logger.info(f"  Successfully parsed JSON with {len(adjusted_data.get('objects', []))} objects")
        
        adjusted_objects = []
        
        for obj_data in adjusted_data.get("objects", []):
            # Find the original object to preserve shuffled_id
            # Match by shuffled_id if available, otherwise by id
            original_obj = None
            for o in objects:
                if (o.shuffled_id is not None and o.shuffled_id == obj_data["id"]) or \
                   (o.shuffled_id is None and o.id == obj_data["id"]):
                    original_obj = o
                    break
            
            adjusted_obj = ObjectInfo(
                id=original_obj.id if original_obj else obj_data["id"],
                description=obj_data["description"],
                position=tuple(obj_data["position"]) if obj_data.get("position") else None,
                shuffled_id=original_obj.shuffled_id if original_obj else None
            )
            adjusted_objects.append(adjusted_obj)
        
        self.logger.info(f"  Adjusted {len(adjusted_objects)} object coordinates:")
        for obj in adjusted_objects:
            original_obj = next((o for o in objects if o.id == obj.id), None)
            if original_obj and original_obj.position and obj.position:
                x_diff = abs(obj.position[0] - original_obj.position[0])
                y_diff = abs(obj.position[1] - original_obj.position[1])
                if x_diff > 0.005 or y_diff > 0.005:  # Any noticeable change (0.5% threshold)
                    self.logger.info(f"    {obj.id}. {obj.description}: ({original_obj.position[0]:.3f}, {original_obj.position[1]:.3f}) -> ({obj.position[0]:.3f}, {obj.position[1]:.3f}) [Δx={x_diff:.3f}, Δy={y_diff:.3f}]")
                else:
                    self.logger.info(f"    {obj.id}. {obj.description}: No significant adjustment (Δx={x_diff:.3f}, Δy={y_diff:.3f})")
        
        return adjusted_objects
    
    def generate_overlay_image(self, reference_image_path: str, objects: List[ObjectInfo], output_dir: str = None, video_path: str = None) -> str:
        """
        Generate an overlay image with numbered object annotations and save object list.
        Uses ReferenceImageGenerator for consistent overlay generation.
        
        Args:
            reference_image_path: Path to the original reference image
            objects: List of detected objects
            output_dir: Optional output directory (if None, uses reference_images in current output)
            video_path: Path to the original video (for extracting first frame)
            
        Returns:
            Path to the generated overlay image
        """
        if not os.path.exists(reference_image_path):
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
        
        # Check cache
        cache_key = f"{reference_image_path}_{len(objects)}"
        if cache_key in self._overlay_cache:
            return self._overlay_cache[cache_key]
        
        # Import ReferenceImageGenerator
        from src.utils.reference_image_generator import ReferenceImageGenerator
        
        # Determine output directory and create proper file names
        if output_dir is None:
            raise ValueError("output_dir must be provided for overlay image generation")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create proper file names
        base_name = self._get_clean_base_name(reference_image_path)
        overlay_image_path = os.path.join(output_dir, f"{base_name}_reference_auto_detection_overlay.jpg")
        object_list_json_path = os.path.join(output_dir, f"{base_name}_reference_auto_detection_objects.json")
        
        # Initialize ReferenceImageGenerator and use its method
        ref_generator = ReferenceImageGenerator()
        
        # Use ReferenceImageGenerator to create overlay from video first frame
        if not video_path:
            raise ValueError("video_path must be provided for overlay image generation")
        
        success, random_numbers = ref_generator.create_reference_image_with_overlay(
            video_path=video_path,
            objects=objects,
            output_path=overlay_image_path,
            use_object_ids=True  # Use object IDs for overlay text
        )
        
        if not success:
            raise RuntimeError(f"Failed to generate overlay image: {overlay_image_path}")
        
        # Save the object list as JSON file
        self._save_adjusted_object_list_json(objects, object_list_json_path)
        
        # Cache the result
        self._overlay_cache[cache_key] = overlay_image_path
        return overlay_image_path
    
    
    def generate_adjusted_overlay_and_list(self, reference_image_path: str, original_objects: List[ObjectInfo], 
                                         output_dir: str, video_id: str = None, video_path: str = None) -> Tuple[str, str, List[ObjectInfo]]:
        """
        Generate adjusted overlay image and object list using coordinate adjustment.
        Re-loads existing JSON and overlay files for proper adjustment.
        
        Args:
            reference_image_path: Path to the original reference image
            original_objects: List of original detected objects
            output_dir: Output directory for adjusted files
            video_id: Optional video identifier
            video_path: Path to the original video (for extracting first frame)
            
        Returns:
            Tuple of (adjusted_overlay_path, adjusted_object_list_path, adjusted_objects)
        """
        # Determine base name for file paths
        base_name = self._get_clean_base_name(reference_image_path)
        
        # Define paths for existing files
        existing_overlay_path = os.path.join(output_dir, f"{base_name}_reference_auto_detection_overlay.jpg")
        existing_json_path = os.path.join(output_dir, f"{base_name}_reference_auto_detection_objects.json")
        
        # Load objects: either from existing JSON or from provided original_objects
        if os.path.exists(existing_json_path):
            # Re-load existing JSON file if it exists
            self.logger.info(f"  Re-loading existing JSON: {existing_json_path}")
            with open(existing_json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Parse objects from JSON
            objects_to_adjust = []
            for obj_data in json_data:
                obj_info = ObjectInfo(
                    id=obj_data.get("shuffled_id", obj_data.get("id", 1)),
                    description=obj_data["description"],
                    position=(obj_data["x"], obj_data["y"]),
                    shuffled_id=obj_data.get("shuffled_id")
                )
                objects_to_adjust.append(obj_info)
            
            self.logger.info(f"  ✓ Re-loaded {len(objects_to_adjust)} objects from JSON")
        else:
            # Use provided original_objects
            self.logger.info(f"  Using provided objects (no existing JSON found)")
            objects_to_adjust = original_objects
        
        # Use existing overlay image if available, otherwise generate it
        if not os.path.exists(existing_overlay_path):
            self.logger.info(f"  Overlay image not found, generating: {existing_overlay_path}")
            overlay_path_for_adjustment = self.generate_overlay_image(reference_image_path, objects_to_adjust, output_dir, video_path)
        else:
            self.logger.info(f"  Using existing overlay for adjustment: {existing_overlay_path}")
            overlay_path_for_adjustment = existing_overlay_path
        
        # Adjust coordinates using the overlay image
        adjusted_objects = self.adjust_object_coordinates(overlay_path_for_adjustment, objects_to_adjust, video_id)
        
        # Define paths for adjusted files
        adjusted_overlay_path = os.path.join(output_dir, f"{base_name}_reference_auto_detection_overlay_adj.jpg")
        auto_legend_json_path = os.path.join(output_dir, f"{base_name}_reference_auto_detection_objects_adj.json")
        
        # Import ReferenceImageGenerator
        from src.utils.reference_image_generator import ReferenceImageGenerator
        
        # Initialize ReferenceImageGenerator and use its method for adjusted overlay
        ref_generator = ReferenceImageGenerator()
        
        success, random_numbers = ref_generator.create_reference_image_with_overlay(
            video_path=video_path,
            objects=adjusted_objects,
            output_path=adjusted_overlay_path,
            use_object_ids=True  # Use object IDs for overlay text
        )
        
        if not success:
            raise RuntimeError(f"Failed to generate adjusted overlay image: {adjusted_overlay_path}")
        
        # Save the adjusted object list as JSON
        self._save_adjusted_object_list_json(adjusted_objects, auto_legend_json_path)
        
        return adjusted_overlay_path, auto_legend_json_path, adjusted_objects
    
    def _save_adjusted_object_list_json(self, objects: List[ObjectInfo], output_path: str) -> None:
        """
        Save the adjusted objects list to a JSON file.
        
        Args:
            objects: List of adjusted objects
            output_path: Path to save the JSON file
        """
        # Sort objects by shuffled_id for consistent ordering
        sorted_objects = sorted(objects, key=lambda obj: obj.shuffled_id if obj.shuffled_id is not None else obj.id)
        
        legend_objects = []
        for obj in sorted_objects:
            obj_data = {
                "shuffled_id": obj.shuffled_id if obj.shuffled_id is not None else obj.id,
                "description": obj.description,
                "x": round(obj.position[0], 2) if obj.position else 0.0,
                "y": round(obj.position[1], 2) if obj.position else 0.0
            }
            legend_objects.append(obj_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(legend_objects, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"  Saved adjusted object list JSON: {output_path}")
    
    def format_auto_detected_objects(self, objects: List[ObjectInfo]) -> str:
        """
        Format auto-detected object information for prompt inclusion.
        
        Args:
            objects: List of auto-detected ObjectInfo instances
            
        Returns:
            Formatted string for prompt
        """
        if not objects:
            return "No objects were automatically detected in the reference image."
        
        lines = ["Object Legend for Reference Image:", "=" * 40]
        
        # Sort objects by shuffled_id for consistent ordering
        sorted_objects = sorted(objects, key=lambda obj: obj.shuffled_id if obj.shuffled_id is not None else obj.id)
        
        for obj in sorted_objects:
            # Use shuffled_id for display if available, otherwise use original id
            display_id = obj.shuffled_id if obj.shuffled_id is not None else obj.id
            if obj.position:
                # Format position with 2 decimal places
                x_pos = round(obj.position[0], 2)
                y_pos = round(obj.position[1], 2)
                line = f"{display_id}. {obj.description} (x={x_pos:.2f}, y={y_pos:.2f})"
            else:
                line = f"{display_id}. {obj.description}"
            lines.append(line)
        
        return "\n".join(lines)
    
    
    
    def generate_object_name_mapping(self, auto_detected_objects: List[ObjectInfo], 
                                   manual_overlay_path: str, manual_legend_json_path: str, 
                                   output_dir: str, video_id: str, auto_overlay_path: str = None) -> str:
        """
        Generate object name mapping between auto-detected and manual objects using GPT.
        
        Args:
            auto_detected_objects: List of automatically detected objects
            manual_overlay_path: Path to manual reference image with overlay
            manual_legend_json_path: Path to manual object legend JSON file
            output_dir: Output directory for mapping file
            video_id: Video identifier
            auto_overlay_path: Path to auto-detected overlay image (optional)
            
        Returns:
            Path to the generated mapping file
            
        Raises:
            RuntimeError: If mapping generation fails
        """
        # Read manual object legend JSON
        if not os.path.exists(manual_legend_json_path):
            raise RuntimeError(f"Manual legend JSON file not found: {manual_legend_json_path}")
        
        with open(manual_legend_json_path, 'r', encoding='utf-8') as f:
            manual_legend_data = json.load(f)
        
        # Parse manual objects from JSON legend
        manual_objects = self._parse_manual_objects_from_json(manual_legend_data)
        
        # Format auto-detected objects for prompt
        auto_objects_text = self._format_objects_for_mapping_prompt(auto_detected_objects, "auto")
        manual_objects_text = self._format_objects_for_mapping_prompt(manual_objects, "manual")
        
        # Load mapping prompt
        mapping_prompt = prompt_loader.get_prompt("common_captioning_prompts.json", "object_name_mapping")
        
        # Format prompt with object information
        formatted_prompt = mapping_prompt.format(
            auto_objects_text=auto_objects_text,
            manual_objects_text=manual_objects_text
        )
        
        # Call GPT to generate mapping
        self.logger.info(f"  Generating object name mapping: {len(auto_detected_objects)} auto → {len(manual_objects)} manual objects")
        
        # Prepare image paths for analysis
        image_paths = [manual_overlay_path]
        if auto_overlay_path and os.path.exists(auto_overlay_path):
            image_paths.append(auto_overlay_path)
        
        response = self.client.analyze_frames(
            frame_paths=image_paths,
            prompt=formatted_prompt
        )
        
        if not response or not response.strip():
            raise RuntimeError("GPT API returned empty or whitespace-only response for object name mapping")
        
        if len(response.strip()) < 10:
            raise RuntimeError(f"GPT response too short to be valid JSON: '{response}'")
        
        # Validation function for mapping format
        def validate_mapping(mapping_data):
            # Filter out non-string values and validate mapping format
            clean_mapping = {}
            invalid_pairs = []
            
            for key, value in mapping_data.items():
                if isinstance(key, str) and isinstance(value, str):
                    if key.strip() and value.strip():
                        clean_mapping[key.strip()] = value.strip()
                    else:
                        invalid_pairs.append(f"'{key}' -> '{value}' (empty strings)")
                else:
                    invalid_pairs.append(f"'{key}' -> '{value}' (non-string types: {type(key).__name__} -> {type(value).__name__})")
            
            # Log invalid pairs for debugging
            if invalid_pairs:
                self.logger.debug(f"Skipped {len(invalid_pairs)} invalid mapping pairs: {invalid_pairs[:3]}")
            
            # Ensure we have at least some valid mappings
            if not clean_mapping:
                raise RuntimeError(f"No valid string-to-string mappings found. Original had {len(mapping_data)} pairs, all invalid.")
            
            # Replace original data with cleaned version
            mapping_data.clear()
            mapping_data.update(clean_mapping)
        
        # Use unified parser with mapping-specific validation
        mapping_data = self._parse_gpt_json_response(response, expected_type='dict', validate_fn=validate_mapping)
        
        # Save mapping to file
        base_name = self._get_clean_base_name(manual_overlay_path)
        mapping_file_path = os.path.join(output_dir, f"{base_name}_object_name_mapping.json")
        
        with open(mapping_file_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"  Generated object name mapping: {len(mapping_data)} pairs → {mapping_file_path}")
        
        return mapping_file_path
    
    
    def _parse_manual_objects_from_json(self, legend_data: List[Dict]) -> List[ObjectInfo]:
        """
        Parse manual objects from JSON legend data.
        
        Args:
            legend_data: List of object dictionaries from JSON legend
            
        Returns:
            List of ObjectInfo objects parsed from JSON
        """
        objects = []
        
        for obj_data in legend_data:
            objects.append(ObjectInfo(
                id=obj_data["shuffled_id"],
                description=obj_data["description"],
                position=(obj_data["x"], obj_data["y"]),
                shuffled_id=obj_data["shuffled_id"]
            ))
        
        return objects
    
    def _format_objects_for_mapping_prompt(self, objects: List[ObjectInfo], object_type: str) -> str:
        """
        Format objects for mapping prompt.
        
        Args:
            objects: List of ObjectInfo objects
            object_type: Type identifier ("auto" or "manual")
            
        Returns:
            Formatted string for prompt
        """
        if not objects:
            return f"No {object_type} objects available."
        
        lines = []
        for obj in objects:
            if obj.position:
                line = f"{obj.id}. {obj.description} (x={obj.position[0]:.2f}, y={obj.position[1]:.2f})"
            else:
                line = f"{obj.id}. {obj.description}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _apply_unique_naming(self, objects: List[ObjectInfo]) -> List[ObjectInfo]:
        """
        Apply unique naming to objects with duplicate descriptions.
        
        Args:
            objects: List of ObjectInfo objects
            
        Returns:
            List of ObjectInfo objects with unique descriptions
        """
        # Count occurrences of each description
        description_counts = {}
        for obj in objects:
            base_desc = obj.description.strip()
            description_counts[base_desc] = description_counts.get(base_desc, 0) + 1
        
        # Apply unique naming for duplicates
        description_counters = {}
        updated_objects = []
        
        for obj in objects:
            base_desc = obj.description.strip()
            
            if description_counts[base_desc] > 1:
                # Multiple objects with same description - add identifier
                counter = description_counters.get(base_desc, 0) + 1
                description_counters[base_desc] = counter
                
                # Generate unique identifier (A, B, C, etc.)
                identifier = chr(ord('A') + counter - 1)
                unique_description = f"{base_desc} {identifier}"
                
                self.logger.info(f"    Renamed duplicate: '{base_desc}' → '{unique_description}'")
            else:
                # Unique description - keep as is
                unique_description = base_desc
            
            # Create new ObjectInfo with updated description
            updated_obj = ObjectInfo(
                id=obj.id,
                description=unique_description,
                position=obj.position
            )
            updated_objects.append(updated_obj)
        
        return updated_objects
    
    def _parse_gpt_json_response(self, response: str, expected_type: str = 'dict', validate_fn=None) -> Optional[any]:
        """
        Unified JSON extraction and parsing utility for GPT responses.
        
        Performs: code block removal → JSON boundary detection → parsing → type validation → custom validation
        
        Args:
            response: Raw GPT response
            expected_type: Expected JSON type ('dict' or 'list')
            validate_fn: Optional function(data) that raises RuntimeError if validation fails
            
        Returns:
            Parsed and validated JSON data (dict or list)
            
        Raises:
            RuntimeError: If response is empty, JSON is invalid, or validation fails
        """
        if not response or not response.strip():
            raise RuntimeError("Empty response received from GPT API")
        
        # Step 1 & 2: Remove code blocks and common prefixes
        response = self._preprocess_gpt_response(response)
        
        # Step 3: JSON boundary detection
        start_char, end_char = ('{', '}') if expected_type == 'dict' else ('[', ']')
        start_idx = response.find(start_char)
        end_idx = response.rfind(end_char)
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            raise RuntimeError(f"No valid JSON boundaries found (expected {expected_type}). Start: {start_idx}, End: {end_idx}")
        
        # Extract JSON portion
        json_str = response[start_idx:end_idx + 1]
        
        # Validate JSON string is not truncated
        if json_str.count(start_char) != json_str.count(end_char):
            raise RuntimeError(f"Truncated JSON detected: {start_char} count={json_str.count(start_char)}, {end_char} count={json_str.count(end_char)}")
        
        # Step 4: Parse JSON
        data = json.loads(json_str)
        
        # Step 5: Type validation
        if expected_type == 'dict' and not isinstance(data, dict):
            raise RuntimeError(f"Expected dictionary, got {type(data).__name__}")
        elif expected_type == 'list' and not isinstance(data, list):
            raise RuntimeError(f"Expected list, got {type(data).__name__}")
        
        # Check for empty data
        if not data:
            raise RuntimeError(f"Empty {expected_type} received from GPT")
        
        # Step 6: Custom validation
        if validate_fn:
            validate_fn(data)
        
        return data
    
    def _preprocess_gpt_response(self, response: str) -> str:
        """
        Preprocess GPT response to handle common formatting issues.
        
        Args:
            response: Raw GPT response
            
        Returns:
            Cleaned response with common formatting issues resolved
        """
        # Remove code block markers if present
        if response.startswith('```'):
            # Find the end of the opening code block marker
            first_newline = response.find('\n')
            if first_newline != -1:
                response = response[first_newline + 1:]
        
        if response.endswith('```'):
            response = response[:-3]
        
        # Remove "json" language specifier if present
        if response.startswith('json\n'):
            response = response[5:]
        
        # Remove common prefixes that GPT might add
        prefixes_to_remove = [
            'Here is the JSON mapping:',
            'Here\'s the JSON mapping:',
            'The mapping is:',
            'JSON mapping:',
            'Mapping:',
        ]
        
        for prefix in prefixes_to_remove:
            if response.strip().lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        return response.strip()

    def _assign_shuffled_ids(self, objects: List[ObjectInfo]) -> List[ObjectInfo]:
        """
        Assign shuffled IDs to objects for display consistency.
        
        Args:
            objects: List of ObjectInfo instances
            
        Returns:
            List of ObjectInfo instances with shuffled_id assigned
        """
        import random
        
        # Generate shuffled IDs (1 to num_objects)
        num_objects = len(objects)
        shuffled_ids = list(range(1, num_objects + 1))
        random.shuffle(shuffled_ids)
        
        # Assign shuffled IDs to objects
        for i, obj in enumerate(objects):
            obj.shuffled_id = shuffled_ids[i]
        
        return objects
