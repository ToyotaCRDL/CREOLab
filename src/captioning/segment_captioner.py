"""
Segment captioning module
Generic captioner that receives object information lists and generates procedures
"""

from typing import List, Optional, Dict
import os
import re
import json
from datetime import datetime

from src.core import create_llm_client
from src.core.base_models import ExtractedFrame, ProcedureStep, SegmentProcedure, CaptionData, ObjectInfo, create_step_id
from src.utils.prompt_loader import PromptLoader, prompt_loader


class SegmentCaptioner:
    """
    Unified Segment Captioner
    Handles both Manual and Auto object detection approaches with a single processing flow.
    Receives either ObjectInfo lists or pre-formatted text and generates procedures.
    """
    
    def __init__(self, model: Optional[str] = None, is_manual_mode: bool = False, base_prompts_dir: str = None):
        """Initialize the unified captioner.
        
        Args:
            model: LLM model to use (defaults to config setting)
            is_manual_mode: True for manual object detection mode, False for auto detection mode
            base_prompts_dir: Base directory for saving prompts (defaults to "prompts")
        """
        self.client = create_llm_client(model=model)
        self.is_manual_mode = is_manual_mode
        self.prompt_loader = PromptLoader()
        self.segment_prompt_template = prompt_loader.get_prompt("common_captioning_prompts.json", "segment_analysis")
        self.base_prompts_dir = base_prompts_dir or "prompts"


    def _format_objects_info(self, objects: List[ObjectInfo], processing_mode: str = None) -> str:
        """
        Format object information for prompt inclusion.
        Objects should already have correct shuffled_id from direct JSON loading.
        
        Args:
            objects: List of ObjectInfo instances with correct shuffled_id
            processing_mode: Processing mode (not used anymore since objects come pre-formatted)
            
        Returns:
            Formatted string with object descriptions and positions
        """
        if not objects:
            return "No objects detected in the workspace."
        
        # Sort objects by shuffled_id (objects should already have correct shuffled_id)
        sorted_objects = sorted(objects, key=lambda obj: obj.shuffled_id if obj.shuffled_id is not None else obj.id)
        
        formatted_objects = []
        for obj in sorted_objects:
            # Use shuffled_id for display (should always be available now)
            display_id = obj.shuffled_id if obj.shuffled_id is not None else obj.id
            obj_desc = f"- {display_id}: {obj.description}"
            if obj.position:
                # Format position with 2 decimal places
                x_pos = round(obj.position[0], 2)
                y_pos = round(obj.position[1], 2)
                obj_desc += f" (at position ({x_pos:.2f}, {y_pos:.2f}))"
            formatted_objects.append(obj_desc)
        
        return "\n".join(formatted_objects)
    
    
    def _save_segment_logs(self, segment_id: str, processing_mode: str, image_paths: List[str], prompt: str, response: str):
        """
        Save segment captioning logs including images, prompt, and response.
        Creates organized subfolder structure: prompts/{processing_mode}/{segment_id}/
        
        Args:
            segment_id: Segment identifier
            processing_mode: Processing mode (auto_detection or manual_detection)
            image_paths: List of image file paths sent with the prompt
            prompt: The prompt text sent to the LLM
            response: The response received from the LLM
        """
        import os
        import shutil
        from datetime import datetime
        
        # Create prompts directory structure using the configured base directory
        mode_dir = os.path.join(self.base_prompts_dir, processing_mode)
        segment_dir = os.path.join(mode_dir, segment_id)
        
        # Create directories if they don't exist
        os.makedirs(segment_dir, exist_ok=True)
        
        # Save prompt
        prompt_file = os.path.join(segment_dir, "prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Segment ID: {segment_id}\n")
            f.write(f"Processing Mode: {processing_mode}\n")
            f.write(f"Number of Images: {len(image_paths)}\n")
            f.write("=" * 50 + "\n")
            f.write("PROMPT:\n")
            f.write("=" * 50 + "\n")
            f.write(prompt)
        
        # Save response
        response_file = os.path.join(segment_dir, "response.txt")
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Segment ID: {segment_id}\n")
            f.write(f"Processing Mode: {processing_mode}\n")
            f.write("=" * 50 + "\n")
            f.write("RESPONSE:\n")
            f.write("=" * 50 + "\n")
            f.write(response)
        
        # Copy images to the segment directory
        images_dir = os.path.join(segment_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            if os.path.exists(image_path):
                # Create descriptive filename
                if i == 0:
                    # First image is the reference overlay
                    filename = f"00_reference_overlay_{os.path.basename(image_path)}"
                else:
                    # Subsequent images are segment frames
                    filename = f"{i:02d}_segment_frame_{os.path.basename(image_path)}"
                
                dest_path = os.path.join(images_dir, filename)
                shutil.copy2(image_path, dest_path)
            else:
                print(f"Warning: Image not found for logging: {image_path}")
        
        # Create summary file
        summary_file = os.path.join(segment_dir, "summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"SEGMENT CAPTIONING LOG SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Segment ID: {segment_id}\n")
            f.write(f"Processing Mode: {processing_mode}\n")
            f.write(f"Total Images: {len(image_paths)}\n")
            f.write(f"Reference Image: {os.path.basename(image_paths[0]) if image_paths else 'None'}\n")
            f.write(f"Segment Frames: {len(image_paths) - 1 if len(image_paths) > 1 else 0}\n")
            f.write(f"Prompt Length: {len(prompt)} characters\n")
            f.write(f"Response Length: {len(response)} characters\n")
            f.write("\nFiles:\n")
            f.write("- prompt.txt: The prompt sent to the LLM\n")
            f.write("- response.txt: The response received from the LLM\n")
            f.write("- images/: Directory containing all images sent with the prompt\n")
            f.write("  - 00_reference_overlay_*: Reference image with object overlays\n")
            f.write("  - 01_segment_frame_*, 02_segment_frame_*, ...: Video segment frames\n")
        
        print(f"  ✓ Saved segment logs: {segment_dir}")
            
    
    def caption_segment(self,
                       frames: List[ExtractedFrame], 
                       segment_id: str,
                       objects: List[ObjectInfo],
                       processing_mode: str = None) -> SegmentProcedure:
        """
        Unified procedure generation method - single data flow for all approaches.
        
        Args:
            frames: List of extracted frames from the segment
            segment_id: Segment identifier
            objects: ObjectInfo list (required for both Manual and Auto detection)
            processing_mode: Processing mode to set (auto_detection or knowledge_enhanced)
            
        Returns:
            SegmentProcedure with generated steps
        """
        # Determine processing mode first
        if processing_mode is None:
            # Auto-detect based on is_manual_mode flag
            processing_mode = "manual_detection" if self.is_manual_mode else "auto_detection"
        
        # Single unified flow: ObjectInfo list → formatted text → prompt → LLM → steps
        objects_info = self._format_objects_info(objects, processing_mode)
        num_segment_frames = len(frames) - 1  # exclude reference overlay
        frame_interval = 6.0 / (num_segment_frames - 1) if num_segment_frames > 1 else 1.0
        prompt = self.segment_prompt_template.format(
            objects_info=objects_info,
            frame_interval=f"{frame_interval:.1f}",
            num_segment_frames=str(num_segment_frames),
        )
        image_paths = [frame.image_path for frame in frames]
        
        response = self.client.analyze_frames(
            frame_paths=image_paths,
            prompt=prompt
        )
        
        # Save prompt, images, and response immediately after getting response
        self._save_segment_logs(segment_id, processing_mode, image_paths, prompt, response)
        
        steps = self._parse_response_to_steps(response, segment_id)
        
        return SegmentProcedure(
            segment_id=segment_id,
            frames=frames,
            steps=steps,
            processing_mode=processing_mode,
            used_prompt=prompt,
            llm_response=response
        )

    def caption_multiple_segments(self, 
                                 segments_frames: List[List[ExtractedFrame]], 
                                 objects: List[ObjectInfo],
                                 processing_mode: str = None) -> List[SegmentProcedure]:
        """
        Generate procedures for multiple segments using unified approach.
        
        Args:
            segments_frames: List of frame lists for each segment
            objects: ObjectInfo list (same for all segments)
            processing_mode: Processing mode to set (auto_detection or knowledge_enhanced)
            
        Returns:
            List of SegmentProcedure instances
        """
        results = []
        for i, frames in enumerate(segments_frames):
            segment_id = f"segment_{i+1:03d}"
            procedure = self.caption_segment(
                frames=frames,
                segment_id=segment_id,
                objects=objects,
                processing_mode=processing_mode
            )
            results.append(procedure)
            print(f"  ✓ Generated procedure for {segment_id} ({len(procedure.steps)} steps)")
        
        return results

    def _parse_response_to_steps(self, response: str, segment_id: str) -> List[ProcedureStep]:
        """
        Parse LLM response into structured procedure steps.
        """
        steps = []
        lines = response.strip().split('\n')

        step_counter = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for numbered steps (e.g., "1. Take the red book")
            step_match = re.match(r'^(\d+)\.\s*(.+)', line)
            if step_match:
                step_counter += 1
                step_number = int(step_match.group(1))
                description = step_match.group(2).strip()
                
                # Remove object numbers from description
                clean_description = self._remove_object_numbers(description)
                
                step_id = create_step_id(segment_id, step_counter)
                step = ProcedureStep(
                    step_id=step_id,
                    description=clean_description
                )
                steps.append(step)
        
        return steps

    def _remove_object_numbers(self, text: str) -> str:
        """Remove object reference numbers from text (e.g., '(3)' or 'object (3)')."""
        # Remove patterns like "(3)" or " (3)" at the end of object names
        text = re.sub(r'\s*\(\d+\)\s*', ' ', text)
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text