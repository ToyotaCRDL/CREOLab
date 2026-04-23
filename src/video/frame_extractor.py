"""
Frame extraction module for extracting discrete frames from video clips.
Extracts 7 frames at specific timestamps (0s, 1s, 2s, 3s, 4s, 5s, 6s).
"""

import cv2
import os
from typing import List
from pathlib import Path


class FrameExtractor:
    """
    Extracts frames from video clips at specified intervals.
    
    Default configuration:
    - Extracts 7 frames per clip
    - Frame timestamps: 0s, 1s, 2s, 3s, 4s, 5s, 6s
    """
    
    def __init__(self, frame_timestamps: List[float] = None):
        """
        Initialize frame extractor.
        
        Args:
            frame_timestamps: List of timestamps in seconds to extract frames
                            (default: [0, 1, 2, 3, 4, 5, 6])
        """
        if frame_timestamps is None:
            self.frame_timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        else:
            self.frame_timestamps = frame_timestamps
            
    def extract_frames_from_video(self, video_path: str, output_dir: str) -> List[str]:
        """
        Extract frames from a single video file.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            
        Returns:
            List of paths to extracted frame files
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        print(f"Video properties: {total_frames} frames, {fps:.1f} fps, {video_duration:.1f}s duration")
        
        # Get base filename for output
        video_name = Path(video_path).stem
        
        output_files = []
        
        for i, timestamp in enumerate(self.frame_timestamps):
            print(f"Processing timestamp {timestamp}s (frame {i})")
            
            # Adjust timestamp if it exceeds video duration
            actual_timestamp = timestamp
            if timestamp > video_duration:
                actual_timestamp = video_duration
                print(f"Adjusted timestamp from {timestamp}s to {actual_timestamp:.1f}s (final frame)")
            elif timestamp == video_duration:
                # Handle exact duration case - use slightly earlier timestamp
                actual_timestamp = max(0, video_duration - 0.1)
                print(f"Adjusted timestamp from {timestamp}s to {actual_timestamp:.1f}s (avoid boundary)")
                
            # Calculate frame number using actual timestamp
            frame_number = int(actual_timestamp * fps)
            print(f"  Calculated frame number: {frame_number} (actual_timestamp {actual_timestamp:.1f}s * fps {fps:.1f})")
            
            # Ensure frame number is within valid range (0 to total_frames-1)
            if frame_number >= total_frames:
                frame_number = total_frames - 1
                print(f"Adjusted frame number from {int(actual_timestamp * fps)} to {frame_number} (last frame)")
            
            # Set video position to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame at {actual_timestamp:.1f}s (frame {frame_number})")
                continue
            
            # Save frame with simple naming convention
            output_path = os.path.join(output_dir, f"frame_{i}.jpg")
            cv2.imwrite(output_path, frame)
            output_files.append(output_path)
            
            print(f"Extracted frame {i}: {actual_timestamp:.1f}s -> {output_path}")
        
        cap.release()
        return output_files
    
    def extract_context_frames_from_video(self, video_path: str, output_dir: str, num_frames: int = 7) -> List[str]:
        """
        Extract context frames for enhanced object detection.
        
        Frame extraction logic:
        - Context frames: int(total_frames * n/num_frames) for n=1,2,...,num_frames
        - Note: Reference image (0 seconds) is handled separately and not included
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            num_frames: Number of context frames to extract (default: 7)
            
        Returns:
            List of paths to extracted context frame files
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        print(f"  Video properties: {total_frames} frames, {fps:.1f} fps, {video_duration:.1f}s duration")
        
        # Get base filename for output
        video_name = Path(video_path).stem
        
        output_files = []
        
        # Note: 0-second frame exists separately as reference image, so not included in context
        frame_numbers = []
        
        print(f"  Calculating context frames using formula: int({total_frames} * n/{num_frames})")
        for n in range(1, num_frames + 1):
            frame_number = int(total_frames * n / num_frames)
            if frame_number >= total_frames:
                frame_number = total_frames - 1
            print(f"    n={n}: int({total_frames} * {n}/{num_frames}) = {frame_number}")
            frame_numbers.append(frame_number)
        
        print(f"  Final context frame numbers to extract: {frame_numbers}")
        
        for i, frame_number in enumerate(frame_numbers):
            # Set video position to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Calculate timestamp for logging
            timestamp = frame_number / fps
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame at {timestamp:.1f}s")
                continue
            
            # Save frame with context naming convention
            output_path = os.path.join(output_dir, f"context_frame_{i:02d}_{timestamp:.1f}s.jpg")
            cv2.imwrite(output_path, frame)
            output_files.append(output_path)
            
            print(f"Extracted context frame {i}: {timestamp:.1f}s (frame {frame_number}) -> {output_path}")
        
        cap.release()
        return output_files
    
    def extract_first_frame(self, video_path: str, output_dir: str, output_filename: str = None) -> str:
        """
        Extract the first frame (at 0 seconds) from a video file.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save the extracted frame
            output_filename: Optional custom filename (if None, uses default naming)
            
        Returns:
            Path to the extracted first frame file
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Set video position to first frame (frame 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Could not read first frame from video: {video_path}")
        
        # Determine output filename
        if output_filename:
            output_path = os.path.join(output_dir, output_filename)
        else:
            # Default naming for backward compatibility
            output_path = os.path.join(output_dir, "reference_frame_0.0s.jpg")
        
        # Save first frame as reference image
        cv2.imwrite(output_path, frame)
        
        cap.release()
        print(f"Extracted first frame: 0.0s -> {output_path}")
        
        return output_path



