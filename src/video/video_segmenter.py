"""
Video segmentation module for processing experiment videos.
Segments videos into overlapping clips (6s duration, 5s stride, 1s overlap).
"""

import cv2
import os
from typing import List, Tuple, Optional
from pathlib import Path


class VideoSegmenter:
    """
    Segments videos into overlapping clips for procedure analysis.
    
    Default configuration:
    - Clip duration: 6 seconds
    - Stride: 5 seconds  
    - Overlap: 1 second
    """
    
    def __init__(self, clip_duration: float = 6.0, stride: float = 5.0):
        """
        Initialize video segmenter.
        
        Args:
            clip_duration: Duration of each clip in seconds (default: 6.0)
            stride: Time stride between clips in seconds (default: 5.0)
        """
        self.clip_duration = clip_duration
        self.stride = stride
        self.overlap = clip_duration - stride
        
    def get_segment_timestamps(self, video_duration: float) -> List[Tuple[float, float]]:
        """
        Calculate segment timestamps for the given video duration.
        
        Args:
            video_duration: Total duration of the video in seconds
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        segments = []
        start_time = 0.0
        
        while start_time < video_duration:
            end_time = min(start_time + self.clip_duration, video_duration)
            segments.append((start_time, end_time))
            
            # If this segment reaches the end of the video, no more segments needed
            if end_time >= video_duration:
                break
            
            # Move to next segment
            start_time += self.stride
            
            # Break if the next segment would be too short
            if start_time >= video_duration:
                break
                
        return segments
    
    def segment_video(self, video_path: str, output_dir: str, max_duration: Optional[float] = None) -> List[str]:
        """
        Segment video into clips and save them.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save segmented clips
            max_duration: Maximum duration to process (None for full video)
            
        Returns:
            List of paths to generated clip files
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
        
        # Apply duration limit if specified
        actual_duration = video_duration
        if max_duration is not None:
            actual_duration = min(video_duration, max_duration)
            print(f"  Duration limited to {actual_duration:.1f}s (original: {video_duration:.1f}s)")
        
        # Get segment timestamps
        segments = self.get_segment_timestamps(actual_duration)
        
        # Get base filename for output
        video_name = Path(video_path).stem
        
        output_files = []
        
        for i, (start_time, end_time) in enumerate(segments):
            # Calculate frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Set video position to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Output video path
            output_path = os.path.join(output_dir, f"{video_name}_segment_{i:03d}.mp4")
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
            # Write frames
            frame_count = 0
            while frame_count < (end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                out.write(frame)
                frame_count += 1
            
            out.release()
            output_files.append(output_path)
            
            print(f"Created segment {i}: {start_time:.1f}s - {end_time:.1f}s -> {output_path}")
        
        cap.release()
        return output_files
    
    def get_video_info(self, video_path: str) -> dict:
        """Get basic video information."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        return {'duration': duration, 'fps': fps}
