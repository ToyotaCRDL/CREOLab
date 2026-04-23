"""
Base data models for the video procedure generation system.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class VideoSegment:
    """Represents a video segment with metadata."""
    segment_id: str
    video_path: str
    start_time: float
    end_time: float
    duration: float
    segment_index: int


@dataclass
class ExtractedFrame:
    """Represents an extracted frame with metadata."""
    frame_id: str
    image_path: str
    timestamp: float
    frame_index: int
    segment_id: str


@dataclass
class ObjectInfo:
    """Represents object information from caption data."""
    id: int
    description: str
    position: Tuple[float, float]  # (x, y) normalized coordinates
    shuffled_id: Optional[int] = None  # Shuffled ID for display consistency


@dataclass
class ProcedureStep:
    """Represents a single procedure step."""
    step_id: str
    description: str
    source: Optional[str] = None  # 'abstraction' or 'vision_knowledge'


@dataclass
class SegmentProcedure:
    """Represents procedure steps for a video segment."""
    segment_id: str
    frames: List[ExtractedFrame]
    steps: List[ProcedureStep]
    objects: Optional[List[ObjectInfo]] = None
    processing_mode: str = "manual_detection"  # Processing mode used
    used_prompt: Optional[str] = None  # The actual prompt sent to the LLM
    llm_response: Optional[str] = None  # The raw response from the LLM



@dataclass
class CaptionData:
    """Represents caption data loaded from JSON files."""
    video_id: str
    objects: List[ObjectInfo]
    caption: str
    
    @classmethod
    def from_dict(cls, data: Dict, auto_assign_shuffled_id: bool = True) -> 'CaptionData':
        """Create CaptionData from dictionary."""
        import random
        
        objects = [
            ObjectInfo(
                id=obj.get('id', idx + 1),  # Use provided id or sequential numbering
                description=obj['description'],
                position=(obj['position'][0], obj['position'][1])
            )
            for idx, obj in enumerate(data.get('objects', []))
        ]
        
        # Assign shuffled IDs for display consistency only if requested
        # For manual detection, shuffled_id should be loaded from object_legend.json
        if auto_assign_shuffled_id and objects:
            num_objects = len(objects)
            shuffled_ids = list(range(1, num_objects + 1))
            random.shuffle(shuffled_ids)
            
            for i, obj in enumerate(objects):
                obj.shuffled_id = shuffled_ids[i]
        
        return cls(
            video_id=data['video_id'],
            objects=objects,
            caption=data.get('caption', '')
        )


def create_frame_id(segment_id: str, frame_index: int, timestamp: float) -> str:
    """Create a unique frame ID."""
    return f"{segment_id}_frame_{frame_index:02d}_{timestamp:.0f}s"


def create_segment_id(video_name: str, segment_index: int) -> str:
    """Create a unique segment ID."""
    return f"{video_name}_segment_{segment_index:03d}"


def create_step_id(segment_id: str, step_index: int) -> str:
    """Create a unique step ID."""
    return f"{segment_id}_step_{step_index:02d}"
