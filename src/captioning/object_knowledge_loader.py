"""
Object knowledge loader for reading caption data from JSON files.
Loads object information and captions for knowledge-enhanced processing.
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path

from src.core.base_models import CaptionData, ObjectInfo


class ObjectKnowledgeLoader:
    """
    Loads object knowledge and caption data from JSON files.
    """
    
    def __init__(self, captions_dir: str = "data/captions"):
        """
        Initialize object knowledge loader.
        
        Args:
            captions_dir: Directory containing caption JSON files
        """
        self.captions_dir = captions_dir
        self._caption_cache = {}
    
    def load_caption_data(self, json_file_path: str) -> Optional[CaptionData]:
        """
        Load caption data from a specific JSON file path.
        This method is for reference caption loading, not for segment captioning.
        
        Args:
            json_file_path: Path to the JSON caption file
            
        Returns:
            CaptionData object or None if not found
        """
        # Normalize path for caching
        normalized_path = os.path.abspath(json_file_path)
        
        # Check cache first
        if normalized_path in self._caption_cache:
            return self._caption_cache[normalized_path]
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # For reference caption loading, use normal CaptionData creation
        caption_data = CaptionData.from_dict(data, auto_assign_shuffled_id=True)
        
        # Cache the result
        self._caption_cache[normalized_path] = caption_data
        
        return caption_data
    
    def load_objects_from_legend(self, legend_file_path: str) -> List[ObjectInfo]:
        """
        Load objects directly from a legend JSON file.
        
        Args:
            legend_file_path: Direct path to the legend JSON file
            
        Returns:
            List of ObjectInfo instances loaded from the file
        """
        with open(legend_file_path, 'r', encoding='utf-8') as f:
            legend_data = json.load(f)
        
        print(f"  Loading objects directly from legend file: {legend_file_path}")
        print(f"  Legend data contains {len(legend_data)} objects")
        
        # Create objects directly from legend data
        objects = []
        for legend_obj in legend_data:
            obj = ObjectInfo(
                id=legend_obj["shuffled_id"],
                description=legend_obj["description"],
                position=(legend_obj["x"], legend_obj["y"]),
                shuffled_id=legend_obj["shuffled_id"]
            )
            objects.append(obj)
            print(f"    Loaded: {obj.shuffled_id}. {obj.description} at ({obj.position[0]:.2f}, {obj.position[1]:.2f})")
        
        if not objects:
            raise RuntimeError(f"CRITICAL ERROR: No valid objects loaded from legend file {legend_file_path}.")
        
        print(f"  ✓ Loaded {len(objects)} objects directly from legend file")
        return objects
