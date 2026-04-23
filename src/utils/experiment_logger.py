"""
Experiment logger for creating timestamped output directories and saving experiment conditions.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from .config_loader import config_loader


class ExperimentLogger:
    """
    Manages experiment output directories and logging of experimental conditions.
    """
    
    def __init__(self, base_output_dir: str = "output", batch_mode: bool = False, split_name: str = None):
        """
        Initialize experiment logger.
        
        Args:
            base_output_dir: Base directory for all outputs
            batch_mode: Whether running in batch mode
            split_name: Name of the dataset split (dev/test) for batch mode
        """
        self.base_output_dir = base_output_dir
        self.batch_mode = batch_mode
        self.split_name = split_name
        self.experiment_dir = None
        self.take_dir = None
        self.timestamp = None
        self.experiment_config = {}
    
    def create_experiment_directory(self, take_name: str = None) -> str:
        """
        Create a timestamped experiment directory.
        
        Args:
            take_name: Name of the take (scenario##_decoy# format, for batch mode)
        
        Returns:
            Path to the created experiment directory
        """
        # Generate timestamp in format: YYYYMMDD_HHMMSS
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        provider = config_loader.get_provider()
        
        if self.batch_mode:
            # Batch mode: provider/batch_TIMESTAMP/SPLIT/TAKE/
            prefix = "batch"
            self.experiment_dir = os.path.join(self.base_output_dir, provider, f"{prefix}_{self.timestamp}")
            if self.split_name:
                self.experiment_dir = os.path.join(self.experiment_dir, self.split_name)
            if take_name:
                self.take_dir = os.path.join(self.experiment_dir, take_name)
                Path(self.take_dir).mkdir(parents=True, exist_ok=True)
                return self.take_dir
        else:
            # Single mode: provider/single_TIMESTAMP/TAKE/
            prefix = "single"
            self.experiment_dir = os.path.join(self.base_output_dir, provider, f"{prefix}_{self.timestamp}")
            if take_name:
                self.take_dir = os.path.join(self.experiment_dir, take_name)
                Path(self.take_dir).mkdir(parents=True, exist_ok=True)
                return self.take_dir
        
        # Create base experiment directory
        Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)
        return self.experiment_dir
    
    def create_iteration_directory(self, iteration_num: int) -> str:
        """
        Create directory for a specific iteration.
        
        Args:
            iteration_num: Iteration number (1-based)
            
        Returns:
            Path to the iteration directory
        """
        if not self.take_dir:
            raise ValueError("Take directory not created. Call create_experiment_directory() with take_name first.")
        
        iter_dir = os.path.join(self.take_dir, f"iter_{iteration_num:02d}")
        Path(iter_dir).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for this iteration
        subdirs = ["integrations", "evaluation", "prompts"]
        for subdir in subdirs:
            Path(os.path.join(iter_dir, subdir)).mkdir(exist_ok=True)
        
        return iter_dir
    
    def create_aggregate_directory(self) -> str:
        """
        Create directory for aggregate results.
        
        Returns:
            Path to the aggregate directory
        """
        if not self.take_dir:
            raise ValueError("Take directory not created. Call create_experiment_directory() with take_name first.")
        
        agg_dir = os.path.join(self.take_dir, "aggregate")
        Path(agg_dir).mkdir(parents=True, exist_ok=True)
        return agg_dir
    
    def log_experiment_conditions(self, 
                                 video_path: str,
                                 mode: str,
                                 max_duration: float,
                                 caption_file: Optional[str] = None,
                                 additional_config: Optional[Dict[str, Any]] = None):
        """
        Save experiment conditions to a JSON file.
        
        Args:
            video_path: Path to input video
            mode: Processing mode
            max_duration: Maximum analysis duration
            caption_file: Caption file used (if any)
            additional_config: Additional configuration parameters
        """
        if not self.experiment_dir:
            raise ValueError("Experiment directory not created. Call create_experiment_directory() first.")
        
        # Collect experiment conditions
        self.experiment_config = {
            "experiment_info": {
                "timestamp": self.timestamp,
                "experiment_id": f"experiment_{self.timestamp}",
                "created_at": datetime.now().isoformat()
            },
            "input_parameters": {
                "video_path": video_path,
                "video_filename": os.path.basename(video_path),
                "caption_file": caption_file,
                "processing_mode": mode,
                "max_duration_seconds": max_duration
            },
            "processing_settings": {
                "clip_duration": 6.0,
                "stride": 5.0,
                "frames_per_clip": config_loader.get_max_images() or 7,
                "reference_image_enabled": True
            },
            "llm_config": {
                "provider": config_loader.get_provider(),
                "model": config_loader.get_model(),
                "max_tokens": config_loader.get_llm_settings().get("max_tokens"),
            },
            "system_info": {
                "python_version": None,  # Can be filled if needed
                "opencv_version": None,  # Can be filled if needed
            }
        }
        
        # Add additional configuration if provided
        if additional_config:
            self.experiment_config["additional_config"] = additional_config
        
        # Save to JSON file
        config_path = os.path.join(self.experiment_dir, "experiment_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_config, f, indent=2, ensure_ascii=False)
        
        print(f"Experiment conditions saved: {config_path}")
    
    def get_subdirectory(self, subdir_name: str) -> str:
        """
        Get path to a subdirectory within the experiment directory.
        
        Args:
            subdir_name: Name of subdirectory
            
        Returns:
            Full path to subdirectory
        """
        if not self.experiment_dir:
            raise ValueError("Experiment directory not created.")
        
        return os.path.join(self.experiment_dir, subdir_name)
    
    def get_subdir(self, subdir_name: str) -> str:
        """
        Get path to a subdirectory, automatically handling take_dir or experiment_dir.
        Creates the directory if it doesn't exist.
        
        Args:
            subdir_name: Name of subdirectory (e.g., "segments", "frames", "prompts")
            
        Returns:
            Full path to subdirectory (directory is created)
        """
        # Use take_dir if available (batch mode with take), otherwise experiment_dir
        base = self.take_dir if self.take_dir else self.experiment_dir
        if not base:
            raise ValueError("Experiment directory not created. Call create_experiment_directory() first.")
        
        path = os.path.join(base, subdir_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
    
    def log_processing_results(self, 
                              segments_created: int,
                              frames_extracted: int,
                              vision_only_results: Optional[int] = None,
                              knowledge_enhanced_results: Optional[int] = None,
                              integrations_created: Optional[int] = None):
        """
        Log processing results summary.
        
        Args:
            segments_created: Number of video segments created
            frames_extracted: Total number of frames extracted
            vision_only_results: Number of vision-only procedures generated
            knowledge_enhanced_results: Number of knowledge-enhanced procedures generated
            integrations_created: Number of integrated procedures created
        """
        if not self.experiment_dir:
            return
        
        results = {
            "processing_results": {
                "segments_created": segments_created,
                "frames_extracted": frames_extracted,
                "vision_only_procedures": vision_only_results,
                "knowledge_enhanced_procedures": knowledge_enhanced_results,
                "integrated_procedures": integrations_created,
                "completed_at": datetime.now().isoformat()
            }
        }
        
        # Update experiment config with results
        self.experiment_config.update(results)
        
        # Save updated config
        config_path = os.path.join(self.experiment_dir, "experiment_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_config, f, indent=2, ensure_ascii=False)
    
    def get_experiment_summary(self) -> str:
        """
        Generate a human-readable experiment summary.
        
        Returns:
            Formatted summary string
        """
        if not self.experiment_config:
            return "No experiment data available."
        
        exp_info = self.experiment_config.get("experiment_info", {})
        input_params = self.experiment_config.get("input_parameters", {})
        llm_cfg = self.experiment_config.get("llm_config", {})
        results = self.experiment_config.get("processing_results", {})
        
        summary = f"""
=== Experiment Summary ===
Experiment ID: {exp_info.get('experiment_id', 'N/A')}
Execution Time: {exp_info.get('created_at', 'N/A')}
Output Directory: {self.experiment_dir}

=== LLM Configuration ===
Provider: {llm_cfg.get('provider', 'N/A')}
Model: {llm_cfg.get('model', 'N/A')}

=== Input Parameters ===
Video File: {input_params.get('video_filename', 'N/A')}
Processing Mode: {input_params.get('processing_mode', 'N/A')}
Max Analysis Duration: {input_params.get('max_duration_seconds', 'N/A')} seconds

=== Processing Results ===
Segments Created: {results.get('segments_created', 'N/A')}
Frames Extracted: {results.get('frames_extracted', 'N/A')}
Integrated Procedures: {results.get('integrated_procedures', 'N/A')}
        """.strip()
        
        return summary
