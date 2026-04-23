"""
Pipeline step functions for procedure generation.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from src.core.base_models import ExtractedFrame, create_segment_id, create_frame_id
from src.video.video_segmenter import VideoSegmenter
from src.video.frame_extractor import FrameExtractor
from src.utils.reference_image_generator import ReferenceImageGenerator
from src.captioning.object_knowledge_loader import ObjectKnowledgeLoader
from src.captioning.segment_captioner import SegmentCaptioner
from src.captioning.auto_object_detector import AutoObjectDetector
from src.core.base_client import BaseLLMClient
from src.core.procedure_integrator import ProcedureIntegrator
from src.evaluation.procedure_evaluator import ProcedureEvaluator
from src.utils.config_loader import config_loader

from io_utils import save_text, save_integration_result, save_error_log


# ============================================================================
# Helper Functions
# ============================================================================

def process_segment_captioning(
    detection_type: str,
    legend_json_path: str,
    overlay_path: str,
    segments_frames: List,
    knowledge_loader: ObjectKnowledgeLoader,
    prompts_dir: str
) -> List:
    """
    Unified segment captioning processing for both Auto and Manual detection.
    
    Args:
        detection_type: "auto_detection" or "manual_detection"
        legend_json_path: Path to the legend JSON file
        overlay_path: Path to the overlay image
        segments_frames: List of segment frames
        knowledge_loader: Object knowledge loader instance
        prompts_dir: Directory for saving prompts
        
    Returns:
        List of segment procedure results
    """
    print(f"  Generating {detection_type} procedures...")
    
    # Initialize captioner
    is_manual_mode = (detection_type == "manual_detection")
    captioner = SegmentCaptioner(is_manual_mode=is_manual_mode, base_prompts_dir=prompts_dir)
    
    # Load objects from legend file
    print(f"  Loading {detection_type} objects from file...")
    objects = knowledge_loader.load_objects_from_legend(legend_json_path)
    if not objects:
        raise RuntimeError(f"CRITICAL ERROR: No objects found in {detection_type} legend file")
    
    # Insert overlay image as first frame for each segment
    segments_frames_with_overlay = []
    
    for segment_frames in segments_frames:
        if segment_frames:  # Only process non-empty segments
            # Create reference frame with overlay image
            reference_frame = ExtractedFrame(
                frame_id=f"reference_{detection_type}",
                image_path=overlay_path,
                timestamp=0.0,
                frame_index=-1,  # Use -1 to indicate reference frame
                segment_id=segment_frames[0].segment_id
            )
            # Insert reference frame at the beginning
            frames_with_overlay = [reference_frame] + segment_frames
            segments_frames_with_overlay.append(frames_with_overlay)
        else:
            segments_frames_with_overlay.append(segment_frames)
    
    # Generate segment procedures
    results = captioner.caption_multiple_segments(
        segments_frames=segments_frames_with_overlay,
        objects=objects,
        processing_mode=detection_type
    )
    print(f"  ✓ Generated {detection_type} procedures for {len(results)} segments")
    
    return results


# ============================================================================
# Pipeline Step Functions
# ============================================================================


def prepare_references(
    video_path: str,
    video_name: str,
    caption_file: Optional[str],
    ref_images_dir: str,
    knowledge_loader: ObjectKnowledgeLoader
) -> Tuple[Optional[str], Optional[str], Any, Optional[List]]:
    """
    Prepare reference images and legend for object detection.
    
    Args:
        video_path: Path to input video
        video_name: Video stem name
        caption_file: Path to caption JSON file
        ref_images_dir: Directory for reference images
        knowledge_loader: Knowledge loader instance
        
    Returns:
        tuple: (reference_with_overlay, reference_without_overlay, caption_data, random_numbers)
    """
    print("Step 0: Generating reference images...")
    
    reference_generator = ReferenceImageGenerator()
    frame_extractor = FrameExtractor()
    
    reference_image_with_overlay = None
    reference_image_without_overlay = None
    caption_data = None
    random_numbers = None
    
    # Load caption data
    caption_data = knowledge_loader.load_caption_data(caption_file)
    print(f"  Loaded caption data from: {caption_file}")
    
    # Generate reference image with overlay
    if caption_data:
        overlay_path = os.path.join(ref_images_dir, f"{video_name}_reference_with_overlay.jpg")
        success, overlay_random_numbers = reference_generator.create_reference_image_with_overlay(
            video_path, caption_data, overlay_path
        )
        if success:
            reference_image_with_overlay = overlay_path
            random_numbers = overlay_random_numbers
            print(f"  Created reference image with overlay: {overlay_path}")
            
            # Generate JSON legend
            legend_json_path = os.path.join(ref_images_dir, f"{video_name}_object_legend.json")
            reference_generator.generate_object_legend_json(caption_data, legend_json_path, random_numbers)
            print(f"  Created object legend JSON: {legend_json_path}")
        else:
            print(f"  Failed to create reference image with overlay")
    else:
        print(f"  No caption data found from {caption_file}, skipping overlay generation")
        
    # Generate reference image without overlay
    no_overlay_path = os.path.join(ref_images_dir, f"{video_name}_reference_no_overlay.jpg")
    os.makedirs(ref_images_dir, exist_ok=True)
    output_filename = f"{video_name}_reference_no_overlay.jpg"
    reference_image_path = frame_extractor.extract_first_frame(video_path, ref_images_dir, output_filename)
    reference_image_without_overlay = reference_image_path
    print(f"  Created reference image without overlay: {reference_image_path}")
    
    return reference_image_with_overlay, reference_image_without_overlay, caption_data, random_numbers


def segment_and_extract_frames(
    video_path: str,
    video_name: str,
    segments_dir: str,
    frames_dir: str,
    max_duration: Optional[float] = None
) -> List[Tuple[str, List[ExtractedFrame]]]:
    """
    Segment video and extract frames from each segment.
    
    Args:
        video_path: Path to input video
        video_name: Video stem name
        segments_dir: Directory for segments
        frames_dir: Directory for frames
        max_duration: Maximum duration in seconds (None = entire video)
        
    Returns:
        list: List of (segment_id, frames) tuples
    """
    # Step 1: Video Segmentation
    print("Step 1: Segmenting video...")
    segmenter = VideoSegmenter(clip_duration=6.0, stride=5.0)
    
    max_images = config_loader.get_max_images()
    if max_images:
        num_segment_frames = max_images - 1
        segment_duration = 6.0
        timestamps = [segment_duration * i / (num_segment_frames - 1) for i in range(num_segment_frames)]
        frame_extractor = FrameExtractor(frame_timestamps=timestamps)
    else:
        frame_extractor = FrameExtractor()
    
    # Get video info
    video_info = segmenter.get_video_info(video_path)
    if max_duration is None:
        actual_duration = video_info['duration']
    else:
        actual_duration = min(video_info['duration'], max_duration)
    
    print(f"  Video duration: {video_info['duration']:.1f}s")
    print(f"  Analysis duration: {actual_duration:.1f}s")
    
    # Calculate expected segments
    segment_timestamps = segmenter.get_segment_timestamps(actual_duration)
    print(f"  Expected segments: {len(segment_timestamps)}")
    
    # Segment the video
    segment_files = segmenter.segment_video(video_path, segments_dir, max_duration=max_duration)
    print(f"  Created {len(segment_files)} segments")
    
    # Step 2: Frame Extraction
    print("\nStep 2: Extracting frames...")
    all_segment_data = []
    
    for i, segment_file in enumerate(segment_files):
        segment_name = Path(segment_file).stem
        segment_id = create_segment_id(video_name, i)
        segment_frames_dir = os.path.join(frames_dir, segment_name)
        
        # Get segment start time
        if i < len(segment_timestamps):
            segment_start_time, segment_end_time = segment_timestamps[i]
        else:
            segment_start_time = i * segmenter.stride
        
        # Extract frames
        frame_files = frame_extractor.extract_frames_from_video(segment_file, segment_frames_dir)
        print(f"  Segment {i}: extracted {len(frame_files)} frames")
        
        # Create ExtractedFrame objects
        frames = []
        for j, frame_file in enumerate(frame_files):
            actual_timestamp = segment_start_time + frame_extractor.frame_timestamps[j] if j < len(frame_extractor.frame_timestamps) else segment_start_time + float(j)
            frame_id = create_frame_id(segment_id, j, actual_timestamp)
            frames.append(ExtractedFrame(
                frame_id=frame_id,
                image_path=frame_file,
                timestamp=actual_timestamp,
                frame_index=j,
                segment_id=segment_id
            ))
        
        all_segment_data.append((segment_id, frames))
    
    return all_segment_data


def generate_procedures(
    video_path: str,
    video_name: str,
    all_segment_data: List[Tuple[str, List[ExtractedFrame]]],
    ref_images_dir: str,
    reference_image_with_overlay: Optional[str],
    prompts_dir: str,
    openai_client: BaseLLMClient,
    knowledge_loader: ObjectKnowledgeLoader
) -> Tuple[Any, Any, Optional[str]]:
    """
    Generate procedures using auto and manual object detection.
    
    Args:
        video_path: Path to input video
        video_name: Video stem name
        all_segment_data: List of (segment_id, frames) tuples
        ref_images_dir: Reference images directory
        reference_image_with_overlay: Path to manual overlay image
        prompts_dir: Directory for prompts
        openai_client: OpenAI client instance
        knowledge_loader: Knowledge loader instance
        
    Returns:
        tuple: (manual_results, auto_results, mapping_file_path)
    """
    print(f"\nStep 3: Generating procedures (manual_auto mode)...")
    
    # Generate auto object detection procedures
    print("  Generating auto object detection procedures...")
    auto_detector = AutoObjectDetector(openai_client)
    
    # Enhanced object detection
    print("  Using enhanced multi-frame object detection...")
    detected_objects = auto_detector.detect_objects_from_multi_frame_context(
        video_path, video_name, ref_images_dir
    )
    
    # Pre-generate coordinate adjustment
    print("  Pre-generating coordinate adjustment...")
    reference_image_path = os.path.join(ref_images_dir, f"{video_name}_reference_no_overlay.jpg")
    
    adjusted_overlay_path, auto_legend_json_path, adjusted_objects = auto_detector.generate_adjusted_overlay_and_list(
        reference_image_path, detected_objects, ref_images_dir, video_name, video_path
    )
    
    # Generate object name mapping if manual objects are available
    manual_overlay_path = os.path.join(ref_images_dir, f"{video_name}_reference_with_overlay.jpg")
    manual_legend_json_path = os.path.join(ref_images_dir, f"{video_name}_object_legend.json")
    
    mapping_file_path = None
    if os.path.exists(manual_overlay_path) and os.path.exists(manual_legend_json_path):
        print("  Generating object name mapping between auto-detected and manual objects...")
        auto_overlay_path = adjusted_overlay_path
        print(f"    Auto overlay path: {auto_overlay_path}")
        
        mapping_file_path = auto_detector.generate_object_name_mapping(
            adjusted_objects, manual_overlay_path, manual_legend_json_path, 
            ref_images_dir, video_name, auto_overlay_path
        )
        print(f"  ✓ Generated object name mapping: {mapping_file_path}")
    else:
        print("  Manual overlay or legend not found, skipping object name mapping generation")
    
    # Prepare segments frames for processing
    segments_frames = [frames for segment_id, frames in all_segment_data]
    
    # Generate auto object detection procedures
    auto_results = process_segment_captioning(
        detection_type="auto_detection",
        legend_json_path=auto_legend_json_path,
        overlay_path=adjusted_overlay_path,
        segments_frames=segments_frames,
        knowledge_loader=knowledge_loader,
        prompts_dir=prompts_dir
    )
    
    # Generate manual object detection procedures
    manual_results = None
    if reference_image_with_overlay and manual_legend_json_path and os.path.exists(manual_legend_json_path):
        manual_results = process_segment_captioning(
            detection_type="manual_detection", 
            legend_json_path=manual_legend_json_path,
            overlay_path=reference_image_with_overlay,
            segments_frames=segments_frames,
            knowledge_loader=knowledge_loader,
            prompts_dir=prompts_dir
        )
    else:
        print("  Skipping manual object detection: required files not available")
    
    print(f"✓ Procedure generation completed:")
    print(f"  - Manual detection: {len(manual_results) if manual_results else 0} segments")
    print(f"  - Auto detection: {len(auto_results)} segments")
    
    return manual_results, auto_results, mapping_file_path


def integrate_procedures(
    manual_results: Any,
    auto_results: Any,
    experiment_dir: str,
    integration_dir: str,
    prompts_dir: str,
    openai_client: BaseLLMClient,
    integration_types: List[str]
) -> Dict[str, str]:
    """
    Integrate procedures and save results.
    
    Args:
        manual_results: Manual detection results
        auto_results: Auto detection results
        experiment_dir: Experiment directory
        integration_dir: Integration output directory
        prompts_dir: Prompts directory
        openai_client: OpenAI client instance
        integration_types: List of integration type keys
        
    Returns:
        dict: Integration results
    """
    print(f"\nStep 4: Integrating procedures...")
    
    integrator = ProcedureIntegrator(openai_client, max_retries=3)
    
    print("  Executing integration process...")
    integration_results = integrator.generate_two_method_integrations(
        manual_results, auto_results, experiment_dir
    )
    print("  ✓ Integration process completed successfully")
    
    # Save integration results
    for integration_type in integration_types:
        if integration_type in integration_results:
            # Save integrated procedure
            output_file = os.path.join(integration_dir, f"{integration_type}_integrated_procedure.txt")
            integration_content = integration_results[integration_type]
            
            if integration_content:
                save_integration_result(output_file, integration_content, integration_type)
                print(f"    Saved {integration_type}: {output_file} ({len(integration_content)} chars)")
            else:
                warning = (f"[WARNING: Integration result was empty or None]\n"
                            f"Integration content type: {type(integration_content)}\n"
                            f"Integration content repr: {repr(integration_content)}")
                save_integration_result(output_file, warning, integration_type)
                print(f"    Saved {integration_type}: {output_file} (EMPTY)")
            
            # Save integration prompt and response if available
            prompt_key = f"{integration_type}_prompt"
            response_key = f"{integration_type}_response"
            
            if prompt_key in integration_results:
                prompt_file = os.path.join(prompts_dir, f"integration_{integration_type}_prompt.txt")
                header = f"=== {integration_type.upper()} INTEGRATION PROMPT ==="
                save_text(prompt_file, integration_results[prompt_key], header=header)
            
            if response_key in integration_results:
                response_file = os.path.join(prompts_dir, f"integration_{integration_type}_response.txt")
                header = f"=== {integration_type.upper()} INTEGRATION RESPONSE ==="
                save_text(response_file, integration_results[response_key], header=header)
    
    return integration_results


def evaluate_procedures(
    integration_results: Dict[str, str],
    caption_file: Optional[str],
    mapping_file_path: Optional[str],
    output_dir: str,
    openai_client: BaseLLMClient,
    proc_manual: str,
    proc_auto: str
) -> Optional[Dict[str, Any]]:
    """
    Evaluate procedures against ground truth.
    
    Args:
        integration_results: Integration results dict
        caption_file: Path to caption file with ground truth
        mapping_file_path: Path to object name mapping file
        output_dir: Output directory for evaluation results
        openai_client: OpenAI client instance
        proc_manual: Manual procedure key
        proc_auto: Auto procedure key
        
    Returns:
        Evaluation results dict or None
    """
    print(f"\nStep 6: Running Procedure Evaluation")
    print("=" * 50)
    
    # Load ground truth from caption file
    from io_utils import load_json
    
    ground_truth = None
    
    if caption_file:
        raw_json = load_json(caption_file)
        ground_truth = raw_json.get('reference_procedure') or raw_json.get('caption')
        if ground_truth:
            if 'reference_procedure' in raw_json:
                print(f"Ground truth loaded from 'reference_procedure' field: {caption_file}")
            elif 'caption' in raw_json:
                print(f"Ground truth loaded from 'caption' field: {caption_file}")
        else:
            print("Warning: No 'reference_procedure' or 'caption' found in caption file")
    
    if not ground_truth:
        print("Skipping evaluation: No ground truth available")
        return None
    
    # Initialize evaluator
    evaluator = ProcedureEvaluator(openai_client)
    
    # Run evaluation
    evaluation_results = evaluator.run_rubric_evaluation(
        gold_reference=ground_truth,
        manual_object_detection_procedure=integration_results.get(proc_manual, ''),
        auto_object_detection_procedure=integration_results.get(proc_auto, ''),
        object_name_mapping_file=mapping_file_path,
        output_dir=output_dir
    )
    return evaluation_results


def display_and_return_results(
    results: Any,
    integration_results: Dict[str, str],
    evaluation_results: Optional[Dict[str, Any]],
    experiment_dir: str,
    exp_logger: Any,
    integration_types: List[str]
) -> Dict[str, Any]:
    """
    Display results summary and return final results dict.
    
    Args:
        results: Procedure generation results
        integration_results: Integration results
        evaluation_results: Evaluation results (can be None)
        experiment_dir: Experiment directory path
        exp_logger: Experiment logger instance (can be None)
        integration_types: List of integration type keys
        
    Returns:
        dict: Final results dictionary
    """
    # Step 5: Display Results
    print(f"\nStep 5: Results Summary")
    print("=" * 50)
    
    total_steps = 0
    for i, procedure in enumerate(results):
        print(f"\nSegment {i}: {procedure.segment_id}")
        print(f"  Processing mode: {procedure.processing_mode}")
        print(f"  Number of frames: {len(procedure.frames)}")
        print(f"  Number of steps: {len(procedure.steps)}")
        
        if procedure.objects:
            print(f"  Objects available: {len(procedure.objects)}")
        
        print("  Steps:")
        for step in procedure.steps:
            print(f"    {step.step_id}: {step.description}")
        
        total_steps += len(procedure.steps)
    
    print(f"\nTotal steps generated: {total_steps}")
    
    # Display integration results
    if integration_results:
        print("\n" + "=" * 50)
        print("INTEGRATED PROCEDURES")
        print("=" * 50)
        
        for integration_type in integration_types:
            if integration_type in integration_results:
                integrated_procedure = integration_results[integration_type]
                if integrated_procedure:
                    print(f"\n{integration_type.replace('_', ' ').title()} Integration:")
                    print("-" * 40)
                    lines = integrated_procedure.split('\n')[:5]
                    for line in lines:
                        if line.strip():
                            print(f"  {line}")
                    if len(lines) >= 5:
                        print("  ...")
                    print(f"  (Full procedure saved to integrations/{integration_type}_integrated_procedure.txt)")
    
    # Final summary
    print("\nPipeline completed successfully!")
    print(f"Experiment directory: {experiment_dir}")
    if evaluation_results:
        print(f"Evaluation results saved to: {evaluation_results['evaluation_dir']}")
    if exp_logger:
        print(exp_logger.get_experiment_summary())
    
    return {
        "results": results,
        "experiment_dir": experiment_dir,
        "experiment_summary": exp_logger.get_experiment_summary() if exp_logger else "No experiment summary available",
        "evaluation_results": evaluation_results
    }

