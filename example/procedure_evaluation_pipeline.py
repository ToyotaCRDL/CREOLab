#!/usr/bin/env python3
"""
Basic procedure generation pipeline example.
Demonstrates the complete workflow from video to procedure generation.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.captioning.object_knowledge_loader import ObjectKnowledgeLoader
from src.core.openai_client import OpenAIVisionClient
from src.utils.experiment_logger import ExperimentLogger
from src.core.base_models import CaptionData

# Local helper modules
from chart_generators import (
    create_iteration_bar_chart, compute_take_aggregate_results,
    create_cross_analysis_charts, create_batch_total_bar_chart,
    PROC_MANUAL, PROC_AUTO, PROCEDURE_MAPPING, PROCEDURE_ORDER, PROCEDURE_LABELS,
    INTEGRATION_TYPES, RUBRIC_EVALUATION_RESULTS_FILE
)
from io_utils import save_text
from pipeline_steps import (
    prepare_references,
    segment_and_extract_frames,
    generate_procedures,
    integrate_procedures,
    evaluate_procedures,
    display_and_return_results
)


# ============================================================================
# Helper Functions
# ============================================================================

def load_caption_bundle(caption_file):
    """
    Load caption file once and extract all needed information.
    
    Args:
        caption_file: Path to caption JSON file
        
    Returns:
        tuple: (CaptionData object, ground_truth text, raw JSON dict)
    """
    with open(caption_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    # Create CaptionData object
    caption_data = CaptionData.from_dict(raw)
    
    # Extract ground truth procedure text
    ground_truth = None
    if 'reference_procedure' in raw:
        rp = raw['reference_procedure']
        if isinstance(rp, list):
            ground_truth = '\n'.join(f"{i+1}. {step}" for i, step in enumerate(rp))
        else:
            ground_truth = rp
    elif 'caption' in raw:
        ground_truth = raw['caption']
    
    return caption_data, ground_truth, raw


def get_output_directory(exp_logger, base_dir, subdir_name):
    """
    Get output directory path with consistent logic.
    
    Args:
        exp_logger: ExperimentLogger instance (can be None)
        base_dir: Base experiment directory (used only if exp_logger is None)
        subdir_name: Subdirectory name (e.g., "segments", "frames", "prompts")
    
    Returns:
        str: Directory path (created if needed)
    """
    if exp_logger:
        # Use ExperimentLogger's unified method
        return exp_logger.get_subdir(subdir_name)
    else:
        # No exp_logger (create directory manually)
        dir_path = os.path.join(base_dir, subdir_name)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path


# ============================================================================
# Main Processing Functions
# ============================================================================

def run_basic_pipeline(video_path: str, caption_file: str = None, output_base_dir: str = "output", max_duration: float = None, enable_evaluation: bool = True):
    """
    Run the basic procedure generation pipeline.
    
    Args:
        video_path: Path to input video file
        caption_file: Path to caption file (for experiment logging)
        output_base_dir: Base directory for outputs (if it's an existing directory path, use it directly)
        max_duration: Maximum duration to analyze from the video (seconds, None = entire video)
        enable_evaluation: Whether to run procedure evaluation against ground truth (default: True)
    """
    # ============================================================================
    # 1. Initialization and Directory Setup
    # ============================================================================
    
    # Check if output_base_dir is an existing specific directory (like iter_01)
    if os.path.exists(output_base_dir) and os.path.basename(output_base_dir).startswith(('iter_', 'iteration_')):
        # Use the provided directory directly (for iteration runs)
        experiment_dir = output_base_dir
        exp_logger = None  # Don't use logger for iteration runs
    else:
        # Initialize experiment logger for standalone runs
        exp_logger = ExperimentLogger(output_base_dir)
        experiment_dir = exp_logger.create_experiment_directory()
    
    print(f"Starting basic procedure generation pipeline")
    print(f"Video: {video_path}")
    print(f"Max duration: {'Entire video' if max_duration is None else f'{max_duration}s'}")
    print(f"Experiment directory: {experiment_dir}")
    print("-" * 50)
    
    # Log experiment conditions (only for standalone runs)
    if exp_logger:
        exp_logger.log_experiment_conditions(
            video_path=video_path,
            max_duration=max_duration,
            caption_file=caption_file
        )
    
    # Get output subdirectories
    segments_dir = get_output_directory(exp_logger, experiment_dir, "segments")
    frames_dir = get_output_directory(exp_logger, experiment_dir, "frames")
    ref_images_dir = get_output_directory(exp_logger, experiment_dir, "reference_images")
    prompts_dir = get_output_directory(exp_logger, experiment_dir, "prompts")
    integration_dir = get_output_directory(exp_logger, experiment_dir, "integrations")
    
    # Get video name for IDs
    video_name = Path(video_path).stem
    
    # Initialize OpenAI client and knowledge loader
    openai_client = OpenAIVisionClient()
    knowledge_loader = ObjectKnowledgeLoader()
    
    # ============================================================================
    # 2. Prepare Reference Images
    # ============================================================================
    
    reference_image_with_overlay, reference_image_without_overlay, caption_data, random_numbers = prepare_references(
        video_path=video_path,
        video_name=video_name,
        caption_file=caption_file,
        ref_images_dir=ref_images_dir,
        knowledge_loader=knowledge_loader
    )
    
    # ============================================================================
    # 3. Segment Video and Extract Frames
    # ============================================================================
    
    all_segment_data = segment_and_extract_frames(
        video_path=video_path,
        video_name=video_name,
        segments_dir=segments_dir,
        frames_dir=frames_dir,
        max_duration=max_duration
    )
    
    # ============================================================================
    # 4. Generate Procedures (Auto and Manual)
    # ============================================================================
    
    manual_results, auto_results, mapping_file_path = generate_procedures(
        video_path=video_path,
        video_name=video_name,
        all_segment_data=all_segment_data,
        ref_images_dir=ref_images_dir,
        reference_image_with_overlay=reference_image_with_overlay,
        prompts_dir=prompts_dir,
        openai_client=openai_client,
        knowledge_loader=knowledge_loader
    )
    
    # ============================================================================
    # 5. Integrate Procedures
    # ============================================================================
    
    integration_results = integrate_procedures(
        manual_results=manual_results,
        auto_results=auto_results,
        experiment_dir=experiment_dir,
        integration_dir=integration_dir,
        prompts_dir=prompts_dir,
        openai_client=openai_client,
        integration_types=INTEGRATION_TYPES
    )
    
    # ============================================================================
    # 6. Evaluate Procedures (if enabled)
    # ============================================================================
    
    evaluation_results = None
    if enable_evaluation and integration_results:
        # Use take_dir if available, otherwise experiment_dir
        eval_output_dir = exp_logger.take_dir if exp_logger and exp_logger.take_dir else experiment_dir
        
        evaluation_results = evaluate_procedures(
            integration_results=integration_results,
            caption_file=caption_file,
            mapping_file_path=mapping_file_path,
            output_dir=eval_output_dir,
            openai_client=openai_client,
            proc_manual=PROC_MANUAL,
            proc_auto=PROC_AUTO
        )
    
    # ============================================================================
    # 7. Display Results and Log
    # ============================================================================
    
    # Determine which results to return for display
    results = manual_results if manual_results else auto_results
    
    # Log processing results (only for standalone runs)
    if exp_logger and manual_results:
        total_frames = sum(len(proc.frames) for proc in manual_results)
        exp_logger.log_processing_results(
            segments_created=len(all_segment_data),
            frames_extracted=total_frames,
            manual_object_detection_results=len(manual_results) if manual_results else None,
            manual_detection_results=len(manual_results) if manual_results else None,
            integrations_created=len(integration_results) if integration_results else None
        )
    
    # Display and return final results
    return display_and_return_results(
        results=results,
        integration_results=integration_results,
        evaluation_results=evaluation_results,
        experiment_dir=experiment_dir,
        exp_logger=exp_logger,
        integration_types=INTEGRATION_TYPES
    )


def find_latest_batch_directory(output_base_dir: str, split_name: str) -> str:
    """
    Find the latest batch experiment directory for the given split.
    
    Args:
        output_base_dir: Base output directory (e.g., "output")
        split_name: Dataset split name ('dev', 'test', etc.)
    
    Returns:
        Path to the latest split directory (e.g., "output/batch_XXXXX/debug"), or None if not found
    """
    if not os.path.exists(output_base_dir):
        return None
    
    batch_dirs = []
    for item in os.listdir(output_base_dir):
        if item.startswith("batch_") and os.path.isdir(os.path.join(output_base_dir, item)):
            # Check if this batch directory contains the split subdirectory
            batch_path = os.path.join(output_base_dir, item)
            split_path = os.path.join(batch_path, split_name)
            if os.path.exists(split_path) and os.path.isdir(split_path):
                batch_dirs.append((item, split_path))
    
    if not batch_dirs:
        return None
    
    # Sort by directory name (which includes timestamp) and return the latest
    batch_dirs.sort(key=lambda x: x[0], reverse=True)
    return batch_dirs[0][1]


def run_batch_processing(split_name: str, num_iterations: int = 5, max_duration: float = None, resume_mode: bool = False):
    """
    Run batch processing mode: process all takes in specified dataset split.
    
    Args:
        split_name: Dataset split name ('dev' or 'test')
        num_iterations: Number of iterations per take (default: 5)
        max_duration: Maximum duration to analyze from each video (seconds, None = entire video)
        resume_mode: If True, resume from the latest batch experiment directory
    """
    print(f"\n{'='*80}")
    if resume_mode:
        print(f"BATCH PROCESSING MODE - {split_name.upper()} SPLIT (RESUME)")
    else:
        print(f"BATCH PROCESSING MODE - {split_name.upper()} SPLIT")
    print(f"{'='*80}")
    
    # Load dataset splits
    splits_file = "data/dataset_splits.json"
    if not os.path.exists(splits_file):
        print(f"Error: Dataset splits file not found: {splits_file}")
        return
    
    try:
        with open(splits_file, 'r', encoding='utf-8') as f:
            splits_data = json.load(f)
    except Exception as e:
        print(f"Error loading dataset splits: {e}")
        return
    
    if split_name not in splits_data:
        print(f"Error: Split '{split_name}' not found in dataset splits")
        print(f"Available splits: {list(splits_data.keys())}")
        return
    
    take_files = splits_data[split_name]
    if not take_files:
        print(f"Warning: No takes found in '{split_name}' split")
        return
    
    print(f"Configuration:")
    print(f"  - Dataset split: {split_name}")
    print(f"  - Takes to process: {len(take_files)}")
    print(f"  - Iterations per take: {num_iterations}")
    print(f"  - Evaluations per take: {num_iterations}")
    print(f"  - Total evaluations: {len(take_files)} × {num_iterations} = {len(take_files) * num_iterations}")
    if max_duration:
        print(f"  - Duration per video: {max_duration}s")
    else:
        print(f"  - Duration per video: Entire video")
    
    # Handle resume mode or create new batch experiment
    if resume_mode:
        # Find the latest batch experiment directory
        batch_experiment_dir = find_latest_batch_directory("output", split_name)
        if not batch_experiment_dir:
            print(f"Error: No existing batch experiment directory found for split '{split_name}'")
            print("Available directories:")
            output_dir = "output"
            if os.path.exists(output_dir):
                for item in sorted(os.listdir(output_dir)):
                    if item.startswith("batch_") and os.path.isdir(os.path.join(output_dir, item)):
                        print(f"  {item}")
            return
        print(f"Resuming batch experiment: {batch_experiment_dir}")
    else:
        # Create new batch experiment logger
        batch_logger = ExperimentLogger("output", batch_mode=True, split_name=split_name)
        batch_experiment_dir = batch_logger.create_experiment_directory()
        print(f"Batch experiment directory: {batch_experiment_dir}")
    
    # Process each take
    successful_takes = 0
    failed_takes = []
    
    for i, take_file in enumerate(take_files, 1):
        take_name = take_file.replace('.json', '')
        
        # Check iteration completion in resume mode
        skip_take = False
        completed_iterations = 0
        if resume_mode:
            take_dir_path = os.path.join(batch_experiment_dir, take_name)
            if os.path.exists(take_dir_path) and os.path.isdir(take_dir_path):
                # Check how many iterations are completed
                for iter_num in range(1, num_iterations + 1):
                    iter_dir = os.path.join(take_dir_path, f"iter_{iter_num:02d}")
                    evaluation_file = os.path.join(iter_dir, "evaluation", RUBRIC_EVALUATION_RESULTS_FILE)
                    if os.path.exists(evaluation_file):
                        completed_iterations += 1
                
                if completed_iterations == num_iterations:
                    print(f"\n{'='*60}")
                    print(f"PROCESSING TAKE {i}/{len(take_files)}: {take_name} (RESUME - AGGREGATE ONLY)")
                    print(f"  All {num_iterations} iterations completed")
                    print(f"  Regenerating aggregate results...")
                    print(f"{'='*60}")
                    # Don't skip - process aggregate generation
                    skip_take = False
                else:
                    print(f"\n{'='*60}")
                    print(f"RESUMING TAKE {i}/{len(take_files)}: {take_name}")
                    print(f"  Completed iterations: {completed_iterations}/{num_iterations}")
                    print(f"  Will continue from iteration {completed_iterations + 1}")
                    print(f"{'='*60}")
        
        if skip_take:
            continue
        
        print(f"\n{'='*60}")
        print(f"PROCESSING TAKE {i}/{len(take_files)}: {take_name}")
        print(f"{'='*60}")
        
        # Setup paths
        caption_file_path = os.path.join("data/captions", take_file)
        if not os.path.exists(caption_file_path):
            print(f"Error: Caption file not found: {caption_file_path}")
            failed_takes.append(take_name)
            continue
        
        # Load caption data to get video filename
        caption_data, _, _ = load_caption_bundle(caption_file_path)
        
        video_filename = caption_data.video_id
        if not video_filename:
            print(f"Error: No video_id found in {caption_file_path}")
            failed_takes.append(take_name)
            continue
        
        video_path = f"data/videos/{video_filename}"
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            failed_takes.append(take_name)
            continue
        
        print(f"Processing: {take_name}")
        print(f"  Caption: {caption_file_path}")
        print(f"  Video: {video_path}")
        
        # Create take directory
        if resume_mode:
            # Use existing batch directory structure
            take_dir = os.path.join(batch_experiment_dir, take_name)
            os.makedirs(take_dir, exist_ok=True)
        else:
            take_dir = batch_logger.create_experiment_directory(take_name)
        print(f"  Output: {take_dir}")
        
        # Run full evaluation for this take
        # Run take evaluation with multiple iterations (supports resume)
        # Pass skip_take flag to indicate if only aggregate processing is needed
        aggregate_only = (resume_mode and completed_iterations == num_iterations)
        run_take_evaluation(video_path, caption_file_path, take_dir, num_iterations, max_duration, resume_mode, aggregate_only)
        
        successful_takes += 1
        print(f"✓ Take {take_name} completed successfully")
    
    # Create batch total bar chart and cross-analysis charts
    if successful_takes > 0:
        print(f"\n{'='*60}")
        print(f"CREATING BATCH BAR CHARTS")
        print(f"{'='*60}")
        
        if resume_mode:
            print(f"  (Regenerating all analysis charts in resume mode)")
        
        create_batch_total_bar_chart(batch_experiment_dir)
        
        print(f"\n{'='*60}")
        print(f"CREATING CROSS-ANALYSIS CHARTS")
        print(f"{'='*60}")
        create_cross_analysis_charts(batch_experiment_dir, split_name)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total takes: {len(take_files)}")
    print(f"Successful: {successful_takes}")
    print(f"Failed: {len(failed_takes)}")
    if failed_takes:
        print(f"Failed takes: {', '.join(failed_takes)}")
    
    # batch_experiment_dir is output/batch_XXXXXXXX/test, get parent for display
    batch_root_dir = os.path.dirname(batch_experiment_dir)
    print(f"\nBatch results saved to: {batch_root_dir}")
    
    if successful_takes > 0:
        print(f"\nBar charts created:")
        print(f"  - Individual iteration charts: Each iter_XX/ directory")
        print(f"  - Take aggregate charts: Each take/aggregate/ directory")
        print(f"  - Cross-analysis charts: {batch_root_dir}/cross_analysis/")
        print(f"    * By scenario number: cross_analysis/by_scenario_number/")
        print(f"    * By decoy number: cross_analysis/by_decoy_number/")
        if split_name != "debug":
            print(f"  - Quality progression charts: {batch_root_dir}/quality_progression/")
            print(f"    * Method-wise decoy progression: quality_progression/scenarioXX/")
        else:
            print(f"  - Quality progression: Skipped for debug split (incomplete decoy variations)")


def run_take_evaluation(video_path: str, caption_file: str, take_dir: str, num_iterations: int = 5, max_duration: float = None, resume_mode: bool = False, aggregate_only: bool = False) -> List[Dict]:
    """
    Run evaluation for a single take with multiple iterations.
    
    Args:
        video_path: Path to input video file
        caption_file: Path to caption file
        take_dir: Directory to save take results
        num_iterations: Number of iterations to run
        max_duration: Maximum duration to analyze from the video (seconds, None = entire video)
        resume_mode: If True, skip completed iterations
        aggregate_only: If True, only regenerate aggregate results (skip all iterations)
    """
    all_results = []
    
    # If aggregate_only is True, skip iteration processing and load existing results
    if aggregate_only:
        print(f"  Loading existing results for aggregate processing...")
        for iteration in range(num_iterations):
            iter_num = iteration + 1
            iter_dir = os.path.join(take_dir, f"iter_{iter_num:02d}")
            evaluation_file = os.path.join(iter_dir, "evaluation", RUBRIC_EVALUATION_RESULTS_FILE)
            
            if os.path.exists(evaluation_file):
                # Create a results structure for aggregate processing
                results = {
                    "iteration": iter_num,
                    "evaluation_results": {
                        "evaluation_dir": os.path.join(iter_dir, "evaluation")
                    },
                    "integration_results": {
                        PROC_MANUAL: "",
                        PROC_AUTO: ""
                    }
                }
                all_results.append(results)
        
        print(f"  Loaded {len(all_results)} existing iterations for aggregate processing")
    else:
        # Normal iteration processing
        for iteration in range(num_iterations):
            iter_num = iteration + 1
            iter_dir = os.path.join(take_dir, f"iter_{iter_num:02d}")
            
            # Check if iteration is already completed in resume mode
            if resume_mode:
                evaluation_file = os.path.join(iter_dir, "evaluation", RUBRIC_EVALUATION_RESULTS_FILE)
                if os.path.exists(evaluation_file):
                    print(f"\n  Iteration {iter_num}/{num_iterations} - SKIPPING (already completed)")
                    # Load existing results for aggregate processing
                    try:
                        # Create a complete results structure for aggregate processing
                        results = {
                            "iteration": iter_num,
                            "evaluation_results": {
                                "evaluation_dir": os.path.join(iter_dir, "evaluation")
                            },
                            "integration_results": {
                                # These will be loaded during aggregate processing if needed
                                PROC_MANUAL: "",
                                PROC_AUTO: ""
                            }
                        }
                        all_results.append(results)
                        print(f"    Loaded existing results for aggregate processing")
                    except Exception as e:
                        print(f"    Warning: Could not load existing results: {e}")
                    continue
            
            print(f"\n  Iteration {iter_num}/{num_iterations}")
            
            # Create iteration directory
            os.makedirs(iter_dir, exist_ok=True)
            
            # Run pipeline for this iteration
            results = run_basic_pipeline(
                video_path, caption_file, iter_dir, 
                max_duration=max_duration, enable_evaluation=True
            )
            
            if results:
                # Add iteration number to results for consistency
                results["iteration"] = iter_num
                all_results.append(results)
                
                print(f"    ✓ Iteration {iter_num} completed")
            else:
                print(f"    ✗ Iteration {iter_num} failed")

    
    # Generate aggregate results (always regenerate in resume mode)
    # This ensures that aggregate charts are refreshed even when iterations are skipped
    if all_results:
        print(f"\n  Computing aggregate results from {len(all_results)} iterations...")
        if resume_mode:
            print(f"    (Regenerating aggregate results and charts in resume mode)")
        take_name = os.path.splitext(os.path.basename(video_path))[0]
        compute_take_aggregate_results(take_dir, take_name, exp_logger=None)
        create_iteration_bar_chart(take_dir, take_name)
        print(f"  ✓ Aggregate results saved to: {os.path.join(take_dir, 'aggregate')}")
    else:
        print(f"  ✗ No successful iterations to aggregate")
    
    return all_results


def run_full_evaluation(video_path: str, caption_file: str, max_duration: float = None, num_iterations: int = 5):
    """
    Run full evaluation mode: multiple procedure generation seeds with rubric evaluation.
    
    Args:
        video_path: Path to input video file
        caption_file: Path to caption file
        max_duration: Maximum duration to analyze from the video (seconds, None = entire video)
        num_iterations: Number of iterations to run (default: 5)
    """
    print(f"\n{'='*80}")
    print(f"FULL EVALUATION MODE ({num_iterations} iterations)")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Procedure generation iterations: {num_iterations}")
    print(f"  - Evaluation method: Rubric evaluation")
    print(f"  - Total evaluations: {num_iterations}")
    print(f"  - Video: {video_path}")
    print(f"  - Caption: {caption_file}")
    if max_duration:
        print(f"  - Duration: {max_duration}s")
    else:
        print(f"  - Duration: Entire video")
 
    
    # Create main experiment directory using new structure
    # Extract take name from caption file
    take_name = os.path.basename(caption_file).replace('.json', '')
    
    exp_logger = ExperimentLogger("output", batch_mode=False)
    take_dir = exp_logger.create_experiment_directory(take_name)
    
    print(f"Take directory: {take_dir}")
    
    # Use the new take evaluation function (aggregation is done internally)
    all_evaluation_results = run_take_evaluation(video_path, caption_file, take_dir, num_iterations, max_duration)
    
    if all_evaluation_results:
        # Save summary
        summary_file = os.path.join(take_dir, "full_evaluation_summary.txt")
        summary_content = (
            f"Total iterations: {len(all_evaluation_results)}\n"
            f"Total evaluations: {len(all_evaluation_results)}\n"
            f"Seed functionality: Disabled for GPT-5 compatibility\n\n"
        )
            
        for result in all_evaluation_results:
            summary_content += (
                f"Iteration {result['iteration']}:\n"
                f"  Experiment dir: {result['experiment_dir']}\n"
                f"  Evaluation dir: {result['evaluation_results'].get('evaluation_dir', 'N/A')}\n\n"
            )
        
        header = f"FULL EVALUATION SUMMARY\n{'=' * 50}"
        save_text(summary_file, summary_content, header=header)
        
        print(f"✓ Full evaluation completed successfully!")
        print(f"  - Total iterations: {len(all_evaluation_results)}")
        print(f"  - Total evaluations: {len(all_evaluation_results)}")
        print(f"  - Results saved to: {take_dir}")
        print(f"  - Summary: {summary_file}")
        print(f"  - Note: All directories preserved for analysis (cleanup disabled)")
        
    else:
        print("✗ Full evaluation failed: No successful iterations")


def main():
    """Main function for running the example."""
    
    parser = argparse.ArgumentParser(
        description="Procedure evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file processing
  python %(prog)s --caption-file data/captions/scenario05_decoy0.json --max-duration 30.0

  # Batch processing
  python %(prog)s --batch debug --iterations 3 --max-duration 30.0
  python %(prog)s --batch dev --iterations 5 --resume
  python %(prog)s --batch test --iterations 10

Batch splits:
  debug: Quick test with 2 takes
  dev:   Full development set
  test:  Full test set
        """
    )
    
    # Main operation modes
    parser.add_argument('--batch', choices=['debug', 'dev', 'test'],
                       help='Run batch processing on specified dataset split')
    
    # File and processing options
    parser.add_argument('--caption-file', type=str,
                       default='data/captions/scenario05_decoy0.json',
                       help='Path to caption file for single file processing (default: %(default)s)')
    parser.add_argument('--max-duration', type=float,
                       help='Maximum duration in seconds to process (default: entire video)')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations for processing (default: %(default)s)')
    
    # Control options
    parser.add_argument('--resume', action='store_true',
                       help='Resume from the latest batch experiment directory')
    
    args = parser.parse_args()
    
    # Handle batch processing mode
    if args.batch:
        print(f"Running batch processing on '{args.batch}' split")
        print(f"Iterations: {args.iterations}")
        if args.max_duration:
            print(f"Max duration: {args.max_duration}s")
        if args.resume:
            print("Resume mode enabled")
        
        run_batch_processing(args.batch, args.iterations, args.max_duration, args.resume)
        return
    
    # Single file processing mode
    caption_file = args.caption_file
    max_duration = args.max_duration
    
    print(f"Processing single file: {caption_file}")
    if max_duration:
        print(f"Max duration: {max_duration}s")
    
    # Check if caption file exists
    if not os.path.exists(caption_file):
        print(f"Error: Caption file not found: {caption_file}")
        print("Please ensure the caption file exists or provide a different path.")
        print("\nUse --help for usage information.")
        return
    
    # Load caption data to get video filename
    caption_data, _, _ = load_caption_bundle(caption_file)
        
    video_filename = caption_data.video_id
    if not video_filename:
        print(f"Error: No video_id found in {caption_file}")
        return
    
    # Construct video path
    video_path = f"data/videos/{video_filename}"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print(f"Please ensure the video file '{video_filename}' exists in data/videos/")
        return
    
    print(f"Using caption file: {caption_file}")
    print(f"Using video file: {video_path}")
    print("-" * 50)
    
    # Display processing scope
    if max_duration is None:
        print(f"Processing scope: Entire video")
    else:
        print(f"Processing scope: First {max_duration} seconds")
    
    # Run in iteration mode
    print(f"Running {args.iterations} iteration{'s' if args.iterations > 1 else ''} with different procedure generation seeds...")
    run_full_evaluation(video_path, caption_file, max_duration, args.iterations)


if __name__ == "__main__":
    main()


