"""
Procedure evaluation module for rubric-based absolute scoring with retry functionality.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.core.openai_client import OpenAIVisionClient
from src.utils.config_loader import config_loader


class ProcedureEvaluator:
    """
    Evaluates integrated procedures using rubric-based absolute scoring with GPT-5.
    Compares Manual Object Detection and Auto Object Detection methods using 100-point rubric system.
    
    This evaluator uses rubric-based scoring to provide absolute quality scores
    for experimental procedure reproducibility assessment with retry functionality.
    """
    
    def __init__(self, openai_client: OpenAIVisionClient, max_retries: int = 3):
        """
        Initialize procedure evaluator.
        
        Args:
            openai_client: OpenAI client for evaluation
            max_retries: Maximum number of retry attempts for failed evaluations
        """
        self.client = openai_client
        self.max_retries = max_retries
        
        # Load evaluation configuration (uses api config)
        self.config = config_loader.get_api_config()
        
        # Evaluation criteria (from config or fallback)
        self.criteria_labels = self.config.get("evaluation_criteria", [
            "Completeness",
            "Conciseness", 
            "Consistency",
            "Accuracy of Item Names",
            "Accuracy of Action Names"
        ])
        
        # Procedure mapping (two-method comparison)
        self.procedure_labels = ["Manual Object Detection", "Auto Object Detection"]
        
        # Load rubric evaluation prompts
        rubric_prompts_path = "config/prompts/rubric_evaluation_prompts.json"
        with open(rubric_prompts_path, 'r', encoding='utf-8') as f:
            self.rubric_prompts = json.load(f)
    
    def evaluate_procedure_with_rubric(self, 
                           gold_reference: str,
                           candidate_procedure: str,
                           procedure_name: str = "Candidate",
                           object_name_mapping_file: str = None) -> Dict[str, Any]:
        """
        Evaluate a single procedure using rubric-based scoring with retry mechanism.
        
        Args:
            gold_reference: Ground truth procedure
            candidate_procedure: Procedure to evaluate
            procedure_name: Name of the procedure being evaluated
            object_name_mapping_file: Path to object name mapping JSON file (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*50}")
        print(f"RUBRIC EVALUATION: {procedure_name}")
        print(f"{'='*50}")
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"  Evaluating {procedure_name} (attempt {attempt + 1}/{self.max_retries})...")
                
                # Prepare evaluation prompt
                eval_prompt = self.rubric_prompts.get("rubric_evaluation", {}).get("prompt", "")
                if not eval_prompt:
                    raise ValueError("Rubric evaluation prompt not found")
                
                # Load object name mapping if provided
                object_name_mapping = "No object name mapping available."
                if object_name_mapping_file and os.path.exists(object_name_mapping_file):
                    try:
                        with open(object_name_mapping_file, 'r', encoding='utf-8') as f:
                            mapping_data = json.load(f)
                        # Format mapping for prompt
                        mapping_lines = []
                        for auto_name, manual_name in mapping_data.items():
                            mapping_lines.append(f"  \"{auto_name}\" → \"{manual_name}\"")
                        object_name_mapping = "\n".join(mapping_lines) if mapping_lines else "No mappings found."
                    except Exception as e:
                        print(f"  WARNING: Failed to load object name mapping: {e}")
                        object_name_mapping = "Failed to load object name mapping."
                
                # Format the prompt with actual procedures
                formatted_prompt = eval_prompt.format(
                    gold_reference=gold_reference,
                    candidate_procedure=candidate_procedure,
                    object_name_mapping=object_name_mapping
                )
                
                # JSON format is already specified in the rubric prompt
                
                # Get evaluation from GPT-5
                response = self.client.analyze_text_only(
                    prompt=formatted_prompt
                )
                
                # Parse JSON response - no fallback processing allowed
                # Clean the response by extracting JSON content
                cleaned_response = response.strip()
                
                # Find JSON boundaries
                start_idx = cleaned_response.find('{')
                end_idx = cleaned_response.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_content = cleaned_response[start_idx:end_idx + 1]
                    result = json.loads(json_content)
                else:
                    # Try parsing the entire response
                    result = json.loads(cleaned_response)
                        
                
                # Validate required fields
                required_fields = ["initial_score", "deductions", "final_score", "summary"]
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")
                
                # Extract and validate data
                initial_score = result.get("initial_score", 100)
                deductions = result.get("deductions", [])
                final_score = result.get("final_score", 0)
                summary = result.get("summary", "")
                
                # Calculate score from deductions for verification
                calculated_score = initial_score - sum(d.get("points_deducted", 0) for d in deductions)
                calculated_score = max(0, calculated_score)  # Ensure non-negative
                
                # Parse category summary
                parsed_category_summary = result.get("category_summary", {})
                
                print(f"  ✓ {procedure_name} evaluation successful on attempt {attempt + 1}")
                print(f"  Score: {final_score}/100 (calculated: {calculated_score})")
                
                # Print deduction details
                if deductions:
                    print(f"  Deduction Details:")
                    for deduction in deductions:
                        category = deduction.get("category", "Unknown")
                        description = deduction.get("description", "No description")
                        points = deduction.get("points_deducted", 0)
                        print(f"    - {category}: -{points} points - {description}")
                
                return {
                    "procedure_name": procedure_name,
                    "initial_score": initial_score,
                    "deductions": deductions,
                    "final_score": final_score,
                    "calculated_score": calculated_score,
                    "summary": summary,
                    "category_summary": parsed_category_summary,
                    "raw_response": response,
                    "evaluation_prompt": formatted_prompt,
                    "attempts": attempt + 1
                }
                
            except Exception as e:
                print(f"  ⚠ Evaluation error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    print(f"  Retrying evaluation in 2 seconds...")
                    time.sleep(2)  # Brief delay before retry
                    continue
        
        # All retries failed - terminate execution
        error_msg = f"CRITICAL ERROR: {procedure_name} evaluation failed after {self.max_retries} attempts: {last_error}"
        print(f"  ✗ {error_msg}")
        print(f"  Terminating execution due to evaluation failure.")
        raise RuntimeError(error_msg)
    
    def run_rubric_evaluation(self,
                                gold_reference: str,
                            manual_object_detection_procedure: str,
                            auto_object_detection_procedure: str,
                            object_name_mapping_file: str = None,
                                output_dir: str = None) -> Dict[str, Any]:
        """
        Run rubric-based evaluation for both procedures with retry functionality.
        
        Args:
            gold_reference: Ground truth procedure
            manual_object_detection_procedure: Manual object detection integrated procedure
            auto_object_detection_procedure: Auto object detection integrated procedure
            object_name_mapping_file: Path to object name mapping JSON file (optional)
            output_dir: Output directory for results
            
        Returns:
            Dictionary with evaluation results and file paths
        """
        print("\n" + "="*60)
        print("STARTING RUBRIC-BASED PROCEDURE EVALUATION")
        print("="*60)
        
        # Create evaluation subdirectory
        eval_dir = os.path.join(output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Evaluate both procedures with retry - fail fast on error (auto first, then manual)
        evaluations = {}
        
        auto_evaluation = self.evaluate_procedure_with_rubric(
            gold_reference=gold_reference, 
            candidate_procedure=auto_object_detection_procedure, 
            procedure_name="Auto Object Detection", 
            object_name_mapping_file=object_name_mapping_file, 
        )
        evaluations["auto_object_detection"] = auto_evaluation
        
        manual_evaluation = self.evaluate_procedure_with_rubric(
            gold_reference=gold_reference, 
            candidate_procedure=manual_object_detection_procedure, 
            procedure_name="Manual Object Detection", 
            object_name_mapping_file=object_name_mapping_file, 
        )
        evaluations["manual_object_detection"] = manual_evaluation
        
        # Create comprehensive results
        results = {
            "evaluations": evaluations,
            "comparison": self._create_comparison_summary_multi(evaluations),
            "timestamp": datetime.now().isoformat(),
            "evaluation_settings": {
                "max_retries": self.max_retries
            }
        }
        
        # Save detailed results
        results_json_path = os.path.join(eval_dir, "rubric_evaluation_results.json")
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        summary_path = os.path.join(eval_dir, "evaluation_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(results["comparison"]["summary_report"])
        
        print(f"\n✓ Evaluation completed successfully")
        print(f"  Results saved to: {results_json_path}")
        print(f"  Summary saved to: {summary_path}")
        
        return {
            "results": results,
            "results_json": results_json_path,
            "summary_file": summary_path,
            "evaluation_dir": eval_dir
        }
    
    def _create_comparison_summary_multi(self, evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create comparison summary for multiple evaluations (2 or 3 methods)."""
        
        # Extract scores
        scores = {}
        attempts = {}
        summaries = {}
        
        for method_key, eval_data in evaluations.items():
            method_name = eval_data.get("procedure_name", method_key.replace("_", " ").title())
            scores[method_name] = eval_data.get("final_score", 0)
            attempts[method_name] = eval_data.get("attempts", "Unknown")
            summaries[method_name] = eval_data.get("summary", "No summary available")
        
        # Find winner (highest score)
        if scores:
            winner_name = max(scores, key=scores.get)
            winner_score = scores[winner_name]
            
            # Check for ties
            tied_methods = [name for name, score in scores.items() if abs(score - winner_score) < 1.0]
            if len(tied_methods) > 1:
                winner = "Tie"
                winner_text = f"Result: Tie between {', '.join(tied_methods)} (scores within 1 point)"
            else:
                winner = winner_name
                winner_text = f"Winner: {winner_name} ({winner_score}/100)"
        else:
            winner = "Unknown"
            winner_text = "No valid scores found"
        
        # Create detailed summary report
        summary_report = f"""RUBRIC EVALUATION COMPARISON REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL SCORES:
"""
        
        for method_name, score in scores.items():
            summary_report += f"  {method_name}: {score}/100\n"
        
        summary_report += f"\n{winner_text}\n\nEVALUATION ATTEMPTS:\n"
        
        for method_name, attempt_count in attempts.items():
            summary_report += f"  {method_name}: {attempt_count} attempts\n"
        
        summary_report += "\nDETAILED ANALYSIS:\n\n"
        
        for method_name, summary in summaries.items():
            summary_report += f"{method_name}:\n{summary}\n\n"
        
        return {
            "winner": winner,
            "scores": scores,
            "summary_report": summary_report,
            **{f"{method.lower().replace(' ', '_')}_score": score for method, score in scores.items()},
            **{f"{method.lower().replace(' ', '_')}_attempts": attempts[method] for method in scores.keys()}
        }
