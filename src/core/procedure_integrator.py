"""
Procedure integration module for combining multiple segment procedures into complete workflows.
"""

import time
import os
import json
from typing import List, Dict, Any, Tuple
from .base_models import SegmentProcedure, ProcedureStep, create_step_id
from .openai_client import OpenAIVisionClient

from src.utils.config_loader import config_loader
from src.utils.prompt_loader import prompt_loader


class ProcedureIntegrator:
    """
    Integrates multiple segment procedures into complete experimental workflows.
    
    Supports two integration methods:
    1. Manual Object Detection (uses predefined object knowledge from caption files)
    2. Auto Object Detection (uses automatically detected objects from reference images)
    
    Both methods use the common_integration prompt for consistency.
    """
    
    def __init__(self, openai_client: OpenAIVisionClient, max_retries: int = 3):
        """Initialize the procedure integrator."""
        self.client = openai_client
        self.max_retries = max_retries
    
    def _is_valid_integration_result(self, result: str) -> Tuple[bool, str]:
        """
        Check if the integration result is valid.
        
        Args:
            result: Integration result text
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        if not result:
            return False, "Result is None"
        
        # Allow shorter results if they contain the "no verifiable actions" message
        if len(result.strip()) < 10 and "no verifiable actions found" not in result.lower():
            return False, f"Result too short: {len(result.strip())} characters (minimum 10 required)"
        
        # Check if it contains error messages (but allow "No verifiable actions" as valid)
        if "no verifiable actions found" in result.lower():
            # This is a valid result from segment verification indicating no actions could be verified
            return True, "Valid: No verifiable actions found (legitimate verification result)"
        
        error_indicators = [
            "Integration failed:",
            "Error:",
            "No procedures to integrate",
            "failed to",
            "could not",
            "unable to"
        ]
        
        result_lower = result.lower()
        for indicator in error_indicators:
            if indicator.lower() in result_lower:
                return False, f"Contains error indicator: '{indicator}'"
        
        # Check if it contains numbered steps (basic format validation)
        lines = result.strip().split('\n')
        numbered_steps = 0
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('Step')):
                numbered_steps += 1
        
        # Should have at least 1 numbered step
        if numbered_steps < 1:
            return False, f"No numbered steps found (found {numbered_steps}, minimum 1 required)"
        
        return True, f"Valid: {numbered_steps} numbered steps found, {len(result)} characters total"
    
    def _save_warning_log(self, integration_type: str, attempt: int, error: str, output_dir: str = None, 
                         raw_response: str = None, prompt_used: str = None, validation_details: str = None):
        """
        Save detailed warning log for failed integration attempts.
        
        Args:
            integration_type: Type of integration (manual_object_detection, auto_object_detection)
            attempt: Attempt number
            error: Error message
            output_dir: Output directory for logs
            raw_response: Raw GPT response (if any)
            prompt_used: Prompt that was sent to GPT
            validation_details: Details about why validation failed
        """
        if not output_dir:
            return
        
        log_dir = os.path.join(output_dir, "integration_warnings")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{integration_type}_attempt_{attempt}_{timestamp}.log")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Integration Type: {integration_type}\n")
            f.write(f"Attempt: {attempt}/{self.max_retries}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Error: {error}\n")
            f.write(f"\n" + "="*50 + "\n")
            
            # Response analysis
            if raw_response is not None:
                f.write(f"RAW RESPONSE RECEIVED: YES\n")
                f.write(f"Response Length: {len(raw_response)} characters\n")
                f.write(f"Response Preview (first 200 chars): {raw_response[:200]}\n")
                f.write(f"Response Preview (last 200 chars): {raw_response[-200:]}\n")
                f.write(f"Response is empty/whitespace only: {not raw_response.strip()}\n")
                
                if validation_details:
                    f.write(f"\nValidation Failure Details: {validation_details}\n")
                
                f.write(f"\n" + "-"*30 + " FULL RAW RESPONSE " + "-"*30 + "\n")
                f.write(raw_response)
                f.write(f"\n" + "-"*75 + "\n")
            else:
                f.write(f"RAW RESPONSE RECEIVED: NO (API call failed)\n")
            
            # Prompt analysis
            if prompt_used:
                f.write(f"\n" + "-"*30 + " PROMPT USED " + "-"*30 + "\n")
                f.write(f"Prompt Length: {len(prompt_used)} characters\n")
                f.write(f"Prompt Preview: {prompt_used[:300]}...\n")
                f.write(f"\n" + "-"*75 + "\n")
        
        print(f"  Warning: Integration attempt {attempt} failed. Detailed log saved: {log_file}")
    
    def integrate_procedures(self, procedures: List[SegmentProcedure], output_dir: str = None) -> tuple[str, str, str]:
        """
        Integrate segment procedures into a complete workflow.
        
        Args:
            procedures: List of segment procedures to integrate
            output_dir: Output directory for warning logs
            
        Returns:
            Tuple of (integrated_procedure_text, prompt_used, gpt_response)
        """
        if not procedures:
            return "No procedures to integrate.", "", ""
        
        # Format segment procedures for prompt
        segment_text = self._format_procedures_for_prompt(procedures)
        print(f"  DEBUG: Formatted segment text length: {len(segment_text)}")
        print(f"  DEBUG: Number of procedures: {len(procedures)}")
        if procedures:
            print(f"  DEBUG: First procedure steps count: {len(procedures[0].steps) if procedures[0].steps else 0}")
            if procedures[0].steps:
                print(f"  DEBUG: First step: {procedures[0].steps[0].description[:100] if procedures[0].steps[0].description else 'Empty'}")
        print(f"  DEBUG: Segment text preview: {segment_text[:200]}...")
        
        # Use common integration prompt
        prompt_template = prompt_loader.get_prompt("procedure_integration_prompts.json", "common_integration")
        prompt = prompt_template.format(segment_procedures=segment_text)
        
        last_error = None
        last_response = ""
        
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"  Calling GPT-5 API for integration (attempt {attempt}/{self.max_retries})...")
                
                # Use analyze_text_only method for text-only integration
                gpt_response = self.client.analyze_text_only(prompt=prompt)
                print(f"  Received response from GPT-5 for integration")
                print(f"  DEBUG: Response length: {len(gpt_response) if gpt_response else 0}")
                print(f"  DEBUG: Response is None: {gpt_response is None}")
                print(f"  DEBUG: Response is empty string: {gpt_response == '' if gpt_response is not None else 'N/A'}")
                print(f"  DEBUG: Response preview: {gpt_response[:100] if gpt_response else 'None'}...")
                
                # Validate the response
                is_valid, validation_details = self._is_valid_integration_result(gpt_response)
                print(f"  DEBUG: Validation result: {is_valid}, Details: {validation_details}")
                
                if is_valid:
                    print(f"  Integration successful on attempt {attempt}")
                    return gpt_response, prompt, gpt_response
                else:
                    error_msg = f"Invalid integration result (attempt {attempt}): {validation_details}"
                    print(f"  Warning: {error_msg}")
                    self._save_warning_log("integration", attempt, error_msg, output_dir, 
                                         raw_response=gpt_response, prompt_used=prompt, 
                                         validation_details=validation_details)
                    last_error = error_msg
                    last_response = gpt_response
                    
                    if attempt < self.max_retries:
                        print(f"  Retrying integration...")
                        time.sleep(2)  # Brief delay before retry
                
            except Exception as e:
                error_msg = f"API error in integration (attempt {attempt}): {e}"
                print(f"  Error: {error_msg}")
                self._save_warning_log("integration", attempt, error_msg, output_dir, 
                                     raw_response=None, prompt_used=prompt, 
                                     validation_details=f"API call failed: {str(e)}")
                last_error = str(e)
                
                if attempt < self.max_retries:
                    print(f"  Retrying integration...")
                    time.sleep(2)  # Brief delay before retry
        
        # All attempts failed - terminate immediately without fallback
        final_error = f"Integration failed after {self.max_retries} attempts. Last error: {last_error}"
        print(f"  CRITICAL: {final_error}")
        raise RuntimeError(final_error)
    
    
    
    def _format_procedures_for_prompt(self, procedures: List[SegmentProcedure]) -> str:
        """Format segment procedures for inclusion in prompts."""
        if not procedures:
            return "No procedures available."
        
        formatted_segments = []
        
        for i, procedure in enumerate(procedures):
            segment_text = f"**Segment {i+1}**:\n"
            
            if procedure.steps:
                for j, step in enumerate(procedure.steps, 1):
                    segment_text += f"{j}. {step.description}\n"
            else:
                segment_text += "No steps recorded.\n"
            
            formatted_segments.append(segment_text)
        
        return "\n".join(formatted_segments)
    
    def _format_single_segment_for_prompt(self, segment: SegmentProcedure, segment_number: int = 1) -> str:
        """Format a single segment procedure for inclusion in prompts."""
        segment_text = f"**Segment {segment_number}**:\n"
        
        if segment.steps:
            for j, step in enumerate(segment.steps, 1):
                segment_text += f"{j}. {step.description}\n"
        else:
            segment_text += "No steps recorded.\n"
        
        return segment_text

    def generate_two_method_integrations(self, 
                                         manual_procedures: List[SegmentProcedure], 
                                         auto_procedures: List[SegmentProcedure],
                                         output_dir: str = None) -> Dict[str, str]:
        """
        Generate integrated procedures using two methods (Manual and Auto).
        
        Args:
            manual_procedures: List of manual object detection segment procedures
            auto_procedures: List of auto object detection segment procedures
            output_dir: Output directory for warning logs
            
        Returns:
            Dictionary with both integration results and prompt information
        """
        print(f"DEBUG: generate_two_method_integrations called")
        
        results = {}
        
        # Method 1: Manual Object Detection
        print(f"DEBUG: Running manual object detection integration...")
        manual_result, manual_prompt, manual_response = self.integrate_procedures(manual_procedures, output_dir)
        results['manual_object_detection'] = manual_result
        results['manual_object_detection_prompt'] = manual_prompt
        results['manual_object_detection_response'] = manual_response
        print(f"DEBUG: Manual integration completed successfully")
        
        # Method 2: Auto Object Detection
        print(f"DEBUG: Running auto object detection integration...")
        auto_result, auto_prompt, auto_response = self.integrate_procedures(auto_procedures, output_dir)
        results['auto_object_detection'] = auto_result
        results['auto_object_detection_prompt'] = auto_prompt
        results['auto_object_detection_response'] = auto_response
        print(f"DEBUG: Auto integration completed successfully")
        
        return results
