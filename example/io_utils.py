"""
I/O utility functions for file saving operations.

This module centralizes all file I/O operations to separate concerns
from business logic and computation.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


def save_text(path: str, content: str, header: Optional[str] = None) -> str:
    """
    Save text content to file with unified error handling.
    
    Args:
        path: Output file path
        content: Text content to save
        header: Optional header to prepend
    
    Returns:
        Path to saved file
    
    Raises:
        RuntimeError: If file saving fails
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + "\n\n")
        f.write("" if content is None else str(content))
    return path


def save_json(path: str, data: Dict[str, Any], indent: int = 2) -> str:
    """
    Save data as JSON file.
    
    Args:
        path: Output file path
        data: Dictionary to save as JSON
        indent: JSON indentation level
    
    Returns:
        Path to saved file
    
    Raises:
        RuntimeError: If file saving fails
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return path


def load_json(path: str) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        path: Input file path
    
    Returns:
        Loaded JSON data as dictionary
    
    Raises:
        RuntimeError: If file loading fails
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_directory(path: str) -> str:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        The directory path
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_output_path(base_dir: str, *parts: str) -> str:
    """
    Build output file path from components.
    
    Args:
        base_dir: Base directory
        *parts: Path components to join
    
    Returns:
        Full output path
    """
    return os.path.join(base_dir, *parts)


def save_procedure_text(path: str, procedure: str, method_name: str) -> str:
    """
    Save procedure text with standardized format.
    
    Args:
        path: Output file path
        procedure: Procedure text
        method_name: Method name for header
    
    Returns:
        Path to saved file
    """
    header = f"{method_name} Procedure\n{'=' * 50}"
    return save_text(path, procedure, header=header)


def save_integration_result(path: str, integration_result: str, integration_type: str) -> str:
    """
    Save integration result with standardized format.
    
    Args:
        path: Output file path
        integration_result: Integration result text
        integration_type: Integration type for header
    
    Returns:
        Path to saved file
    """
    header = f"Integrated Procedure ({integration_type})\n{'=' * 50}"
    return save_text(path, integration_result, header=header)


def save_error_log(path: str, error_message: str, context: str = "") -> str:
    """
    Save error log with standardized format.
    
    Args:
        path: Output file path
        error_message: Error message
        context: Optional context information
    
    Returns:
        Path to saved file
    """
    header = f"Error Log\n{'=' * 50}"
    content = f"{context}\n\n{error_message}" if context else error_message
    return save_text(path, content, header=header)

