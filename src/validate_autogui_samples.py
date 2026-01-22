#!/usr/bin/env python3
"""
Grounding Dataset Sample Validation Script

Comprehensively validates UI-TARS SupervisedDataset JSON files generated from
grounding datasets (AutoGUI, Salesforce grounding_dataset, etc.), ensuring data 
integrity and correctness of the sampling/conversion logic.

Works with any JSON file following the UI-TARS SupervisedDataset format:
- id: unique identifier
- image: list of image paths
- conversations: list with human/gpt messages containing instructions and coordinates
"""

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Set
import math

from PIL import Image, ImageDraw, ImageFont
import ujson
import sys
import os

# #region agent log
log_path = "/Users/lockewang/FIG/Qwen2-VL-Finetune/.cursor/debug.log"
with open(log_path, "a") as f:
    import json
    import time
    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "validate_autogui_samples.py:21", "message": "Before import attempt", "data": {"cwd": os.getcwd(), "sys_path": sys.path, "script_dir": os.path.dirname(os.path.abspath(__file__)), "workspace_root": str(Path(__file__).parent.parent)}, "timestamp": int(time.time() * 1000)}) + "\n")
# #endregion

# Import coordinate normalization logic from sample_autogui.py
# Try multiple import strategies
try:
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "validate_autogui_samples.py:30", "message": "Attempting absolute import", "data": {"import_path": "src.sample_autogui"}, "timestamp": int(time.time() * 1000)}) + "\n")
    # #endregion
    from src.sample_autogui import (
        smart_resize,
        prepare_training_coordinates,
        compute_center_from_box,
        IMAGE_FACTOR,
        MIN_PIXELS,
        MAX_PIXELS,
        MAX_RATIO,
    )
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "validate_autogui_samples.py:40", "message": "Absolute import succeeded", "data": {}, "timestamp": int(time.time() * 1000)}) + "\n")
    # #endregion
except ImportError as e:
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "validate_autogui_samples.py:43", "message": "Absolute import failed", "data": {"error": str(e), "error_type": type(e).__name__}, "timestamp": int(time.time() * 1000)}) + "\n")
    # #endregion
    # Try adding workspace root to sys.path
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    if str(workspace_root) not in sys.path:
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "validate_autogui_samples.py:50", "message": "Adding workspace root to sys.path", "data": {"workspace_root": str(workspace_root), "sys_path_before": sys.path.copy()}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        sys.path.insert(0, str(workspace_root))
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "validate_autogui_samples.py:55", "message": "Retrying import after sys.path modification", "data": {"sys_path_after": sys.path}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        try:
            from src.sample_autogui import (
                smart_resize,
                prepare_training_coordinates,
                compute_center_from_box,
                IMAGE_FACTOR,
                MIN_PIXELS,
                MAX_PIXELS,
                MAX_RATIO,
            )
            # #region agent log
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "validate_autogui_samples.py:67", "message": "Import succeeded after sys.path modification", "data": {}, "timestamp": int(time.time() * 1000)}) + "\n")
            # #endregion
        except ImportError as e2:
            # #region agent log
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "D", "location": "validate_autogui_samples.py:70", "message": "Import still failed after sys.path modification", "data": {"error": str(e2), "error_type": type(e2).__name__}, "timestamp": int(time.time() * 1000)}) + "\n")
            # #endregion
            # Try relative import as fallback
            try:
                # #region agent log
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "validate_autogui_samples.py:74", "message": "Attempting relative import", "data": {}, "timestamp": int(time.time() * 1000)}) + "\n")
                # #endregion
                from .sample_autogui import (
                    smart_resize,
                    prepare_training_coordinates,
                    compute_center_from_box,
                    IMAGE_FACTOR,
                    MIN_PIXELS,
                    MAX_PIXELS,
                    MAX_RATIO,
                )
                # #region agent log
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "validate_autogui_samples.py:84", "message": "Relative import succeeded", "data": {}, "timestamp": int(time.time() * 1000)}) + "\n")
                # #endregion
            except ImportError as e3:
                # #region agent log
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "F", "location": "validate_autogui_samples.py:87", "message": "All import methods failed", "data": {"error": str(e3), "error_type": type(e3).__name__}, "timestamp": int(time.time() * 1000)}) + "\n")
                # #endregion
                raise


def parse_coordinates_from_response(response: str) -> Optional[Tuple[int, int]]:
    """
    Extract coordinates from assistant response.
    
    Format: "Action: click(start_box='(x, y)')"
    
    Returns:
        (x, y) tuple if found, None otherwise
    """
    # Pattern to match start_box='(x, y)' or start_box="(x, y)"
    pattern = r"start_box=['\"]([^'\"]+)['\"]"
    matches = re.findall(pattern, response)
    
    if not matches:
        return None
    
    # Take first match (for click actions, there's only one)
    coords_str = matches[0].strip()
    
    # Remove box markers if present
    coords_str = coords_str.replace("<|box_start|>", "").replace("<|box_end|>", "")
    coords_str = coords_str.strip()
    
    try:
        # Try parsing as tuple/list
        parsed = ast.literal_eval(coords_str)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            x, y = int(parsed[0]), int(parsed[1])
            return (x, y)
    except (ValueError, SyntaxError):
        pass
    
    # Try parsing as "(x,y)" format
    if coords_str.startswith("(") and coords_str.endswith(")"):
        coords_str = coords_str[1:-1]
        parts = coords_str.split(",")
        if len(parts) >= 2:
            try:
                x = int(float(parts[0].strip()))
                y = int(float(parts[1].strip()))
                return (x, y)
            except ValueError:
                pass
    
    return None


def reverse_engineer_original_coords(
    norm_x: int, norm_y: int, img_width: int, img_height: int
) -> Tuple[float, float]:
    """
    Apply inverse transformation of prepare_training_coordinates.
    
    Given normalized coordinates and original image dimensions, compute
    what the original center point would have been.
    
    Args:
        norm_x, norm_y: Normalized coordinates in smart-resized space
        img_width, img_height: Original image dimensions
    
    Returns:
        (original_x, original_y): Reconstructed original center coordinates
    """
    # Get smart-resized dimensions (same logic as prepare_training_coordinates)
    smart_h, smart_w = smart_resize(
        height=img_height,
        width=img_width,
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS
    )
    
    # Inverse transformation
    # norm_x = original_x * smart_w / img_width
    # Therefore: original_x = norm_x * img_width / smart_w
    original_x = norm_x * img_width / smart_w
    original_y = norm_y * img_height / smart_h
    
    return (original_x, original_y)


def validate_entry(
    entry: Dict[str, Any],
    json_path: Path,
    image_folder: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Validate a single entry from the JSON file.
    
    Args:
        entry: JSON entry dict
        json_path: Path to the JSON file (for resolving relative paths)
        image_folder: Base folder for images (default: auto-detect from JSON location)
    
    Returns:
        Validation result dict with:
        - valid: bool
        - errors: List[str]
        - warnings: List[str]
        - metadata: Dict with image dimensions, coordinates, etc.
    """
    errors = []
    warnings = []
    metadata = {}
    
    # 1. JSON Schema Validation
    if "id" not in entry:
        errors.append("Missing required field: 'id'")
        return {"valid": False, "errors": errors, "warnings": warnings, "metadata": metadata}
    
    if "image" not in entry:
        errors.append("Missing required field: 'image'")
        return {"valid": False, "errors": errors, "warnings": warnings, "metadata": metadata}
    
    if "conversations" not in entry:
        errors.append("Missing required field: 'conversations'")
        return {"valid": False, "errors": errors, "warnings": warnings, "metadata": metadata}
    
    entry_id = entry["id"]
    image_refs = entry["image"]
    if isinstance(image_refs, str):
        image_refs = [image_refs]
    
    if not image_refs:
        errors.append("'image' field is empty")
        return {"valid": False, "errors": errors, "warnings": warnings, "metadata": metadata}
    
    # 2. ID-Filename Matching
    image_ref = image_refs[0]  # Use first image reference
    image_filename = Path(image_ref).name
    expected_id = image_filename.replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
    
    if entry_id != expected_id:
        errors.append(f"ID mismatch: entry ID '{entry_id}' doesn't match image filename '{expected_id}'")
    
    # 3. File Existence and Loadability
    if image_folder is None:
        # Auto-detect: assume images are in a directory named {json_stem}_images
        json_dir = json_path.parent
        json_stem = json_path.stem
        image_folder = json_dir / f"{json_stem}_images"
    
    image_path = image_folder / image_filename
    if not image_path.exists():
        # Try relative to JSON file
        image_path = json_path.parent / image_ref
        if not image_path.exists():
            errors.append(f"Image file not found: {image_ref}")
            return {"valid": False, "errors": errors, "warnings": warnings, "metadata": metadata}
    
    # Try to load image
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size
        metadata["image_dimensions"] = [img_width, img_height]
        metadata["image_loaded"] = True
    except Exception as e:
        errors.append(f"Failed to load image: {e}")
        return {"valid": False, "errors": errors, "warnings": warnings, "metadata": metadata}
    
    # 4. Instruction Validation
    conversations = entry["conversations"]
    if not isinstance(conversations, list) or len(conversations) < 2:
        errors.append("Invalid conversations format: expected at least 2 entries")
        return {"valid": False, "errors": errors, "warnings": warnings, "metadata": metadata}
    
    user_prompt = None
    assistant_response = None
    
    for conv in conversations:
        if conv.get("from") == "human":
            user_prompt = conv.get("value", "")
        elif conv.get("from") == "gpt":
            assistant_response = conv.get("value", "")
    
    if not user_prompt:
        errors.append("Missing user prompt in conversations")
    elif not user_prompt.strip():
        warnings.append("User prompt is empty")
    else:
        # Check if instruction is present (should contain "## User Instruction")
        if "## User Instruction" not in user_prompt:
            warnings.append("User prompt doesn't contain '## User Instruction' marker")
        else:
            # Extract instruction text
            instruction_match = re.search(r"## User Instruction\s+(.+)", user_prompt, re.DOTALL)
            if instruction_match:
                instruction = instruction_match.group(1).strip()
                if not instruction:
                    warnings.append("Instruction text is empty")
                metadata["instruction_length"] = len(instruction)
    
    if not assistant_response:
        errors.append("Missing assistant response in conversations")
        return {"valid": False, "errors": errors, "warnings": warnings, "metadata": metadata}
    
    # 5. Coordinate Parsing
    coords = parse_coordinates_from_response(assistant_response)
    if coords is None:
        errors.append(f"Failed to parse coordinates from response: {assistant_response[:100]}")
        return {"valid": False, "errors": errors, "warnings": warnings, "metadata": metadata}
    
    norm_x, norm_y = coords
    metadata["normalized_coords"] = [norm_x, norm_y]
    
    # 6. Coordinate Format Validation
    if norm_x < 0 or norm_y < 0:
        errors.append(f"Coordinates are negative: ({norm_x}, {norm_y})")
    
    # 7. Bounds Checking (using smart_resize)
    try:
        # #region agent log
        log_path = "/Users/lockewang/FIG/Qwen2-VL-Finetune/.cursor/debug.log"
        with open(log_path, "a") as f:
            import json
            import time
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "G", "location": "validate_autogui_samples.py:325", "message": "Before smart_resize", "data": {"img_width": img_width, "img_height": img_height, "norm_x": norm_x, "norm_y": norm_y}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        smart_h, smart_w = smart_resize(
            height=img_height,
            width=img_width,
            factor=IMAGE_FACTOR,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS
        )
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "G", "location": "validate_autogui_samples.py:336", "message": "After smart_resize", "data": {"smart_h": smart_h, "smart_w": smart_w, "smart_dims_as_stored": [smart_w, smart_h]}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        metadata["smart_dimensions"] = [smart_w, smart_h]
        
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "G", "location": "validate_autogui_samples.py:340", "message": "Bounds check", "data": {"norm_x": norm_x, "norm_y": norm_y, "smart_w": smart_w, "smart_h": smart_h, "x_check": f"{norm_x} >= {smart_w}", "y_check": f"{norm_y} >= {smart_h}", "x_in_bounds": norm_x < smart_w, "y_in_bounds": norm_y < smart_h}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        if norm_x >= smart_w or norm_y >= smart_h:
            errors.append(
                f"Coordinates out of bounds: ({norm_x}, {norm_y}) exceeds "
                f"smart-resized dimensions ({smart_w}, {smart_h})"
            )
        else:
            metadata["coords_in_bounds"] = True
    except Exception as e:
        errors.append(f"Failed to compute smart-resized dimensions: {e}")
        metadata["coords_in_bounds"] = False
    
    # 8. Reverse-Engineering Validation
    try:
        original_x, original_y = reverse_engineer_original_coords(
            norm_x, norm_y, img_width, img_height
        )
        metadata["reconstructed_original_coords"] = [original_x, original_y]
        
        # Check if reconstructed point is within image bounds
        if 0 <= original_x <= img_width and 0 <= original_y <= img_height:
            metadata["reconstructed_in_bounds"] = True
        else:
            warnings.append(
                f"Reconstructed original coordinates ({original_x:.1f}, {original_y:.1f}) "
                f"are outside image bounds ({img_width}, {img_height})"
            )
    except Exception as e:
        warnings.append(f"Failed to reverse-engineer original coordinates: {e}")
    
    valid = len(errors) == 0
    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "metadata": metadata
    }


def create_visualization(
    entry: Dict[str, Any],
    validation_result: Dict[str, Any],
    image_path: Path,
    output_path: Path
) -> bool:
    """
    Create a visualization image with overlaid coordinates and bounding box.
    
    Args:
        entry: JSON entry dict
        validation_result: Result from validate_entry
        image_path: Path to original image
        output_path: Path to save visualization
    
    Returns:
        True if successful, False otherwise
    """
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        metadata = validation_result.get("metadata", {})
        norm_coords = metadata.get("normalized_coords")
        reconstructed_coords = metadata.get("reconstructed_original_coords")
        img_dims = metadata.get("image_dimensions", [])
        smart_dims = metadata.get("smart_dimensions", [])
        
        # Draw normalized coordinate point (red)
        if norm_coords:
            # Scale normalized coords to original image size for visualization
            if smart_dims and img_dims:
                smart_w, smart_h = smart_dims
                img_w, img_h = img_dims
                vis_x = int(norm_coords[0] * img_w / smart_w)
                vis_y = int(norm_coords[1] * img_h / smart_h)
            else:
                vis_x, vis_y = norm_coords
            
            # Draw red circle for normalized coordinate
            radius = max(5, min(img_dims) // 100) if img_dims else 5
            draw.ellipse(
                [vis_x - radius, vis_y - radius, vis_x + radius, vis_y + radius],
                fill="red",
                outline="darkred",
                width=2
            )
        
        # Draw reconstructed original coordinate point (blue)
        if reconstructed_coords:
            orig_x, orig_y = reconstructed_coords
            if 0 <= orig_x <= img_dims[0] and 0 <= orig_y <= img_dims[1]:
                radius = max(3, min(img_dims) // 150) if img_dims else 3
                draw.ellipse(
                    [int(orig_x) - radius, int(orig_y) - radius,
                     int(orig_x) + radius, int(orig_y) + radius],
                    fill="blue",
                    outline="darkblue",
                    width=1
                )
        
        # Extract instruction text for display
        instruction_text = ""
        conversations = entry.get("conversations", [])
        for conv in conversations:
            if conv.get("from") == "human":
                user_prompt = conv.get("value", "")
                # Extract instruction text
                instruction_match = re.search(r"## User Instruction\s+(.+)", user_prompt, re.DOTALL)
                if instruction_match:
                    instruction_text = instruction_match.group(1).strip()
                    # Truncate if too long
                    if len(instruction_text) > 200:
                        instruction_text = instruction_text[:197] + "..."
                break
        
        # Add text labels
        try:
            # Try to use a default font
            font = ImageFont.load_default()
            # Try to get a larger font if available
            try:
                font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except:
                try:
                    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except:
                    font_large = font
        except:
            font = None
            font_large = None
        
        label_y = 10
        label_x = 10
        
        # Draw instruction text with background
        if instruction_text:
            # Calculate text size
            if font_large:
                bbox = draw.textbbox((0, 0), instruction_text, font=font_large)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # Estimate if font not available
                text_width = len(instruction_text) * 8
                text_height = 20
            
            # Draw semi-transparent background for instruction
            overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            padding = 8
            overlay_draw.rectangle(
                [label_x - padding, label_y - padding, 
                 label_x + text_width + padding, label_y + text_height + padding],
                fill=(255, 255, 200, 200),  # Light yellow background
                outline=(0, 0, 0, 255),
                width=2
            )
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # Draw instruction text
            draw.text((label_x, label_y), instruction_text, fill="black", font=font_large or font)
            label_y += text_height + 15
        
        if norm_coords:
            label = f"Norm: ({norm_coords[0]}, {norm_coords[1]})"
            draw.text((label_x, label_y), label, fill="red", font=font)
            label_y += 20
        
        if reconstructed_coords:
            label = f"Orig: ({reconstructed_coords[0]:.1f}, {reconstructed_coords[1]:.1f})"
            draw.text((label_x, label_y), label, fill="blue", font=font)
            label_y += 20
        
        if img_dims:
            label = f"Size: {img_dims[0]}x{img_dims[1]}"
            draw.text((label_x, label_y), label, fill="black", font=font)
            label_y += 20
        
        if smart_dims:
            label = f"Smart: {smart_dims[0]}x{smart_dims[1]}"
            draw.text((label_x, label_y), label, fill="gray", font=font)
        
        # Save visualization
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return True
    except Exception as e:
        print(f"Warning: Failed to create visualization: {e}")
        return False


def validate_json_file(
    json_path: Path,
    image_folder: Optional[Path] = None,
    num_visualize: int = 5,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Main entry point: validate all entries in JSON file.
    
    Args:
        json_path: Path to JSON file to validate
        image_folder: Base folder for images (default: auto-detect)
        num_visualize: Number of samples to visualize (0 to disable)
        output_dir: Directory for validation outputs (default: same as JSON location)
    
    Returns:
        Comprehensive validation report dict
    """
    print(f"Loading JSON file: {json_path}")
    try:
        with open(json_path, 'r') as f:
            entries = ujson.load(f)
    except Exception as e:
        return {
            "error": f"Failed to load JSON file: {e}",
            "summary": {},
            "entries": []
        }
    
    if not isinstance(entries, list):
        return {
            "error": "JSON file does not contain a list of entries",
            "summary": {},
            "entries": []
        }
    
    print(f"Found {len(entries)} entries to validate")
    
    # Auto-detect image folder if not provided
    if image_folder is None:
        json_dir = json_path.parent
        json_stem = json_path.stem
        image_folder = json_dir / f"{json_stem}_images"
        print(f"Auto-detected image folder: {image_folder}")
    
    # Set output directory
    if output_dir is None:
        output_dir = json_path.parent
    
    # Validate each entry
    validation_results = []
    errors_by_category = defaultdict(int)
    
    for i, entry in enumerate(entries):
        entry_id = entry.get("id", f"entry_{i}")
        result = validate_entry(entry, json_path, image_folder)
        result["entry_id"] = entry_id
        validation_results.append(result)
        
        # Count errors by category
        for error in result["errors"]:
            if "ID mismatch" in error:
                errors_by_category["id_mismatch"] += 1
            elif "not found" in error.lower():
                errors_by_category["file_not_found"] += 1
            elif "out of bounds" in error.lower():
                errors_by_category["coordinate_out_of_bounds"] += 1
            elif "parse" in error.lower():
                errors_by_category["coordinate_parse_error"] += 1
            elif "load" in error.lower() or "Failed to load" in error:
                errors_by_category["image_load_error"] += 1
            else:
                errors_by_category["other"] += 1
        
        if (i + 1) % 100 == 0:
            print(f"Validated {i + 1}/{len(entries)} entries...")
    
    # Generate visualizations
    vis_dir = None
    if num_visualize > 0:
        vis_dir = output_dir / "validation_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating visualizations for {min(num_visualize, len(entries))} samples...")
        
        # Visualize first N valid entries, or first N entries if none are valid
        visualized = 0
        for i, (entry, result) in enumerate(zip(entries, validation_results)):
            if visualized >= num_visualize:
                break
            
            entry_id = entry.get("id", f"entry_{i}")
            image_refs = entry.get("image", [])
            if isinstance(image_refs, str):
                image_refs = [image_refs]
            
            if not image_refs:
                continue
            
            image_filename = Path(image_refs[0]).name
            image_path = image_folder / image_filename
            
            if image_path.exists():
                vis_path = vis_dir / f"{entry_id}_validation.png"
                if create_visualization(entry, result, image_path, vis_path):
                    visualized += 1
    
    # Compute summary statistics
    valid_count = sum(1 for r in validation_results if r["valid"])
    invalid_count = len(validation_results) - valid_count
    total_warnings = sum(len(r["warnings"]) for r in validation_results)
    
    summary = {
        "total_entries": len(entries),
        "valid_entries": valid_count,
        "invalid_entries": invalid_count,
        "total_warnings": total_warnings,
        "errors_by_category": dict(errors_by_category)
    }
    
    return {
        "summary": summary,
        "entries": validation_results,
        "visualization_dir": str(vis_dir) if vis_dir and vis_dir.exists() else None
    }


def print_console_report(report: Dict[str, Any]) -> None:
    """Print validation report to console."""
    summary = report.get("summary", {})
    
    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)
    print(f"Total entries: {summary.get('total_entries', 0)}")
    print(f"Valid entries: {summary.get('valid_entries', 0)}")
    print(f"Invalid entries: {summary.get('invalid_entries', 0)}")
    print(f"Total warnings: {summary.get('total_warnings', 0)}")
    print()
    
    errors_by_category = summary.get("errors_by_category", {})
    if errors_by_category:
        print("Errors by category:")
        print("-" * 80)
        for category, count in sorted(errors_by_category.items(), key=lambda x: -x[1]):
            print(f"  {category:30s}: {count:4d}")
        print()
    
    # Show sample errors
    entries = report.get("entries", [])
    invalid_entries = [e for e in entries if not e.get("valid", True)]
    
    if invalid_entries:
        print(f"Sample errors (showing first 10 of {len(invalid_entries)}):")
        print("-" * 80)
        for entry in invalid_entries[:10]:
            entry_id = entry.get("entry_id", "unknown")
            errors = entry.get("errors", [])
            print(f"  Entry ID: {entry_id}")
            for error in errors[:3]:  # Show first 3 errors per entry
                print(f"    - {error}")
        if len(invalid_entries) > 10:
            print(f"  ... and {len(invalid_entries) - 10} more invalid entries")
        print()
    
    # Show sample warnings
    entries_with_warnings = [e for e in entries if e.get("warnings")]
    if entries_with_warnings:
        print(f"Sample warnings (showing first 5 of {len(entries_with_warnings)}):")
        print("-" * 80)
        for entry in entries_with_warnings[:5]:
            entry_id = entry.get("entry_id", "unknown")
            warnings = entry.get("warnings", [])
            print(f"  Entry ID: {entry_id}")
            for warning in warnings[:2]:  # Show first 2 warnings per entry
                print(f"    - {warning}")
        if len(entries_with_warnings) > 5:
            print(f"  ... and {len(entries_with_warnings) - 5} more entries with warnings")
        print()
    
    vis_dir = report.get("visualization_dir")
    if vis_dir:
        print(f"Visualizations saved to: {vis_dir}")
    
    print("="*80)


def print_split_validation_summary(report: Dict[str, Any]) -> None:
    """Print a summary of train/val split validation results."""
    split_val = report.get("split_validation", {})
    train_report = report.get("train_report", {})
    val_report = report.get("val_report", {})
    certification = report.get("certification")
    
    print("\n" + "="*80)
    print("SPLIT VALIDATION SUMMARY")
    print("="*80)
    
    # Integrity
    integrity = split_val.get("integrity", {})
    print(f"Split Integrity: {'✓ PASS' if integrity.get('valid') else '✗ FAIL'}")
    if integrity.get("id_overlaps"):
        print(f"  - ID overlaps: {len(integrity['id_overlaps'])}")
    if integrity.get("image_overlaps"):
        print(f"  - Image overlaps: {len(integrity['image_overlaps'])}")
    
    # Metadata
    metadata = split_val.get("metadata", {})
    print(f"Metadata Validation: {'✓ PASS' if metadata.get('valid') else '✗ FAIL'}")
    
    # Distribution
    distribution = split_val.get("distribution", {})
    if distribution.get("train_distribution"):
        print(f"Stratified Distribution: {'✓ MAINTAINED' if not distribution.get('errors') else '✗ ISSUES'}")
        if distribution.get("warnings"):
            print(f"  - Warnings: {len(distribution['warnings'])}")
    
    # Quality
    train_summary = train_report.get("summary", {})
    val_summary = val_report.get("summary", {})
    train_validity = train_summary.get("valid_entries", 0) / max(train_summary.get("total_entries", 1), 1)
    val_validity = val_summary.get("valid_entries", 0) / max(val_summary.get("total_entries", 1), 1)
    
    print(f"\nData Quality:")
    print(f"  Train: {train_summary.get('valid_entries', 0)}/{train_summary.get('total_entries', 0)} valid ({train_validity:.2%})")
    print(f"  Val: {val_summary.get('valid_entries', 0)}/{val_summary.get('total_entries', 0)} valid ({val_validity:.2%})")
    
    # Statistics
    stats = split_val.get("statistics_comparison", {})
    if stats.get("comparisons"):
        print(f"\nStatistical Consistency: {'✓ CONSISTENT' if not stats.get('warnings') else '⚠ DIFFERENCES'}")
        if stats.get("warnings"):
            print(f"  - Warnings: {len(stats['warnings'])}")
    
    # Certification
    if certification:
        print(f"\nGolden Dataset Status: {'✓ CERTIFIED' if certification.get('is_golden') else '✗ NOT CERTIFIED'}")
        if not certification.get("is_golden"):
            print(f"  - Issues: {len(certification.get('issues', []))}")
            print(f"  - Recommendations: {len(certification.get('recommendations', []))}")
    
    print("="*80)


def validate_split_integrity(
    train_entries: List[Dict[str, Any]],
    val_entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate that train and validation sets have no overlaps.
    
    Args:
        train_entries: List of train entries
        val_entries: List of validation entries
    
    Returns:
        Validation result dict with overlaps and uniqueness checks
    """
    # Extract IDs, image filenames, and UUIDs
    train_ids = {entry.get("id") for entry in train_entries if entry.get("id")}
    val_ids = {entry.get("id") for entry in val_entries if entry.get("id")}
    
    train_images = set()
    val_images = set()
    
    for entry in train_entries:
        image_refs = entry.get("image", [])
        if isinstance(image_refs, str):
            image_refs = [image_refs]
        for img_ref in image_refs:
            train_images.add(Path(img_ref).name)
    
    for entry in val_entries:
        image_refs = entry.get("image", [])
        if isinstance(image_refs, str):
            image_refs = [image_refs]
        for img_ref in image_refs:
            val_images.add(Path(img_ref).name)
    
    # Check for overlaps
    id_overlaps = train_ids & val_ids
    image_overlaps = train_images & val_images
    
    result = {
        "valid": len(id_overlaps) == 0 and len(image_overlaps) == 0,
        "train_count": len(train_entries),
        "val_count": len(val_entries),
        "train_unique_ids": len(train_ids),
        "val_unique_ids": len(val_ids),
        "id_overlaps": list(id_overlaps),
        "image_overlaps": list(image_overlaps),
        "errors": []
    }
    
    if id_overlaps:
        result["errors"].append(f"Found {len(id_overlaps)} overlapping entry IDs: {list(id_overlaps)[:10]}")
    
    if image_overlaps:
        result["errors"].append(f"Found {len(image_overlaps)} overlapping image files: {list(image_overlaps)[:10]}")
    
    return result


def validate_split_metadata(
    train_meta_path: Optional[Path],
    val_meta_path: Optional[Path],
    expected_val_ratio: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate split metadata files.
    
    Args:
        train_meta_path: Path to train metadata file
        val_meta_path: Path to validation metadata file
        expected_val_ratio: Expected validation ratio (for verification)
    
    Returns:
        Validation result dict
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "metadata": {}
    }
    
    # Check file existence
    if train_meta_path and not train_meta_path.exists():
        result["errors"].append(f"Train metadata file not found: {train_meta_path}")
        result["valid"] = False
        return result
    
    if val_meta_path and not val_meta_path.exists():
        result["errors"].append(f"Validation metadata file not found: {val_meta_path}")
        result["valid"] = False
        return result
    
    train_meta = {}
    val_meta = {}
    
    if train_meta_path:
        try:
            with open(train_meta_path, 'r') as f:
                train_meta = ujson.load(f)
        except Exception as e:
            result["errors"].append(f"Failed to load train metadata: {e}")
            result["valid"] = False
    
    if val_meta_path:
        try:
            with open(val_meta_path, 'r') as f:
                val_meta = ujson.load(f)
        except Exception as e:
            result["errors"].append(f"Failed to load validation metadata: {e}")
            result["valid"] = False
    
    if not result["valid"]:
        return result
    
    result["metadata"]["train"] = train_meta
    result["metadata"]["val"] = val_meta
    
    # Check split information
    train_split_type = train_meta.get("split_type")
    val_split_type = val_meta.get("split_type")
    
    if train_split_type != "train":
        result["warnings"].append(f"Train metadata split_type is '{train_split_type}', expected 'train'")
    
    if val_split_type != "validation":
        result["warnings"].append(f"Val metadata split_type is '{val_split_type}', expected 'validation'")
    
    # Check consistency
    train_seed = train_meta.get("seed")
    val_seed = val_meta.get("seed")
    if train_seed != val_seed:
        result["errors"].append(f"Seed mismatch: train={train_seed}, val={val_seed}")
        result["valid"] = False
    
    train_source = train_meta.get("source")
    val_source = val_meta.get("source")
    if train_source != val_source:
        result["errors"].append(f"Source mismatch: train={train_source}, val={val_source}")
        result["valid"] = False
    
    # Check split ratio
    train_val_ratio = train_meta.get("val_ratio")
    val_val_ratio = val_meta.get("val_ratio")
    
    if train_val_ratio is not None and val_val_ratio is not None:
        if abs(train_val_ratio - val_val_ratio) > 0.001:
            result["errors"].append(f"Val ratio mismatch: train={train_val_ratio}, val={val_val_ratio}")
            result["valid"] = False
        
        if expected_val_ratio is not None:
            if abs(train_val_ratio - expected_val_ratio) > 0.01:
                result["warnings"].append(
                    f"Val ratio differs from expected: got {train_val_ratio}, expected {expected_val_ratio}"
                )
    
    return result


def validate_stratified_distribution(
    train_entries: List[Dict[str, Any]],
    val_entries: List[Dict[str, Any]],
    train_meta: Optional[Dict[str, Any]] = None,
    val_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate that stratified distribution is maintained.
    
    Args:
        train_entries: List of train entries
        val_entries: List of validation entries
        train_meta: Train metadata (optional)
        val_meta: Validation metadata (optional)
    
    Returns:
        Validation result dict with distribution comparison
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "train_distribution": {},
        "val_distribution": {},
        "distribution_differences": {}
    }
    
    # Try to get distribution from metadata first
    train_dist = train_meta.get("dataset_distribution", {}) if train_meta else {}
    val_dist = val_meta.get("dataset_distribution", {}) if val_meta else {}
    
    # If not in metadata, we can't validate stratified distribution
    if not train_dist or not val_dist:
        result["warnings"].append("Dataset distribution not found in metadata - cannot validate stratified split")
        return result
    
    result["train_distribution"] = train_dist
    result["val_distribution"] = val_dist
    
    # Calculate total counts
    train_total = sum(train_dist.values())
    val_total = sum(val_dist.values())
    
    if train_total == 0 or val_total == 0:
        result["errors"].append("Empty distribution in train or val")
        result["valid"] = False
        return result
    
    # Check that each dataset source has samples in both splits
    all_datasets = set(train_dist.keys()) | set(val_dist.keys())
    
    for dataset in all_datasets:
        train_count = train_dist.get(dataset, 0)
        val_count = val_dist.get(dataset, 0)
        
        if train_count == 0:
            result["warnings"].append(f"Dataset '{dataset}' has no samples in train set")
        
        if val_count == 0:
            result["warnings"].append(f"Dataset '{dataset}' has no samples in validation set")
        
        # Calculate proportions
        train_prop = train_count / train_total if train_total > 0 else 0
        val_prop = val_count / val_total if val_total > 0 else 0
        
        prop_diff = abs(train_prop - val_prop)
        result["distribution_differences"][dataset] = {
            "train_count": train_count,
            "val_count": val_count,
            "train_proportion": train_prop,
            "val_proportion": val_prop,
            "difference": prop_diff
        }
        
        # Flag significant differences (>10% relative difference)
        if train_prop > 0 and prop_diff / train_prop > 0.1:
            result["warnings"].append(
                f"Dataset '{dataset}' has significant distribution difference: "
                f"train={train_prop:.2%}, val={val_prop:.2%}"
            )
    
    return result


def compare_statistics(
    train_report: Dict[str, Any],
    val_report: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare statistical features between train and validation sets.
    
    Args:
        train_report: Validation report for train set
        val_report: Validation report for validation set
    
    Returns:
        Comparison result dict
    """
    result = {
        "valid": True,
        "warnings": [],
        "comparisons": {}
    }
    
    train_summary = train_report.get("summary", {})
    val_summary = val_report.get("summary", {})
    
    # Compare error rates
    train_total = train_summary.get("total_entries", 0)
    val_total = val_summary.get("total_entries", 0)
    
    train_error_rate = train_summary.get("invalid_entries", 0) / train_total if train_total > 0 else 0
    val_error_rate = val_summary.get("invalid_entries", 0) / val_total if val_total > 0 else 0
    
    error_rate_diff = abs(train_error_rate - val_error_rate)
    result["comparisons"]["error_rate"] = {
        "train": train_error_rate,
        "val": val_error_rate,
        "difference": error_rate_diff
    }
    
    if error_rate_diff > 0.05:  # 5% difference threshold
        result["warnings"].append(
            f"Error rate difference >5%: train={train_error_rate:.2%}, val={val_error_rate:.2%}"
        )
    
    # Compare warning rates
    train_warning_rate = train_summary.get("total_warnings", 0) / train_total if train_total > 0 else 0
    val_warning_rate = val_summary.get("total_warnings", 0) / val_total if val_total > 0 else 0
    
    warning_rate_diff = abs(train_warning_rate - val_warning_rate)
    result["comparisons"]["warning_rate"] = {
        "train": train_warning_rate,
        "val": val_warning_rate,
        "difference": warning_rate_diff
    }
    
    if warning_rate_diff > 0.05:
        result["warnings"].append(
            f"Warning rate difference >5%: train={train_warning_rate:.2%}, val={val_warning_rate:.2%}"
        )
    
    # Extract coordinate statistics from entry metadata
    train_coords = []
    val_coords = []
    
    for entry_result in train_report.get("entries", []):
        coords = entry_result.get("metadata", {}).get("normalized_coords")
        if coords:
            train_coords.append(coords)
    
    for entry_result in val_report.get("entries", []):
        coords = entry_result.get("metadata", {}).get("normalized_coords")
        if coords:
            val_coords.append(coords)
    
    if train_coords and val_coords:
        train_x = [c[0] for c in train_coords]
        train_y = [c[1] for c in train_coords]
        val_x = [c[0] for c in val_coords]
        val_y = [c[1] for c in val_coords]
        
        def compute_stats(values):
            if not values:
                return {}
            return {
                "mean": sum(values) / len(values),
                "std": math.sqrt(sum((x - sum(values) / len(values))**2 for x in values) / len(values)),
                "min": min(values),
                "max": max(values)
            }
        
        result["comparisons"]["coordinates_x"] = {
            "train": compute_stats(train_x),
            "val": compute_stats(val_x)
        }
        result["comparisons"]["coordinates_y"] = {
            "train": compute_stats(train_y),
            "val": compute_stats(val_y)
        }
    
    return result


def is_golden_dataset(
    integrity_result: Dict[str, Any],
    metadata_result: Dict[str, Any],
    distribution_result: Dict[str, Any],
    train_report: Dict[str, Any],
    val_report: Dict[str, Any],
    stats_comparison: Dict[str, Any],
    expected_val_ratio: Optional[float] = None
) -> Dict[str, Any]:
    """
    Determine if train/val split meets golden dataset criteria.
    
    Args:
        integrity_result: Result from validate_split_integrity
        metadata_result: Result from validate_split_metadata
        distribution_result: Result from validate_stratified_distribution
        train_report: Validation report for train set
        val_report: Validation report for validation set
        stats_comparison: Result from compare_statistics
        expected_val_ratio: Expected validation ratio
    
    Returns:
        Certification result dict
    """
    criteria = {
        "zero_overlap": False,
        "high_validity": False,
        "balanced_split": False,
        "consistent_quality": False,
        "metadata_consistent": False,
        "stratified_maintained": True  # Default True, set False if issues found
    }
    
    issues = []
    recommendations = []
    
    # 1. Zero overlap
    if integrity_result.get("valid") and len(integrity_result.get("id_overlaps", [])) == 0:
        criteria["zero_overlap"] = True
    else:
        issues.append("Found overlaps between train and validation sets")
        recommendations.append("Remove overlapping entries from one of the splits")
    
    # 2. High validity (>99%)
    train_summary = train_report.get("summary", {})
    val_summary = val_report.get("summary", {})
    
    train_validity = train_summary.get("valid_entries", 0) / max(train_summary.get("total_entries", 1), 1)
    val_validity = val_summary.get("valid_entries", 0) / max(val_summary.get("total_entries", 1), 1)
    
    if train_validity >= 0.99 and val_validity >= 0.99:
        criteria["high_validity"] = True
    else:
        issues.append(f"Validity below 99%: train={train_validity:.2%}, val={val_validity:.2%}")
        recommendations.append("Fix invalid entries or regenerate splits")
    
    # 3. Balanced split
    if expected_val_ratio is not None:
        train_count = train_summary.get("total_entries", 0)
        val_count = val_summary.get("total_entries", 0)
        total = train_count + val_count
        actual_ratio = val_count / total if total > 0 else 0
        
        if abs(actual_ratio - expected_val_ratio) <= 0.01:  # 1% tolerance
            criteria["balanced_split"] = True
        else:
            issues.append(
                f"Split ratio mismatch: expected {expected_val_ratio:.1%}, "
                f"got {actual_ratio:.1%}"
            )
            recommendations.append("Regenerate splits with correct ratio")
    
    # 4. Consistent quality
    error_rate_diff = stats_comparison.get("comparisons", {}).get("error_rate", {}).get("difference", 1.0)
    if error_rate_diff <= 0.05:  # 5% difference threshold
        criteria["consistent_quality"] = True
    else:
        issues.append(f"Quality difference >5% between splits")
        recommendations.append("Investigate quality differences between splits")
    
    # 5. Metadata consistent
    if metadata_result.get("valid"):
        criteria["metadata_consistent"] = True
    else:
        issues.append("Metadata validation failed")
        recommendations.append("Fix metadata files or regenerate splits")
    
    # 6. Stratified maintained (if applicable)
    if distribution_result.get("warnings"):
        # Only fail if there are errors, warnings are acceptable
        if distribution_result.get("errors"):
            criteria["stratified_maintained"] = False
            issues.append("Stratified distribution validation failed")
            recommendations.append("Regenerate splits with proper stratification")
    
    all_passed = all(criteria.values())
    
    return {
        "is_golden": all_passed,
        "criteria": criteria,
        "issues": issues,
        "recommendations": recommendations,
        "train_validity": train_validity,
        "val_validity": val_validity
    }


def validate_train_val_split(
    train_json_path: Path,
    val_json_path: Path,
    num_visualize: int = 5,
    expected_val_ratio: Optional[float] = None,
    certify_golden: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive validation of train/val split pair.
    
    Args:
        train_json_path: Path to train JSON file
        val_json_path: Path to validation JSON file
        num_visualize: Number of samples to visualize
        expected_val_ratio: Expected validation ratio
        certify_golden: Whether to perform golden dataset certification
    
    Returns:
        Comprehensive validation report
    """
    print("="*80)
    print("TRAIN/VAL SPLIT VALIDATION")
    print("="*80)
    print(f"Train file: {train_json_path}")
    print(f"Val file: {val_json_path}")
    print()
    
    # Load JSON files
    try:
        with open(train_json_path, 'r') as f:
            train_entries = ujson.load(f)
        with open(val_json_path, 'r') as f:
            val_entries = ujson.load(f)
    except Exception as e:
        return {
            "error": f"Failed to load JSON files: {e}",
            "valid": False
        }
    
    print(f"Loaded {len(train_entries)} train entries and {len(val_entries)} val entries")
    print()
    
    # 1. Split integrity validation
    print("1. Validating split integrity...")
    integrity_result = validate_split_integrity(train_entries, val_entries)
    if integrity_result["valid"]:
        print("   ✓ No overlaps found")
    else:
        print("   ✗ Overlaps detected!")
        for error in integrity_result["errors"]:
            print(f"     - {error}")
    print()
    
    # 2. Metadata validation
    print("2. Validating split metadata...")
    train_meta_path = train_json_path.with_suffix('.meta.json')
    val_meta_path = val_json_path.with_suffix('.meta.json')
    
    metadata_result = validate_split_metadata(
        train_meta_path if train_meta_path.exists() else None,
        val_meta_path if val_meta_path.exists() else None,
        expected_val_ratio
    )
    
    train_meta = metadata_result.get("metadata", {}).get("train", {})
    val_meta = metadata_result.get("metadata", {}).get("val", {})
    
    if metadata_result["valid"]:
        print("   ✓ Metadata validation passed")
    else:
        print("   ✗ Metadata validation failed!")
        for error in metadata_result["errors"]:
            print(f"     - {error}")
    if metadata_result["warnings"]:
        for warning in metadata_result["warnings"]:
            print(f"     ⚠ {warning}")
    print()
    
    # 3. Stratified distribution validation
    print("3. Validating stratified distribution...")
    distribution_result = validate_stratified_distribution(
        train_entries, val_entries, train_meta, val_meta
    )
    if distribution_result["valid"] and not distribution_result["warnings"]:
        print("   ✓ Stratified distribution maintained")
    elif distribution_result["warnings"]:
        print("   ⚠ Distribution warnings:")
        for warning in distribution_result["warnings"][:5]:
            print(f"     - {warning}")
    print()
    
    # 4. Data quality validation (per split)
    print("4. Validating data quality (train set)...")
    train_report = validate_json_file(
        train_json_path,
        num_visualize=0,  # Don't visualize during split validation
        output_dir=train_json_path.parent
    )
    
    print("5. Validating data quality (validation set)...")
    val_report = validate_json_file(
        val_json_path,
        num_visualize=num_visualize,
        output_dir=val_json_path.parent
    )
    
    train_summary = train_report.get("summary", {})
    val_summary = val_report.get("summary", {})
    
    print(f"   Train: {train_summary.get('valid_entries', 0)}/{train_summary.get('total_entries', 0)} valid")
    print(f"   Val: {val_summary.get('valid_entries', 0)}/{val_summary.get('total_entries', 0)} valid")
    print()
    
    # 5. Statistical comparison
    print("6. Comparing statistics...")
    stats_comparison = compare_statistics(train_report, val_report)
    if stats_comparison["warnings"]:
        for warning in stats_comparison["warnings"]:
            print(f"   ⚠ {warning}")
    else:
        print("   ✓ Statistics are consistent")
    print()
    
    # 6. Golden dataset certification
    certification = None
    if certify_golden:
        print("7. Golden dataset certification...")
        certification = is_golden_dataset(
            integrity_result,
            metadata_result,
            distribution_result,
            train_report,
            val_report,
            stats_comparison,
            expected_val_ratio
        )
        
        if certification["is_golden"]:
            print("   ✓ GOLDEN DATASET CERTIFIED")
        else:
            print("   ✗ Does not meet golden dataset criteria")
            print("   Issues:")
            for issue in certification["issues"]:
                print(f"     - {issue}")
        print()
    
    # Compile comprehensive report
    report = {
        "split_validation": {
            "integrity": integrity_result,
            "metadata": metadata_result,
            "distribution": distribution_result,
            "statistics_comparison": stats_comparison
        },
        "train_report": train_report,
        "val_report": val_report,
        "certification": certification
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate UI-TARS SupervisedDataset JSON files for correctness (works with AutoGUI, Salesforce grounding_dataset, etc.)"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=False,
        help="Path to JSON file to validate (or train JSON if --validate_split)"
    )
    parser.add_argument(
        "--val_json",
        type=str,
        default=None,
        help="Path to validation JSON file (required if --validate_split)"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Base folder for images (default: auto-detect from JSON location)"
    )
    parser.add_argument(
        "--num_visualize",
        type=int,
        default=5,
        help="Number of samples to visualize (default: 5, 0 to disable)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for validation outputs (default: same as JSON location)"
    )
    parser.add_argument(
        "--validate_split",
        action="store_true",
        help="Validate train/val split pair (requires --json_path and --val_json or --auto_detect_val)"
    )
    parser.add_argument(
        "--auto_detect_val",
        action="store_true",
        help="Auto-detect validation file from train file name (e.g., *_train.json -> *_val.json)"
    )
    parser.add_argument(
        "--certify_golden",
        action="store_true",
        help="Perform golden dataset certification (requires --validate_split)"
    )
    parser.add_argument(
        "--expected_val_ratio",
        type=float,
        default=None,
        help="Expected validation ratio (e.g., 0.1 for 10%%) for golden dataset certification"
    )
    
    args = parser.parse_args()
    
    # Validate split mode
    if args.validate_split:
        if not args.json_path:
            parser.error("--json_path is required when using --validate_split")
        
        train_json_path = Path(args.json_path)
        if not train_json_path.exists():
            print(f"Error: Train JSON file not found: {train_json_path}")
            return
        
        # Determine validation JSON path
        if args.val_json:
            val_json_path = Path(args.val_json)
        elif args.auto_detect_val:
            # Auto-detect: replace _train with _val
            stem = train_json_path.stem
            if stem.endswith("_train"):
                val_stem = stem[:-6] + "_val"
            else:
                val_stem = stem + "_val"
            val_json_path = train_json_path.parent / f"{val_stem}{train_json_path.suffix}"
        else:
            parser.error("Either --val_json or --auto_detect_val must be specified with --validate_split")
        
        if not val_json_path.exists():
            print(f"Error: Validation JSON file not found: {val_json_path}")
            return
        
        # Run train/val split validation
        report = validate_train_val_split(
            train_json_path=train_json_path,
            val_json_path=val_json_path,
            num_visualize=args.num_visualize,
            expected_val_ratio=args.expected_val_ratio,
            certify_golden=args.certify_golden
        )
        
        # Print summary
        print_split_validation_summary(report)
        
        # Print detailed certification if applicable
        if args.certify_golden and report.get("certification"):
            cert = report["certification"]
            print("\n" + "="*80)
            print("GOLDEN DATASET CERTIFICATION DETAILS")
            print("="*80)
            if cert["is_golden"]:
                print("✓ CERTIFIED AS GOLDEN DATASET")
            else:
                print("✗ DOES NOT MEET GOLDEN DATASET CRITERIA")
                print("\nCriteria Status:")
                for criterion, passed in cert["criteria"].items():
                    status = "✓" if passed else "✗"
                    print(f"  {status} {criterion}")
                print("\nIssues:")
                for issue in cert["issues"]:
                    print(f"  - {issue}")
                print("\nRecommendations:")
                for rec in cert["recommendations"]:
                    print(f"  - {rec}")
            print("="*80)
        
        # Save comprehensive report
        report_path = train_json_path.parent / f"{train_json_path.stem}_split_validation_report.json"
        print(f"\nSaving split validation report to: {report_path}")
        with open(report_path, 'w') as f:
            ujson.dump(report, f, indent=2)
        
        print(f"\nSplit validation complete! Report saved to: {report_path}")
    
    else:
        # Standard single-file validation
        if not args.json_path:
            parser.error("--json_path is required (or use --validate_split for train/val validation)")
        
        json_path = Path(args.json_path)
        if not json_path.exists():
            print(f"Error: JSON file not found: {json_path}")
            return
        
        image_folder = Path(args.image_folder) if args.image_folder else None
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        # Run validation
        report = validate_json_file(
            json_path=json_path,
            image_folder=image_folder,
            num_visualize=args.num_visualize,
            output_dir=output_dir
        )
        
        # Print console report
        print_console_report(report)
        
        # Save JSON report
        report_path = json_path.with_suffix('.validation_report.json')
        print(f"\nSaving detailed report to: {report_path}")
        with open(report_path, 'w') as f:
            ujson.dump(report, f, indent=2)
        
        print(f"\nValidation complete! Report saved to: {report_path}")


if __name__ == "__main__":
    main()

