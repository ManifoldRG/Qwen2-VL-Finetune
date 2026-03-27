"""
Geometric distance metrics for GUI grounding tasks.

Parses coordinates from UI-TARS action text and computes Euclidean distance
between predicted and ground truth coordinates.
"""

import ast
import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple

# Environment variable to enable verbose geometric metrics debugging
DEBUG_GEOMETRIC_METRICS = os.getenv("DEBUG_GEOMETRIC_METRICS", "false").lower() == "true"

logger = logging.getLogger(__name__)


def parse_coordinates_from_action_text(text: str) -> Optional[Tuple[float, float]]:
    """
    Extract coordinates from UI-TARS action text.
    
    Examples:
        "Action: click(start_box='(739,336)')" -> (739.0, 336.0)
        "Action: click(start_box='<|box_start|>(100,200)<|box_end|>')" -> (100.0, 200.0)
    
    Returns:
        (x, y) tuple if found, None otherwise
    """
    if DEBUG_GEOMETRIC_METRICS:
        logger.debug(f"Parsing coordinates from text: {text[:200]}..." if len(text) > 200 else f"Parsing coordinates from text: {text}")
    
    # Pattern to match start_box='(x, y)' or start_box="(x, y)"
    pattern = r"start_box=['\"]([^'\"]+)['\"]"
    matches = re.findall(pattern, text)
    
    if DEBUG_GEOMETRIC_METRICS:
        logger.debug(f"  Regex matches found: {len(matches)}, matches: {matches}")
    
    if not matches:
        if DEBUG_GEOMETRIC_METRICS:
            logger.debug("  No regex matches found, returning None")
        return None
    
    # Take first match (for click actions, there's only one)
    coords_str = matches[0].strip()
    
    if DEBUG_GEOMETRIC_METRICS:
        logger.debug(f"  Extracted coords string: {coords_str}")
    
    # Remove box markers if present
    coords_str = coords_str.replace("<|box_start|>", "").replace("<|box_end|>", "")
    coords_str = coords_str.strip()
    
    if DEBUG_GEOMETRIC_METRICS:
        logger.debug(f"  After removing box markers: {coords_str}")
    
    try:
        # Try parsing as tuple/list
        parsed = ast.literal_eval(coords_str)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            x, y = float(parsed[0]), float(parsed[1])
            if DEBUG_GEOMETRIC_METRICS:
                logger.debug(f"  Successfully parsed as tuple/list: ({x}, {y})")
            return (x, y)
    except (ValueError, SyntaxError) as e:
        if DEBUG_GEOMETRIC_METRICS:
            logger.debug(f"  Failed to parse as tuple/list: {e}")
        pass
    
    # Try parsing as "(x,y)" format
    if coords_str.startswith("(") and coords_str.endswith(")"):
        coords_str = coords_str[1:-1]
        parts = coords_str.split(",")
        if len(parts) >= 2:
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                if DEBUG_GEOMETRIC_METRICS:
                    logger.debug(f"  Successfully parsed as (x,y) format: ({x}, {y})")
                return (x, y)
            except ValueError as e:
                if DEBUG_GEOMETRIC_METRICS:
                    logger.debug(f"  Failed to parse as (x,y) format: {e}")
                pass
    
    if DEBUG_GEOMETRIC_METRICS:
        logger.warning(f"  Failed to parse coordinates from text, returning None")
    
    return None


def compute_geometric_distance_metrics(
    predictions: List[str], 
    references: List[str]
) -> Dict[str, float]:
    """
    Compute geometric distance metrics between predictions and references.
    
    Args:
        predictions: List of prediction strings (action text)
        references: List of reference strings (ground truth action text)
    
    Returns:
        Dictionary with metrics:
        - mean_distance: average Euclidean distance
        - median_distance: median Euclidean distance
        - std_distance: standard deviation of distances
        - min_distance: minimum distance observed
        - max_distance: maximum distance observed
        - valid_samples: count of valid coordinate pairs
        - total_samples: total number of prediction/reference pairs
    """
    distances = []
    
    # Validate list lengths match
    if len(predictions) != len(references):
        logger.warning(f"predictions and references have different lengths: {len(predictions)} vs {len(references)}. Using minimum length.")
    total_samples = min(len(predictions), len(references))
    
    if DEBUG_GEOMETRIC_METRICS:
        logger.debug(f"Computing geometric distance metrics for {total_samples} prediction/reference pairs")
    
    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        if DEBUG_GEOMETRIC_METRICS and idx < 5:  # Log first 5 pairs in detail
            logger.debug(f"  Processing pair {idx+1}/{total_samples}")
            logger.debug(f"    Prediction: {pred[:150]}..." if len(pred) > 150 else f"    Prediction: {pred}")
            logger.debug(f"    Reference: {ref[:150]}..." if len(ref) > 150 else f"    Reference: {ref}")
        
        pred_coords = parse_coordinates_from_action_text(pred)
        ref_coords = parse_coordinates_from_action_text(ref)
        
        if DEBUG_GEOMETRIC_METRICS and idx < 5:
            logger.debug(f"    Predicted coords: {pred_coords}, Reference coords: {ref_coords}")
        
        if pred_coords is None or ref_coords is None:
            if DEBUG_GEOMETRIC_METRICS and idx < 5:
                logger.warning(f"    Skipping pair {idx+1}: missing coordinates (pred={pred_coords is not None}, ref={ref_coords is not None})")
            continue
        
        # Calculate Euclidean distance
        x_pred, y_pred = pred_coords
        x_ref, y_ref = ref_coords
        distance = math.sqrt((x_pred - x_ref) ** 2 + (y_pred - y_ref) ** 2)
        distances.append(distance)
        
        if DEBUG_GEOMETRIC_METRICS and idx < 5:
            logger.debug(f"    Computed distance: {distance:.2f}")
    
    valid_samples = len(distances)
    
    if DEBUG_GEOMETRIC_METRICS:
        logger.debug(f"  Valid coordinate pairs: {valid_samples}/{total_samples}")
    
    if valid_samples == 0:
        if DEBUG_GEOMETRIC_METRICS:
            logger.warning("  No valid coordinate pairs found, returning zero metrics")
        return {
            "mean_distance": 0.0,
            "median_distance": 0.0,
            "std_distance": 0.0,
            "min_distance": 0.0,
            "max_distance": 0.0,
            "valid_samples": 0,
            "total_samples": total_samples,
        }
    
    distances.sort()
    mean_distance = sum(distances) / valid_samples
    median_distance = distances[valid_samples // 2] if valid_samples > 0 else 0.0
    
    # Standard deviation
    variance = sum((d - mean_distance) ** 2 for d in distances) / valid_samples
    std_distance = math.sqrt(variance)
    
    result = {
        "mean_distance": mean_distance,
        "median_distance": median_distance,
        "std_distance": std_distance,
        "min_distance": distances[0],
        "max_distance": distances[-1],
        "valid_samples": valid_samples,
        "total_samples": total_samples,
    }
    
    if DEBUG_GEOMETRIC_METRICS:
        logger.debug(f"  Final metrics: mean={mean_distance:.2f}, median={median_distance:.2f}, "
                    f"std={std_distance:.2f}, min={distances[0]:.2f}, max={distances[-1]:.2f}")
    
    return result

