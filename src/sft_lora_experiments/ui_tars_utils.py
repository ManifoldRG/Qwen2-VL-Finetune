#!/usr/bin/env python3
"""
Shared UI-TARS utilities for dataset conversion.

This module provides constants and functions used by both AutoGUI and Salesforce
grounding dataset conversion scripts. It serves as the single source of truth
for UI-TARS data format requirements.
"""

import math
from typing import List, Optional, Tuple


# Constants from prepare_mind2web_data.py
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

UITARS_USR_PROMPT_NOTHOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
## Output Format
```
Action: ...
```
## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
## User Instruction
{instruction}"""


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS
) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    
    Args:
        height: Original image height
        width: Original image width
        factor: Factor for dimension rounding (default: IMAGE_FACTOR)
        min_pixels: Minimum total pixels (default: MIN_PIXELS)
        max_pixels: Maximum total pixels (default: MAX_PIXELS)
    
    Returns:
        (new_height, new_width): Resized dimensions
    
    Raises:
        ValueError: If aspect ratio exceeds MAX_RATIO
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def prepare_training_coordinates(
    original_x: float,
    original_y: float,
    original_width: int,
    original_height: int
) -> Tuple[int, int]:
    """
    Convert original image coordinates to smart-resized space for training.
    
    Args:
        original_x: X coordinate in original image
        original_y: Y coordinate in original image
        original_width: Original image width
        original_height: Original image height
    
    Returns:
        (training_x, training_y): Coordinates in smart-resized space
    """
    # Get smart-resized dimensions
    smart_h, smart_w = smart_resize(
        height=original_height,
        width=original_width,
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS
    )

    # Scale coordinates
    training_x = int(original_x * smart_w / original_width)
    training_y = int(original_y * smart_h / original_height)

    return (training_x, training_y)


def compute_center_from_box(unnormalized_box: List[float]) -> Tuple[float, float]:
    """
    Compute center point from unnormalized_box [l, t, r, b].
    
    Args:
        unnormalized_box: [left, top, right, bottom] coordinates
    
    Returns:
        (center_x, center_y): Center point coordinates
    
    Raises:
        ValueError: If box doesn't have exactly 4 elements
    """
    if len(unnormalized_box) != 4:
        raise ValueError(f"Expected 4-element box, got {len(unnormalized_box)}")
    
    l, t, r, b = unnormalized_box
    center_x = (l + r) / 2.0
    center_y = (t + b) / 2.0
    return center_x, center_y


def convert_bbox_to_center(bbox: List[int]) -> Tuple[float, float]:
    """
    Convert [x1, y1, x2, y2] bounding box to center point (x, y).
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
    
    Returns:
        (center_x, center_y): Center point coordinates
    
    Raises:
        ValueError: If bbox doesn't have exactly 4 elements
    """
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox format: expected 4 elements, got {len(bbox)}")
    
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return center_x, center_y


def normalize_coordinates(
    x: float,
    y: float,
    image_width: int,
    image_height: int
) -> Tuple[int, int]:
    """
    Apply smart_resize normalization to coordinates.
    
    Args:
        x: Original x coordinate
        y: Original y coordinate
        image_width: Original image width
        image_height: Original image height
    
    Returns:
        Normalized (x, y) coordinates
    """
    return prepare_training_coordinates(x, y, image_width, image_height)


def parse_image_size(image_size: str) -> Optional[Tuple[int, int]]:
    """
    Parse image_size string to extract width and height.
    Expected formats: "WxH", "W H", or similar.
    
    Args:
        image_size: String representation of image size
    
    Returns:
        (width, height) tuple if parsing succeeds, None otherwise
    """
    if not image_size:
        return None
    
    # Try "WxH" format
    if 'x' in image_size.lower():
        parts = image_size.lower().split('x')
        if len(parts) == 2:
            try:
                return (int(parts[0].strip()), int(parts[1].strip()))
            except ValueError:
                pass
    
    # Try space-separated
    parts = image_size.split()
    if len(parts) >= 2:
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            pass
    
    return None

