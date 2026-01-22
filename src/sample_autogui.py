#!/usr/bin/env python3
"""
AutoGUI 2k Sampling Helper Script

Samples N AutoGUI grounding examples from Hugging Face with stratified sampling,
outputs SupervisedDataset-compatible JSON files for UI-TARS1.5-7B training.
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import uuid

from datasets import load_dataset
from PIL import Image
import ujson

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

# Generic element text patterns for high-ambiguity detection
GENERIC_ELEM_TEXTS = {"button", "link", "icon", "image", "text", "element", "item", "option"}


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
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


def prepare_training_coordinates(original_x: float, original_y: float, 
                                 original_width: int, original_height: int) -> Tuple[int, int]:
    """
    Convert original image coordinates to smart-resized space for training.
    Args:
        original_x, original_y: Click position in original image
        original_width, original_height: Original image dimensions
    Returns:
        training_x, training_y: Coordinates in smart-resized space
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


def parse_image_size(image_size: str) -> Optional[Tuple[int, int]]:
    """
    Parse image_size string to extract width and height.
    Expected formats: "WxH", "W H", or similar.
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


def compute_center_from_box(unnormalized_box: List[float]) -> Tuple[float, float]:
    """
    Compute center point from unnormalized_box [l, t, r, b].
    Args:
        unnormalized_box: [left, top, right, bottom] coordinates
    Returns:
        (center_x, center_y): Center point coordinates
    """
    if len(unnormalized_box) != 4:
        raise ValueError(f"Expected 4-element box, got {len(unnormalized_box)}")
    
    l, t, r, b = unnormalized_box
    center_x = (l + r) / 2.0
    center_y = (t + b) / 2.0
    return center_x, center_y


def normalize_coordinates(x: float, y: float, image_width: int, image_height: int) -> Tuple[int, int]:
    """
    Apply smart_resize normalization to coordinates.
    Args:
        x, y: Original coordinates
        image_width, image_height: Original image dimensions
    Returns:
        Normalized (x, y) coordinates
    """
    return prepare_training_coordinates(x, y, image_width, image_height)


def load_autogui_streaming():
    """Load AutoGUI dataset from Hugging Face with streaming."""
    print("Loading AutoGUI dataset from Hugging Face (streaming mode)...")
    dataset = load_dataset("AutoGUI/AutoGUI-v1-702k", streaming=True, split="train")
    return dataset


def is_grounding_sample(sample: Dict[str, Any]) -> bool:
    """
    Check if sample has valid grounding data.
    Returns True if unnormalized_box exists and answer represents a coordinate.
    """
    # Check for unnormalized_box
    if "unnormalized_box" not in sample or sample["unnormalized_box"] is None:
        return False
    
    box = sample["unnormalized_box"]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    
    # Validate box coordinates are numeric
    try:
        [float(x) for x in box]
    except (ValueError, TypeError):
        return False
    
    # Check if answer represents a coordinate (not a caption)
    # For grounding tasks, answer should be a coordinate format
    # We'll check if it's numeric or coordinate-like
    if "answer" not in sample:
        return False
    
    answer = sample["answer"]
    if answer is None:
        return False
    
    # If answer is a string, check if it looks like a coordinate
    if isinstance(answer, str):
        # Check for coordinate patterns like "(x, y)" or "x, y" or just numbers
        coord_pattern = r'\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?'
        if re.match(coord_pattern, answer.strip()):
            return True
        # If it's a long text description, it's likely a caption
        if len(answer) > 50:
            return False
    
    # If answer is numeric or list/tuple of numbers, it's likely a coordinate
    if isinstance(answer, (int, float)):
        return True
    if isinstance(answer, (list, tuple)) and len(answer) >= 2:
        try:
            [float(x) for x in answer[:2]]
            return True
        except (ValueError, TypeError):
            pass
    
    return True  # Default: assume it's grounding if box exists


def extract_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from sample: device, elem_role, elem_text, box_area_ratio.
    """
    image = sample.get("image")
    image_size_str = sample.get("image_size", "")
    
    # If image is PIL.Image, extract dimensions directly
    if isinstance(image, Image.Image):
        img_width, img_height = image.size
        image_size_str = f"{img_width}x{img_height}"
    
    # Require image_id - fail if not present
    image_id = sample.get("image_id") or sample.get("id")
    if image_id is None:
        raise ValueError(f"Sample missing required 'image_id' or 'id' field. Sample keys: {list(sample.keys())}")
    
    metadata = {
        "device": sample.get("device", "unknown"),
        "elem_role": sample.get("elem_role", ""),
        "elem_text": sample.get("elem_text", ""),
        "instruction": sample.get("instruction", ""),
        "unnormalized_box": sample.get("unnormalized_box"),
        "image_size": image_size_str,
        "image": image,
        "answer": sample.get("answer"),
        "data_source": sample.get("data_source", sample.get("source", "unknown")),  # Track source file/domain
        "image_id": image_id,  # Required field from AutoGUI dataset
    }
    
    # Compute box area ratio
    box = sample.get("unnormalized_box")
    
    box_area_ratio = None
    if box and len(box) == 4:
        try:
            l, t, r, b = [float(x) for x in box]
            box_area = (r - l) * (b - t)
            
            # Try to get image dimensions
            dims = None
            if image_size_str:
                dims = parse_image_size(image_size_str)
            elif isinstance(image, Image.Image):
                dims = image.size  # (width, height)
            
            if dims:
                img_width, img_height = dims
                img_area = img_width * img_height
                if img_area > 0:
                    box_area_ratio = box_area / img_area
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    
    metadata["box_area_ratio"] = box_area_ratio
    return metadata


def assign_bucket(metadata: Dict[str, Any], role_counts: Dict[str, int]) -> Optional[str]:
    """
    Assign sample to priority-ordered bucket.
    Priority: small > high_ambiguity > mobile > web
    """
    # 1. Small/subtle (150): Box area < 1% of image
    if metadata.get("box_area_ratio") is not None:
        if metadata["box_area_ratio"] < 0.01:  # < 1%
            return "small"
    
    # 2. High-ambiguity (250): Generic elem_text or repeated elem_role
    elem_text = metadata.get("elem_text", "").lower().strip()
    elem_role = metadata.get("elem_role", "").lower().strip()
    
    # Check for generic text
    if elem_text in GENERIC_ELEM_TEXTS or any(generic in elem_text for generic in GENERIC_ELEM_TEXTS):
        return "high_ambiguity"
    
    # Check for repeated role (appears many times in dataset)
    if elem_role and role_counts.get(elem_role, 0) > 100:
        return "high_ambiguity"
    
    # 3. Mobile (400): device == "mobile"
    if metadata.get("device", "").lower() == "mobile":
        return "mobile"
    
    # 4. Web (1200): device == "web" (catch-all)
    if metadata.get("device", "").lower() == "web":
        return "web"
    
    # Default: assign to web if device is unknown
    return "web"


def reservoir_sample(reservoir: List[Dict], sample: Dict, quota: int, 
                    items_seen: int, rng: random.Random) -> Tuple[List[Dict], int]:
    """
    Reservoir sampling: maintain uniform sample from stream.
    Args:
        reservoir: Current reservoir for this bucket
        sample: New sample to consider
        quota: Maximum size of reservoir
        items_seen: Number of items seen so far in this bucket
        rng: Random number generator
    Returns:
        (updated_reservoir, updated_items_seen)
    """
    items_seen += 1
    
    if len(reservoir) < quota:
        # Reservoir not full: add sample
        reservoir.append(sample)
    else:
        # Reservoir full: replace with probability k/i
        j = rng.randint(0, items_seen - 1)
        if j < quota:
            reservoir[j] = sample
    
    return reservoir, items_seen


def format_for_supervised_dataset(sample: Dict[str, Any], metadata: Dict[str, Any], 
                                  output_images_dir: Optional[Path] = None,
                                  images_relative_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert sample to UI-TARS SupervisedDataset JSON format.
    
    Args:
        sample: Original sample from AutoGUI dataset
        metadata: Extracted metadata
        output_images_dir: Directory to save images (required for PIL.Image objects)
        images_relative_path: Relative path from JSON file to images directory (for JSON references)
    """
    instruction = metadata.get("instruction", "")
    unnormalized_box = metadata.get("unnormalized_box")
    image_size_str = metadata.get("image_size", "")
    image = metadata.get("image")
    image_id = metadata.get("image_id")
    
    # Compute center point from box
    if not unnormalized_box or len(unnormalized_box) != 4:
        raise ValueError("Invalid unnormalized_box for formatting")
    
    center_x, center_y = compute_center_from_box(unnormalized_box)
    
    # Parse image dimensions
    dims = None
    if image_size_str:
        dims = parse_image_size(image_size_str)
    elif isinstance(image, Image.Image):
        dims = image.size  # (width, height)
    
    if not dims:
        raise ValueError(f"Could not determine image dimensions from image_size: {image_size_str} or image object")
    
    img_width, img_height = dims
    
    # Normalize coordinates
    norm_x, norm_y = normalize_coordinates(center_x, center_y, img_width, img_height)
    
    # Format user prompt
    user_prompt = f"<image>\n{UITARS_USR_PROMPT_NOTHOUGHT.format(instruction=instruction)}"
    
    # Format assistant response
    assistant_response = f"Action: click(start_box='({norm_x}, {norm_y})')"
    
    # Handle image saving and reference
    if isinstance(image, Image.Image):
        # Save PIL.Image to disk if output directory is provided
        if output_images_dir is not None:
            # Create filename from image_id
            image_filename = f"{image_id}.png"
            image_path = output_images_dir / image_filename
            image.save(image_path, "PNG")
            # Use relative path for SupervisedDataset compatibility
            if images_relative_path:
                image_ref = f"{images_relative_path}/{image_filename}"
            else:
                # Fallback: use just filename (assumes images_dir is set as image_folder in training)
                image_ref = image_filename
        else:
            raise ValueError("output_images_dir must be provided to save PIL.Image objects")
    elif isinstance(image, str):
        # Already a string (URL or path) - preserve as-is
        image_ref = image
    else:
        raise ValueError(f"Unexpected image type: {type(image)}")
    
    # Require image_id - fail if not present (should have been validated in extract_metadata)
    if image_id is None:
        raise ValueError("image_id is required but was None. This should have been caught in extract_metadata().")
    
    # Use the AutoGUI image_id directly (convert to string if needed)
    entry_id = str(image_id)
    
    # Create entry
    entry = {
        "id": entry_id,
        "image": [image_ref] if not isinstance(image_ref, list) else image_ref,
        "conversations": [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": assistant_response}
        ]
    }
    
    return entry


def compute_metadata_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate distribution statistics for metadata output.
    """
    device_counts = Counter()
    role_counts = Counter()
    source_counts = Counter()
    box_area_ratios = []
    bucket_counts = Counter()
    
    for sample in samples:
        metadata = sample.get("metadata", {})
        device_counts[metadata.get("device", "unknown")] += 1
        role_counts[metadata.get("elem_role", "")] += 1
        source_counts[metadata.get("data_source", "unknown")] += 1
        if metadata.get("box_area_ratio") is not None:
            box_area_ratios.append(metadata["box_area_ratio"])
        bucket_counts[sample.get("bucket", "unknown")] += 1
    
    stats = {
        "device_distribution": dict(device_counts),
        "source_distribution": dict(source_counts),
        "elem_role_distribution": dict(role_counts.most_common(20)),  # Top 20
        "bucket_counts": dict(bucket_counts),
    }
    
    if box_area_ratios:
        stats["box_area_stats"] = {
            "min": min(box_area_ratios),
            "max": max(box_area_ratios),
            "mean": sum(box_area_ratios) / len(box_area_ratios),
        }
    else:
        stats["box_area_stats"] = {}
    
    return stats


def log_subset_composition(samples: List[Dict[str, Any]], quotas: Dict[str, int]) -> None:
    """
    Log a detailed summary of the subset data composition.
    
    Args:
        samples: List of sampled data with metadata
        quotas: Dictionary of bucket quotas
    """
    if not samples:
        print("\n" + "="*80)
        print("SUBSET COMPOSITION SUMMARY")
        print("="*80)
        print("No samples collected.")
        return
    
    total_samples = len(samples)
    
    # Collect statistics
    device_counts = Counter()
    role_counts = Counter()
    bucket_counts = Counter()
    source_counts = Counter()
    box_area_ratios = []
    elem_text_lengths = []
    
    for sample in samples:
        metadata = sample.get("metadata", {})
        device = metadata.get("device", "unknown")
        role = metadata.get("elem_role", "")
        bucket = sample.get("bucket", "unknown")
        source = metadata.get("data_source", "unknown")
        
        device_counts[device] += 1
        if role:
            role_counts[role] += 1
        
        bucket_counts[bucket] += 1
        source_counts[source] += 1
        
        box_ratio = metadata.get("box_area_ratio")
        if box_ratio is not None:
            box_area_ratios.append(box_ratio)
        
        elem_text = metadata.get("elem_text", "")
        if elem_text:
            elem_text_lengths.append(len(elem_text))
    
    # Print summary
    print("\n" + "="*80)
    print("SUBSET COMPOSITION SUMMARY")
    print("="*80)
    print(f"Total samples: {total_samples}")
    print()
    
    # Bucket distribution
    print("Bucket Distribution:")
    print("-" * 80)
    for bucket in sorted(quotas.keys()):
        count = bucket_counts.get(bucket, 0)
        quota = quotas[bucket]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        quota_percentage = (count / quota * 100) if quota > 0 else 0
        status = "✓" if count >= quota else "⚠"
        print(f"  {status} {bucket:20s}: {count:4d}/{quota:4d} ({percentage:5.1f}% of total, {quota_percentage:5.1f}% of quota)")
    print()
    
    # Device distribution
    print("Device Distribution:")
    print("-" * 80)
    for device, count in device_counts.most_common():
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {device:20s}: {count:4d} ({percentage:5.1f}%)")
    print()
    
    # Source distribution
    print("Source Distribution (AutoGUI source files/domains):")
    print("-" * 80)
    for source, count in source_counts.most_common():
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {source:30s}: {count:4d} ({percentage:5.1f}%)")
    print()
    
    # Top element roles
    print("Top Element Roles (Top 15):")
    print("-" * 80)
    for role, count in role_counts.most_common(15):
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {role:30s}: {count:4d} ({percentage:5.1f}%)")
    print()
    
    # Box area statistics
    if box_area_ratios:
        print("Box Area Statistics (relative to image):")
        print("-" * 80)
        print(f"  Min:    {min(box_area_ratios):.6f} ({min(box_area_ratios)*100:.4f}%)")
        print(f"  Max:    {max(box_area_ratios):.6f} ({max(box_area_ratios)*100:.4f}%)")
        print(f"  Mean:   {sum(box_area_ratios)/len(box_area_ratios):.6f} ({sum(box_area_ratios)/len(box_area_ratios)*100:.4f}%)")
        
        # Count small elements (< 1%)
        small_count = sum(1 for r in box_area_ratios if r < 0.01)
        print(f"  Small (<1%): {small_count:4d} ({small_count/len(box_area_ratios)*100:.1f}%)")
        print()
    
    # Element text statistics
    if elem_text_lengths:
        print("Element Text Length Statistics:")
        print("-" * 80)
        print(f"  Min:    {min(elem_text_lengths):3d} characters")
        print(f"  Max:    {max(elem_text_lengths):3d} characters")
        print(f"  Mean:   {sum(elem_text_lengths)/len(elem_text_lengths):.1f} characters")
        print()
    
    # Sampling quality indicators
    print("Sampling Quality:")
    print("-" * 80)
    all_quotas_met = all(bucket_counts.get(b, 0) >= quotas[b] for b in quotas.keys())
    quota_status = "✓ All quotas met" if all_quotas_met else "⚠ Some quotas not met"
    print(f"  Quota fulfillment: {quota_status}")
    
    if device_counts:
        device_diversity = len(device_counts)
        print(f"  Device diversity: {device_diversity} device type(s)")
    
    if role_counts:
        role_diversity = len(role_counts)
        print(f"  Role diversity: {role_diversity} unique element role(s)")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Sample AutoGUI grounding examples with stratified sampling"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/autogui_2k.json",
        help="Output JSON file path (default: data/autogui_2k.json)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of samples to generate (default: 2000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    rng = random.Random(args.seed)
    
    # Define bucket quotas
    quotas = {
        "small": 150,
        "high_ambiguity": 250,
        "mobile": 400,
        "web": 1200,
    }
    total_quota = sum(quotas.values())
    
    if args.num_samples != total_quota:
        print(f"Warning: num_samples ({args.num_samples}) doesn't match default quota total ({total_quota})")
        print(f"Using quotas: {quotas}")
    
    # Initialize reservoirs and counters
    reservoirs = {bucket: [] for bucket in quotas.keys()}
    items_seen = {bucket: 0 for bucket in quotas.keys()}
    role_counts = defaultdict(int)  # Track role frequencies for high-ambiguity detection
    
    # Load dataset
    dataset = load_autogui_streaming()
    
    print(f"Starting sampling with quotas: {quotas}")
    print("Processing samples...")
    
    samples_processed = 0
    samples_filtered = 0
    
    # First pass: collect role counts for high-ambiguity detection
    print("First pass: collecting role statistics...")
    role_sample_count = 0
    # #region agent log
    import json
    log_path = "/Users/lockewang/FIG/Qwen2-VL-Finetune/.cursor/debug.log"
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "sample_autogui.py:636", "message": "Starting first pass role collection", "data": {"max_samples": 10000}, "timestamp": __import__("time").time() * 1000}) + "\n")
    # #endregion
    for sample in dataset:
        if role_sample_count >= 10000:  # Sample first 10k for role statistics
            break
        if is_grounding_sample(sample):
            metadata = extract_metadata(sample)
            role = metadata.get("elem_role", "").lower().strip()
            if role:
                role_counts[role] += 1
                # #region agent log
                if role_sample_count < 5:  # Log first 5 for debugging
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "sample_autogui.py:645", "message": "Found role in sample", "data": {"role": role, "role_counts_size": len(role_counts), "sample_num": role_sample_count}, "timestamp": __import__("time").time() * 1000}) + "\n")
                # #endregion
        role_sample_count += 1
    
    print(f"Collected role statistics from {role_sample_count} samples")
    print(f"Top roles: {dict(Counter(role_counts).most_common(10))}")
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "sample_autogui.py:653", "message": "First pass complete", "data": {"total_roles": len(role_counts), "top_roles": dict(Counter(role_counts).most_common(5))}, "timestamp": __import__("time").time() * 1000}) + "\n")
    # #endregion
    
    # Main sampling loop
    print("\nSecond pass: stratified sampling...")
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "D", "location": "sample_autogui.py:655", "message": "Starting second pass", "data": {"num_samples": args.num_samples, "total_quota": total_quota, "quotas": quotas}, "timestamp": __import__("time").time() * 1000}) + "\n")
    # #endregion
    for sample in dataset:
        # Check if all quotas are filled OR if we have enough total samples
        all_full = all(len(reservoirs[bucket]) >= quotas[bucket] for bucket in quotas.keys())
        total_collected = sum(len(reservoirs[b]) for b in quotas.keys())
        # #region agent log
        if samples_processed % 50000 == 0:  # Log every 50k samples
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "D", "location": "sample_autogui.py:660", "message": "Early termination check", "data": {"all_full": all_full, "total_collected": total_collected, "num_samples": args.num_samples, "bucket_counts": {b: len(reservoirs[b]) for b in quotas.keys()}}, "timestamp": __import__("time").time() * 1000}) + "\n")
        # #endregion
        if all_full or (args.num_samples < total_quota and total_collected >= args.num_samples):
            print(f"All quotas filled! Stopping early. (Total collected: {total_collected})")
            break
        
        samples_processed += 1
        if samples_processed % 10000 == 0:
            filled = {b: len(reservoirs[b]) for b in quotas.keys()}
            print(f"Processed {samples_processed} samples, filled: {filled}")
        
        # Filter for grounding samples
        if not is_grounding_sample(sample):
            samples_filtered += 1
            continue
        
        # Extract metadata
        try:
            metadata = extract_metadata(sample)
        except Exception as e:
            print(f"Warning: Failed to extract metadata: {e}")
            samples_filtered += 1
            continue
        
        # Assign to bucket
        bucket = assign_bucket(metadata, role_counts)
        # #region agent log
        if samples_processed < 100 or (samples_processed % 10000 == 0 and samples_processed < 100000):  # Log early samples and periodic checks
            elem_text = metadata.get("elem_text", "").lower().strip()
            elem_role = metadata.get("elem_role", "").lower().strip()
            box_ratio = metadata.get("box_area_ratio")
            device = metadata.get("device", "").lower()
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "sample_autogui.py:682", "message": "Bucket assignment", "data": {"bucket": bucket, "elem_text": elem_text[:50], "elem_role": elem_role, "box_ratio": box_ratio, "device": device, "role_count": role_counts.get(elem_role, 0), "is_generic_text": elem_text in GENERIC_ELEM_TEXTS or any(g in elem_text for g in GENERIC_ELEM_TEXTS)}, "timestamp": __import__("time").time() * 1000}) + "\n")
        # #endregion
        if bucket is None:
            samples_filtered += 1
            continue
        
        # Check if bucket is full
        if len(reservoirs[bucket]) >= quotas[bucket]:
            items_seen[bucket] += 1
            continue
        
        # Apply reservoir sampling
        reservoirs[bucket], items_seen[bucket] = reservoir_sample(
            reservoirs[bucket],
            {"sample": sample, "metadata": metadata, "bucket": bucket},
            quotas[bucket],
            items_seen[bucket],
            rng
        )
    
    print(f"\nSampling complete!")
    print(f"Total samples processed: {samples_processed}")
    print(f"Samples filtered out: {samples_filtered}")
    print(f"\nFinal bucket counts:")
    for bucket in quotas.keys():
        count = len(reservoirs[bucket])
        quota = quotas[bucket]
        print(f"  {bucket}: {count}/{quota} ({count/quota*100:.1f}%)")
    
    # Merge all samples
    all_samples = []
    for bucket in quotas.keys():
        all_samples.extend(reservoirs[bucket])
    
    if len(all_samples) < args.num_samples:
        print(f"Warning: Only collected {len(all_samples)} samples, requested {args.num_samples}")
    
    # Log subset composition summary
    log_subset_composition(all_samples, quotas)
    
    # Shuffle merged samples
    rng.shuffle(all_samples)
    
    # Prepare output paths
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create images directory alongside JSON file
    images_dir_name = f"{output_path.stem}_images"
    images_dir = output_path.parent / images_dir_name
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"Images will be saved to: {images_dir}")
    
    # Format samples for SupervisedDataset and save images
    print("\nFormatting samples and saving images...")
    formatted_entries = []
    format_errors = 0
    
    for sample_data in all_samples:
        try:
            entry = format_for_supervised_dataset(
                sample_data["sample"],
                sample_data["metadata"],
                output_images_dir=images_dir,
                images_relative_path=images_dir_name  # Relative path from JSON location
            )
            formatted_entries.append(entry)
        except Exception as e:
            format_errors += 1
            print(f"Warning: Failed to format/save sample: {e}")
            continue
    
    print(f"Formatted {len(formatted_entries)} entries ({format_errors} errors)")
    
    print(f"\nWriting output to {output_path}...")
    with open(output_path, 'w') as f:
        ujson.dump(formatted_entries, f, indent=2)
    
    print(f"Wrote {len(formatted_entries)} entries to {output_path}")
    print(f"Saved {len(formatted_entries)} images to {images_dir}")
    
    # Compute and write metadata
    meta_path = output_path.with_suffix('.meta.json')
    metadata_stats = compute_metadata_stats(all_samples)
    metadata_output = {
        "seed": args.seed,
        "num_samples": len(formatted_entries),
        "requested_samples": args.num_samples,
        "source": "AutoGUI HF streaming",
        "images_dir": str(images_dir.relative_to(output_path.parent)) if images_dir.exists() else None,
        **metadata_stats
    }
    
    print(f"Writing metadata to {meta_path}...")
    with open(meta_path, 'w') as f:
        ujson.dump(metadata_output, f, indent=2)
    
    print(f"\nDone! Output files:")
    print(f"  - {output_path}")
    print(f"  - {meta_path}")
    print(f"  - {images_dir}/ ({len(formatted_entries)} images)")


if __name__ == "__main__":
    main()

