#!/usr/bin/env python3
"""
Fix existing ScreenSpot-v2 results JSON by re-evaluating correctness
with the correct bbox format interpretation.

The bbox format in ScreenSpot-v2 is [x1, y1, width, height], not [x1, y1, x2, y2].
This script re-evaluates all positive samples and recalculates metrics.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add parent directory to path to import evaluation functions
script_dir = Path(__file__).parent
eval_dir = script_dir.parent
sys.path.insert(0, str(eval_dir))

from eval_screenspot_pro import (
    eval_sample_positive_gt,
    evaluate
)


def fix_positive_sample(sample, img_base_path, dataset_path):
    """
    Re-evaluate a positive sample with correct bbox format.
    
    Args:
        sample: Sample dictionary from results
        img_base_path: Base path to images
        dataset_path: Path to dataset folder (for format detection)
    
    Returns:
        Updated sample with corrected correctness
    """
    if sample.get("gt_type") != "positive":
        return sample
    
    if "bbox" not in sample or sample["bbox"] is None:
        return sample
    
    if "pred" not in sample or sample["pred"] is None:
        return sample
    
    # Load image to get dimensions
    img_path = sample.get("img_path")
    if not img_path or not os.path.exists(img_path):
        # Try to construct path from img_base_path
        img_filename = sample.get("img_path", "").split("/")[-1]
        img_path = os.path.join(img_base_path, img_filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found for {sample.get('id')}, skipping")
            return sample
    
    try:
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_size = img.size  # (width, height)
    except Exception as e:
        print(f"Warning: Failed to load image {img_path}: {e}, skipping")
        return sample
    
    # Create response dict with normalized point
    pred_pixel = sample["pred"]
    if pred_pixel is None or len(pred_pixel) < 2:
        return sample
    
    # Normalize prediction to [0, 1]
    point_normalized = [pred_pixel[0] / img_size[0], pred_pixel[1] / img_size[1]]
    response = {"point": point_normalized}
    
    # Re-evaluate with correct bbox format
    sample_with_size = {**sample, "img_size": img_size}
    try:
        correctness = eval_sample_positive_gt(sample_with_size, response, dataset_path=dataset_path)
        sample["correctness"] = correctness
    except Exception as e:
        print(f"Warning: Failed to evaluate {sample.get('id')}: {e}")
        sample["correctness"] = "error"
        sample["error_code"] = "evaluation_error"
        sample["error"] = type(e).__name__
    
    return sample


def fix_results(input_file, output_file, img_base_path, dataset_path):
    """
    Fix results JSON file by re-evaluating all samples.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        img_base_path: Base path to images
        dataset_path: Path to dataset folder (for format detection)
    """
    print(f"Loading results from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    details = data.get("details", [])
    print(f"Found {len(details)} samples to process")
    
    # Fix positive samples
    print("Re-evaluating positive samples with correct bbox format...")
    fixed_details = []
    positive_count = 0
    correctness_changes = {"correct_to_wrong": 0, "wrong_to_correct": 0, "unchanged": 0}
    for sample in tqdm(details):
        if sample.get("gt_type") == "positive":
            original_correctness = sample.get("correctness")
            fixed_sample = fix_positive_sample(sample, img_base_path, dataset_path)
            new_correctness = fixed_sample.get("correctness")
            if original_correctness != new_correctness:
                if original_correctness == "correct" and new_correctness == "wrong":
                    correctness_changes["correct_to_wrong"] += 1
                elif original_correctness == "wrong" and new_correctness == "correct":
                    correctness_changes["wrong_to_correct"] += 1
            else:
                correctness_changes["unchanged"] += 1
            positive_count += 1
            fixed_details.append(fixed_sample)
        else:
            # Negative samples don't need fixing (they use different evaluation)
            fixed_details.append(sample)
    
    # Recalculate all metrics
    print("Recalculating metrics...")
    result_report = evaluate(fixed_details)
    
    # Save fixed results
    print(f"Saving fixed results to {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result_report, f, indent=4)
    
    print(f"Done! Fixed results saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Total samples: {len(fixed_details)}")
    print(f"  Correct: {sum(1 for r in fixed_details if r.get('correctness') == 'correct')}")
    print(f"  Wrong: {sum(1 for r in fixed_details if r.get('correctness') == 'wrong')}")
    print(f"  Error: {sum(1 for r in fixed_details if r.get('correctness') == 'error')}")
    print(f"  Wrong format: {sum(1 for r in fixed_details if r.get('correctness') == 'wrong_format')}")


def main():
    parser = argparse.ArgumentParser(description="Fix ScreenSpot-v2 results with correct bbox format")
    parser.add_argument("--input", type=str, required=True, help="Input JSON results file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON results file")
    parser.add_argument("--img_base_path", type=str, required=True, help="Base path to images")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset folder (for format detection)")
    
    args = parser.parse_args()
    
    fix_results(args.input, args.output, args.img_base_path, args.dataset_path)


if __name__ == "__main__":
    main()

