#!/usr/bin/env python3
"""
Salesforce Grounding Dataset to UI-TARS Converter

Converts Salesforce/grounding_dataset from Hugging Face to UI-TARS1.5-7B 
SupervisedDataset format. The Salesforce dataset has better instruction format 
(action-focused) compared to AutoGUI's verbose descriptions.
"""

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

from PIL import Image
import ujson

# Import shared utilities
from src.sft_lora_experiments.ui_tars_utils import (
    prepare_training_coordinates,
    convert_bbox_to_center,
    UITARS_USR_PROMPT_NOTHOUGHT,
)
from src.sft_lora_experiments.dataset_loaders import load_grounding_dataset
from src.sft_lora_experiments.sampling_strategies import (
    create_sampling_strategy,
    SamplingStrategy,
)


def is_valid_grounding_sample(sample: Dict[str, Any]) -> bool:
    """
    Check if sample has valid grounding data.
    Returns True if bbox exists and is valid [x1, y1, x2, y2].
    
    This function delegates to get_filter_reason() to avoid code duplication.
    """
    return get_filter_reason(sample) is None


def get_filter_reason(sample: Dict[str, Any], dataset_filter: Optional[str] = None) -> Optional[str]:
    """
    Get the reason why a sample was filtered, or None if it's valid.
    
    Returns:
        Filter reason string, or None if sample is valid
    """
    # Check dataset filter first
    if dataset_filter and sample.get("dataset", "").lower() != dataset_filter.lower():
        return f"dataset_filter_mismatch (expected: {dataset_filter}, got: {sample.get('dataset', 'missing')})"
    
    # Check for bbox
    if "bbox" not in sample or sample["bbox"] is None:
        return "missing_bbox"
    
    bbox = sample["bbox"]
    if not isinstance(bbox, (list, tuple)):
        return f"bbox_wrong_type (got: {type(bbox).__name__})"
    
    if len(bbox) != 4:
        return f"bbox_wrong_length (got: {len(bbox)}, expected: 4)"
    
    # Validate bbox coordinates are numeric and valid
    try:
        x1, y1, x2, y2 = [float(x) for x in bbox]
        # Check that x2 > x1 and y2 > y1 (valid bounding box)
        if x2 <= x1:
            return f"bbox_invalid_x (x2={x2} <= x1={x1})"
        if y2 <= y1:
            return f"bbox_invalid_y (y2={y2} <= y1={y1})"
    except (ValueError, TypeError) as e:
        return f"bbox_invalid_coords (error: {type(e).__name__})"
    
    # Check for required fields
    if "instruction" not in sample or not sample["instruction"]:
        return "missing_or_empty_instruction"
    
    if "image" not in sample or sample["image"] is None:
        return "missing_image"
    
    if "uuid" not in sample:
        return "missing_uuid"
    
    return None  # Sample is valid


def extract_metadata_grounding(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from Salesforce grounding dataset sample.
    
    WARNING: Accessing sample["image"] in streaming mode triggers a download of that image.
    This is necessary to get image dimensions for coordinate normalization, but will cause
    images to be downloaded on-demand and cached in ~/.cache/huggingface/.
    
    Args:
        sample: Sample from Salesforce/grounding_dataset
    Returns:
        Metadata dict with instruction, bbox, image info, etc.
    Raises:
        ValueError: If required fields (uuid) are missing
    """
    image = sample.get("image")  # This triggers download in streaming mode
    bbox = sample.get("bbox")
    uuid = sample.get("uuid")
    
    # Require uuid - fail if not present
    if uuid is None:
        raise ValueError(f"Sample missing required 'uuid' field. Sample keys: {list(sample.keys())}")
    
    # Extract image dimensions
    image_size_str = ""
    if isinstance(image, Image.Image):
        img_width, img_height = image.size
        image_size_str = f"{img_width}x{img_height}"
    
    metadata = {
        "instruction": sample.get("instruction", ""),
        "bbox": bbox,
        "image_size": image_size_str,
        "image": image,
        "uuid": uuid,
        "dataset": sample.get("dataset", "unknown"),  # Source dataset (ariaui, omniact, etc.)
        "description": sample.get("description", ""),
        "function": sample.get("function", ""),
    }
    
    # Compute center point from bbox
    if bbox and len(bbox) == 4:
        try:
            center_x, center_y = convert_bbox_to_center(bbox)
            metadata["center_x"] = center_x
            metadata["center_y"] = center_y
            
            # Compute box area ratio for statistics
            x1, y1, x2, y2 = bbox
            box_area = (x2 - x1) * (y2 - y1)
            
            if isinstance(image, Image.Image):
                img_width, img_height = image.size
                img_area = img_width * img_height
                if img_area > 0:
                    metadata["box_area_ratio"] = box_area / img_area
        except (ValueError, TypeError) as e:
            metadata["box_area_ratio"] = None
    
    return metadata


def format_for_supervised_dataset_grounding(
    sample: Dict[str, Any],
    metadata: Dict[str, Any],
    output_images_dir: Optional[Path] = None,
    images_relative_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert Salesforce grounding sample to UI-TARS SupervisedDataset JSON format.
    
    Args:
        sample: Original sample from Salesforce grounding dataset
        metadata: Extracted metadata
        output_images_dir: Directory to save images (required for PIL.Image objects)
        images_relative_path: Relative path from JSON file to images directory
    """
    instruction = metadata.get("instruction", "")
    bbox = metadata.get("bbox")
    image = metadata.get("image")
    uuid = metadata.get("uuid")
    center_x = metadata.get("center_x")
    center_y = metadata.get("center_y")
    
    # Validate required fields
    if not instruction:
        raise ValueError("Instruction is required but was empty")
    
    if bbox is None or len(bbox) != 4:
        raise ValueError("Invalid bbox for formatting")
    
    if center_x is None or center_y is None:
        raise ValueError("Center coordinates not computed from bbox")
    
    # Get image dimensions
    if isinstance(image, Image.Image):
        img_width, img_height = image.size
    else:
        raise ValueError(f"Unexpected image type: {type(image)}. Expected PIL.Image.")
    
    # Normalize coordinates using prepare_training_coordinates
    norm_x, norm_y = prepare_training_coordinates(center_x, center_y, img_width, img_height)
    
    # Format user prompt
    user_prompt = f"<image>\n{UITARS_USR_PROMPT_NOTHOUGHT.format(instruction=instruction)}"
    
    # Format assistant response
    assistant_response = f"Action: click(start_box='({norm_x}, {norm_y})')"
    
    # Handle image saving and reference
    if isinstance(image, Image.Image):
        if output_images_dir is not None:
            # Create filename from uuid
            image_filename = f"{uuid}.png"
            image_path = output_images_dir / image_filename
            image.save(image_path, "PNG")
            # Use relative path for SupervisedDataset compatibility
            if images_relative_path:
                image_ref = f"{images_relative_path}/{image_filename}"
            else:
                image_ref = image_filename
        else:
            raise ValueError("output_images_dir must be provided to save PIL.Image objects")
    else:
        raise ValueError(f"Unexpected image type: {type(image)}")
    
    # Require uuid - fail if not present
    if uuid is None:
        raise ValueError("uuid is required but was None. This should have been caught in extract_metadata_grounding().")
    
    # Use the uuid directly (convert to string if needed)
    entry_id = str(uuid)
    
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
    dataset_counts = Counter()
    box_area_ratios = []
    
    for sample in samples:
        metadata = sample.get("metadata", {})
        dataset_counts[metadata.get("dataset", "unknown")] += 1
        if metadata.get("box_area_ratio") is not None:
            box_area_ratios.append(metadata["box_area_ratio"])
    
    stats = {
        "dataset_distribution": dict(dataset_counts),
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


def split_train_val_stratified(
    samples: List[Dict[str, Any]], 
    val_ratio: float, 
    rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split samples into train and validation sets with stratified sampling by dataset source.
    
    Groups samples by metadata["dataset"] (ariaui, omniact, etc.) and splits each group
    by the specified validation ratio to maintain distribution across train/val.
    
    Args:
        samples: List of sample data dictionaries with metadata
        val_ratio: Validation split ratio (e.g., 0.1 = 10% validation, 90% train)
        rng: Random number generator for reproducibility
    
    Returns:
        Tuple of (train_samples, val_samples) lists
    """
    if val_ratio <= 0.0 or val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    
    # Group samples by dataset source
    dataset_groups: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        dataset = sample.get("metadata", {}).get("dataset", "unknown")
        if dataset not in dataset_groups:
            dataset_groups[dataset] = []
        dataset_groups[dataset].append(sample)
    
    train_samples = []
    val_samples = []
    
    # Split each dataset group
    for dataset, group_samples in dataset_groups.items():
        # Shuffle group for random split
        group_copy = list(group_samples)  # Make a copy to avoid modifying original
        rng.shuffle(group_copy)
        
        # Calculate split point
        n_total = len(group_copy)
        n_val = max(1, int(n_total * val_ratio))  # At least 1 sample for validation
        
        # Split
        val_group = group_copy[:n_val]
        train_group = group_copy[n_val:]
        
        train_samples.extend(train_group)
        val_samples.extend(val_group)
        
        print(f"  {dataset}: {len(train_group)} train, {len(val_group)} val (total: {n_total})")
    
    return train_samples, val_samples


def _save_single_split(
    valid_samples: List[Dict[str, Any]],
    output_path: Path,
    split_name: str,
    seed: int,
    num_samples: Optional[int],
    dataset_filter: Optional[str],
    stratified: bool,
    val_ratio: Optional[float] = None,
    is_validation: bool = False
) -> None:
    """
    Helper function to save a single split (train or val) to disk.
    
    Args:
        valid_samples: List of sample data dictionaries
        output_path: Output JSON file path
        split_name: Name of the split ("train" or "val")
        seed: Random seed used
        num_samples: Number of samples requested (original total)
        dataset_filter: Dataset filter applied
        stratified: Whether stratified sampling was used
        val_ratio: Validation ratio (if split was used)
        is_validation: Whether this is the validation split
    """
    # Create images directory alongside JSON file
    images_dir_name = f"{output_path.stem}_images"
    images_dir = output_path.parent / images_dir_name
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{split_name.upper()}] Images will be saved to: {images_dir}")
    
    # Format samples for SupervisedDataset and save images
    print(f"[{split_name.upper()}] Formatting samples and saving images...")
    formatted_entries = []
    format_errors = 0
    
    for idx, sample_data in enumerate(valid_samples):
        try:
            entry = format_for_supervised_dataset_grounding(
                sample_data["sample"],
                sample_data["metadata"],
                output_images_dir=images_dir,
                images_relative_path=images_dir_name
            )
            formatted_entries.append(entry)
        except Exception as e:
            format_errors += 1
            sample_id = sample_data.get("metadata", {}).get("uuid") or f"sample_{idx}"
            print(f"Warning: Failed to format/save sample {sample_id}: {e}")
            continue
    
    print(f"[{split_name.upper()}] Formatted {len(formatted_entries)} entries ({format_errors} errors)")
    
    print(f"[{split_name.upper()}] Writing output to {output_path}...")
    with open(output_path, 'w') as f:
        ujson.dump(formatted_entries, f, indent=2)
    
    print(f"[{split_name.upper()}] Wrote {len(formatted_entries)} entries to {output_path}")
    print(f"[{split_name.upper()}] Saved {len(formatted_entries)} images to {images_dir}")
    
    # Compute and write metadata
    meta_path = output_path.with_suffix('.meta.json')
    metadata_stats = compute_metadata_stats(valid_samples)
    metadata_output = {
        "seed": seed,
        "num_samples": len(formatted_entries),
        "requested_samples": num_samples,
        "source": "Salesforce/grounding_dataset",
        "dataset_filter": dataset_filter,
        "stratified": stratified,
        "images_dir": str(images_dir.relative_to(output_path.parent)) if images_dir.exists() else None,
        **metadata_stats
    }
    
    # Add split information if this is a split output
    if val_ratio is not None:
        metadata_output["val_ratio"] = val_ratio
        metadata_output["split_type"] = "validation" if is_validation else "train"
    
    print(f"[{split_name.upper()}] Writing metadata to {meta_path}...")
    with open(meta_path, 'w') as f:
        ujson.dump(metadata_output, f, indent=2)
    
    print(f"[{split_name.upper()}] Done! Output files:")
    print(f"  - {output_path}")
    print(f"  - {meta_path}")
    print(f"  - {images_dir}/ ({len(formatted_entries)} images)")


def save_output(
    valid_samples: List[Dict[str, Any]],
    output_json: str,
    seed: int,
    num_samples: Optional[int],
    dataset_filter: Optional[str],
    stratified: bool,
    split_output: bool = False,
    val_ratio: float = 0.1
) -> None:
    """
    Format samples and save to disk.
    
    Args:
        valid_samples: List of sample data dictionaries
        output_json: Output JSON file path
        seed: Random seed used
        num_samples: Number of samples requested
        dataset_filter: Dataset filter applied
        stratified: Whether stratified sampling was used
        split_output: Whether to split into train/val files
        val_ratio: Validation split ratio (only used if split_output=True)
    """
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if split_output:
        # Generate train and val file paths
        train_path = output_path.parent / f"{output_path.stem}_train{output_path.suffix}"
        val_path = output_path.parent / f"{output_path.stem}_val{output_path.suffix}"
        
        # Split samples
        print(f"\nSplitting samples into train/val (ratio: {val_ratio})...")
        train_samples, val_samples = split_train_val_stratified(
            valid_samples, val_ratio, random.Random(seed)
        )
        
        print(f"\nSplit complete: {len(train_samples)} train, {len(val_samples)} val")
        
        # Save train split
        _save_single_split(
            train_samples, train_path, "train", seed, num_samples,
            dataset_filter, stratified, val_ratio, is_validation=False
        )
        
        # Save validation split
        _save_single_split(
            val_samples, val_path, "val", seed, num_samples,
            dataset_filter, stratified, val_ratio, is_validation=True
        )
        
        print(f"\n=== All splits saved successfully ===")
    else:
        # Single output file (original behavior)
        _save_single_split(
            valid_samples, output_path, "output", seed, num_samples,
            dataset_filter, stratified, None, is_validation=False
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert Salesforce grounding dataset to UI-TARS SupervisedDataset format"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/grounding_dataset.json",
        help="Output JSON file path (default: data/grounding_dataset.json)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to generate (default: all samples)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--dataset_filter",
        type=str,
        default=None,
        choices=["ariaui", "omniact", "widget_caption", "ui_vision", "os_atlas"],
        help="Filter by source dataset (optional)"
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Enable stratified sampling by source dataset (default: False)"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download entire dataset instead of streaming (WARNING: ~34.7 GB download). Default is streaming mode."
    )
    parser.add_argument(
        "--max_first_pass_samples",
        type=int,
        default=20000,
        help="Maximum samples to process in first pass for stratified sampling (default: 20000)"
    )
    parser.add_argument(
        "--min_samples_per_dataset",
        type=int,
        default=50,
        help="Minimum samples per dataset before early termination in stratified sampling (default: 50)"
    )
    parser.add_argument(
        "--max_collect_all_samples",
        type=int,
        default=100000,
        help="Maximum samples to collect when using collect-all strategy with streaming (default: 100000). Set to 0 to disable limit (requires --no-streaming)."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1 = 10%% validation, 90%% train)"
    )
    parser.add_argument(
        "--split_output",
        action="store_true",
        help="Enable train/val splitting (when set, generates both train and val JSON files)"
    )
    
    args = parser.parse_args()
    
    # Additional validation
    if args.max_collect_all_samples < 0:
        parser.error("--max_collect_all_samples must be >= 0")
    if args.max_first_pass_samples <= 0:
        parser.error("--max_first_pass_samples must be > 0")
    if args.min_samples_per_dataset <= 0:
        parser.error("--min_samples_per_dataset must be > 0")
    if args.val_ratio <= 0.0 or args.val_ratio >= 1.0:
        parser.error("--val_ratio must be between 0 and 1 (exclusive)")
    
    # Set random seed
    random.seed(args.seed)
    rng = random.Random(args.seed)
    
    # Load dataset
    use_streaming = not args.no_streaming
    dataset = load_grounding_dataset(streaming=use_streaming)
    
    # Validate collect-all with streaming
    if not args.stratified and not args.num_samples:
        if use_streaming and args.max_collect_all_samples == 0:
            raise ValueError(
                "Cannot use collect-all strategy with streaming and max_collect_all_samples=0. "
                "Either use --no-streaming or set --max_collect_all_samples > 0"
            )
    
    print(f"Starting processing...")
    if use_streaming:
        print("WARNING: Using streaming mode. Images will be downloaded on-demand when accessed.")
        print("         This is necessary for coordinate normalization but will grow the cache.")
    if args.dataset_filter:
        print(f"Filtering by dataset: {args.dataset_filter}")
    if args.stratified:
        print(f"Using stratified sampling by source dataset")
    if args.num_samples:
        print(f"Target number of samples: {args.num_samples}")
    if args.split_output:
        print(f"Train/val split enabled: {args.val_ratio*100:.1f}% validation, {(1-args.val_ratio)*100:.1f}% train")
    
    # Create sampling strategy
    strategy = create_sampling_strategy(
        args=args,
        extract_metadata_fn=extract_metadata_grounding,
        is_valid_fn=is_valid_grounding_sample,
        get_filter_reason_fn=get_filter_reason,
        rng=rng
    )
    
    # Collect samples
    valid_samples = strategy.collect_samples(dataset, args)
    
    # Print summary
    summary = strategy.get_summary()
    print(f"\nProcessing complete!")
    print(f"Total samples processed: {summary['samples_processed']}")
    print(f"Samples filtered out: {summary['samples_filtered']}")
    print(f"Valid samples collected: {summary['valid_samples']}")
    
    # Print filter reasons summary
    if summary['filter_reasons']:
        print(f"\nFilter reasons summary:")
        for reason, count in Counter(summary['filter_reasons']).most_common():
            print(f"  {reason}: {count}")
    
    # Shuffle samples
    rng.shuffle(valid_samples)
    
    # Save output
    save_output(
        valid_samples=valid_samples,
        output_json=args.output_json,
        seed=args.seed,
        num_samples=args.num_samples,
        dataset_filter=args.dataset_filter,
        stratified=args.stratified,
        split_output=args.split_output,
        val_ratio=args.val_ratio
    )


if __name__ == "__main__":
    main()
