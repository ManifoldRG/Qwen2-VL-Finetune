"""
Convert the local Salesforce UI dataset (HF datasets load_from_disk) into the
UI-TARS / LLaVA-style JSON format used by this repo's SFT pipeline.

Output schema (list of dicts):
  {
    "id": "salesforce_<dataset>_<uuid>",
    "image": ["screenshots/<dataset>_<uuid>.png"],
    "conversations": [
      {"from": "human", "value": "<image>\\n<prompt template with instruction>"},
      {"from": "gpt", "value": "Action: click(start_box='(x, y)')"}
    ]
  }

The click coordinate is computed from bbox center in original pixel space, then
mapped into the same smart-resized coordinate space as src/prepare_mind2web_data.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from typing import Any, Iterable

from datasets import load_dataset, load_from_disk
from PIL import Image


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
{instruction}
"""


IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that:
      1) both dims divisible by factor
      2) total pixels within [min_pixels, max_pixels]
      3) aspect ratio maintained as closely as possible
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
    original_x: int, original_y: int, original_width: int, original_height: int
) -> tuple[int, int]:
    """Convert original image coordinates to smart-resized space for training."""
    smart_h, smart_w = smart_resize(
        height=original_height,
        width=original_width,
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    training_x = int(original_x * smart_w / original_width)
    training_y = int(original_y * smart_h / original_height)
    return training_x, training_y


def get_image_dimensions(image_path: str) -> tuple[int, int]:
    """
    Get the width and height of an image from its file path.
    Mirrors src/prepare_mind2web_data.py to keep coordinate conversion identical.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}")


def parse_target_coordinates(coord_str: str) -> tuple[int, int]:
    """Parse target_coordinates string '(x, y)' to (x, y) tuple."""
    match = re.match(r"\((\d+),\s*(\d+)\)", coord_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Invalid coordinate format: {coord_str}")


def extract_type_value_from_instruction(instruction: str) -> str | None:
    """Extract type content from instruction. Mirrors src/prepare_mind2web_data.py."""
    patterns = [
        r"Type\s+['\"]([^'\"]+)['\"]",  # Matches both single and double quotes
        r"Type\s+['\"]([^'\"]*?)['\"]",  # Non-greedy version
    ]
    for pattern in patterns:
        match = re.search(pattern, instruction)
        if match:
            return match.group(1)
    return None


def infer_op_from_instruction(instruction: str) -> str:
    """
    Salesforce grounding rows don't always include an explicit `op`.
    This heuristic selects a Mind2Web-compatible op label from the instruction.
    """
    s = (instruction or "").strip().lower()
    if s.startswith("type "):
        return "type"
    if s.startswith("press enter") or s == "press enter":
        return "press enter"
    if s.startswith("hit enter") or s == "hit enter":
        return "press enter"
    if s.startswith("ignore"):
        return "ignore"
    return "click"


def generate_action_prediction(
    *,
    op: str,
    instruction: str,
    screenshot_path: str,
    click_x: int | float | None = None,
    click_y: int | float | None = None,
    target_coordinates: str | None = None,
    type_action_value: str | None = None,
) -> str:
    """
    Generate UI-TARS action string using the same conversion logic as
    src/prepare_mind2web_data.py.

    Salesforce rows typically provide a bbox; callers should pass its center as
    click_x/click_y (original pixel space) plus the associated instruction.
    """
    op = (op or "").lower()

    def _get_click_xy() -> tuple[int | float, int | float]:
        if target_coordinates is not None:
            return parse_target_coordinates(target_coordinates)
        if click_x is not None and click_y is not None:
            return click_x, click_y
        raise ValueError("Missing coordinates for action")

    if op in ["click", "hover", "click (fake)", "select"]:
        cx, cy = _get_click_xy()
        original_width, original_height = get_image_dimensions(screenshot_path)
        tx, ty = prepare_training_coordinates(cx, cy, original_width, original_height)
        return f"Action: click(start_box='({tx}, {ty})')"

    if op == "type":
        cx, cy = _get_click_xy()
        original_width, original_height = get_image_dimensions(screenshot_path)
        tx, ty = prepare_training_coordinates(cx, cy, original_width, original_height)

        if type_action_value is None:
            type_action_value = extract_type_value_from_instruction(instruction or "")

        if type_action_value is None or type_action_value == "":
            return f"Action: click(start_box='({tx}, {ty})')"

        click_action = f"Action: click(start_box='({tx}, {ty})')"
        type_action = f"type(content='{type_action_value}')"
        return f"{click_action}\n\n{type_action}"

    if op == "press enter":
        return "Action: type(content='\\n')"

    if op == "ignore":
        return "Action: wait()"

    raise ValueError(f"Unknown action type: {op}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_json(path: str, data: Any) -> None:
    """
    Write JSON atomically (best-effort) so crashes don't corrupt the main file.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _load_existing_entries(path: str) -> tuple[list[dict[str, Any]], set[str]]:
    """
    Load an existing output JSON (list of dicts) and return (entries, entry_ids).
    If the file cannot be parsed, it is renamed to *.corrupt and ([], set()) is returned.
    """
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError(f"Expected a JSON list, got {type(obj).__name__}")
        entries: list[dict[str, Any]] = [e for e in obj if isinstance(e, dict)]
        ids = {e.get("id") for e in entries if isinstance(e.get("id"), str)}
        return entries, ids  # type: ignore[return-value]
    except Exception as e:
        corrupt_path = path + ".corrupt"
        try:
            os.replace(path, corrupt_path)
        except Exception:
            pass
        print(
            f"[warn] Could not load existing JSON at {path} ({type(e).__name__}: {e}). "
            f"Starting fresh (old file moved aside if possible)."
        )
        return [], set()


def _validate_bbox(bbox: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox (expected 4-item list/tuple): {bbox!r}")
    x1, y1, x2, y2 = bbox
    if not all(isinstance(v, (int, float)) for v in (x1, y1, x2, y2)):
        raise ValueError(f"Invalid bbox values (expected numbers): {bbox!r}")
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
    if x2i <= x1i or y2i <= y1i:
        raise ValueError(f"Invalid bbox ordering (x2>x1 and y2>y1 required): {bbox!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)!r}")
    # Fail-fast if bbox is wildly outside the image; small negatives/overflows are also treated as errors.
    if x1i < 0 or y1i < 0 or x2i > width or y2i > height:
        raise ValueError(f"bbox out of image bounds bbox={bbox!r} image_size={(width, height)!r}")
    return x1i, y1i, x2i, y2i


def _bbox_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    return cx, cy


def _iter_with_index(ds: Iterable[Any]):
    for i, row in enumerate(ds):
        yield i, row


def _load_dataset_auto(input_dir: str) -> Any:
    """
    Load either:
      - a HF save_to_disk directory via load_from_disk, OR
      - raw parquet shards via load_dataset("parquet", data_files=...).

    Returns a `datasets.Dataset` (not a DatasetDict).
    """
    try:
        return load_from_disk(input_dir)
    except FileNotFoundError:
        # Fallback: if `input_dir` is raw parquet shards, load via load_dataset.
        if not os.path.isdir(input_dir):
            raise

        try:
            names = os.listdir(input_dir)
        except Exception:
            raise

        has_parquet_at_root = any(n.endswith(".parquet") for n in names)
        has_train_dirs = any(
            n.startswith("train-") and os.path.isdir(os.path.join(input_dir, n)) for n in names
        )

        data_files: str | None
        if has_parquet_at_root:
            data_files = os.path.join(input_dir, "*.parquet")
        elif has_train_dirs:
            data_files = os.path.join(input_dir, "train-*", "train-*.parquet")
        else:
            raise

        return load_dataset("parquet", data_files=data_files, split="train")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert /mnt/disks/sca-data/salesforce_grounding_dataset_full/data/ into UI-TARS JSON format."
    )
    parser.add_argument(
        "--input_dir",
        default="/mnt/disks/sca-data/salesforce_grounding_dataset_full/data/",
        help="Path passed to datasets.load_from_disk().",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write screenshots/ and the output JSON file.",
    )
    parser.add_argument("--num_samples", type=int, default=24935)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, deletes output_dir before writing.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional explicit JSON output path. Defaults to <output_dir>/salesforce_25k.json (or num_samples-based name).",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=500,
        help="Print progress every N examples.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Atomically save the output JSON every N newly-added entries (0 disables incremental saves).",
    )
    parser.add_argument(
        "--print_examples",
        type=int,
        default=2,
        help="Print the first N generated entries (set 0 to disable).",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress_every must be positive")
    if args.save_every < 0:
        raise ValueError("--save_every must be >= 0")
    if args.print_examples < 0:
        raise ValueError("--print_examples must be >= 0")

    output_dir = os.path.abspath(args.output_dir)
    screenshots_dir = os.path.join(output_dir, "screenshots")

    if args.overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    _ensure_dir(screenshots_dir)

    if args.output_json is None:
        json_name = f"salesforce.json"
        output_json_path = os.path.join(output_dir, json_name)
    else:
        output_json_path = os.path.abspath(args.output_json)
        _ensure_dir(os.path.dirname(output_json_path))

    if args.overwrite and os.path.exists(output_json_path):
        try:
            os.remove(output_json_path)
        except Exception:
            pass

    print(f"Loading dataset from: {args.input_dir}")
    dataset = _load_dataset_auto(args.input_dir)
    print(f"Loaded dataset with {len(dataset)} rows")

    if len(dataset) < args.num_samples:
        raise ValueError(f"Dataset only has {len(dataset)} rows, cannot select {args.num_samples}")

    print(f"Shuffling with seed={args.seed} and selecting first {args.num_samples} rows")
    sampled = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

    entries, done_ids = _load_existing_entries(output_json_path)
    if entries:
        print(f"Resuming from existing JSON: {output_json_path} (loaded {len(entries)} entries)")
    last_saved_len = len(entries)

    for idx, row in _iter_with_index(sampled):
        # Fast-path: if this sample was already saved, skip reprocessing.
        try:
            dataset_id = row.get("dataset")
            uuid = row.get("uuid")
            if dataset_id is not None and uuid is not None:
                safe_dataset_id = str(dataset_id).replace(os.sep, "_")
                entry_id = f"salesforce_{safe_dataset_id}_{uuid}"
                if entry_id in done_ids:
                    if (idx + 1) % args.progress_every == 0:
                        print(f"[resume] Scanned {idx + 1}/{args.num_samples} (already saved {len(entries)})")
                    continue
        except Exception:
            # If even this quick check fails, fall back to the normal per-row handler below.
            pass

        print(f"Processing row {idx}/{args.num_samples}")

        # Best-effort: skip individual bad samples instead of aborting the whole run.
        # Keep the error handling simple; many edge cases shouldn't happen in this dataset.
        image_abs_path: str | None = None
        dataset_id = None
        uuid = None
        try:
            dataset_id = row.get("dataset")
            uuid = row.get("uuid")
            instruction = row.get("instruction")
            bbox = row.get("bbox")
            image = row.get("image")

            if dataset_id is None or uuid is None:
                print(
                    f"[skip] Missing required fields dataset/uuid at index={idx}: keys={list(row.keys())}"
                )
                continue
            if instruction is None:
                print(f"[skip] Missing required field instruction at index={idx}: id={dataset_id}_{uuid}")
                continue
            if image is None or not isinstance(image, Image.Image):
                print(f"[skip] Missing/invalid image at index={idx}: id={dataset_id}_{uuid}")
                continue
            print(f" Instruction: {instruction}")

            # Ensure deterministic file naming and RGB PNG output
            safe_dataset_id = str(dataset_id).replace(os.sep, "_")
            file_stem = f"{safe_dataset_id}_{uuid}"
            image_rel_path = os.path.join("screenshots", f"{file_stem}.png")
            image_abs_path = os.path.join(output_dir, image_rel_path)

            rgb = image.convert("RGB")
            width, height = rgb.size

            x1, y1, x2, y2 = _validate_bbox(bbox, width=width, height=height)
            cx, cy = _bbox_center(x1, y1, x2, y2)

            print(f" Bbox: {bbox}")
            print(f" Bbox center: {cx}, {cy}")
            print(f" Image w/h: {width}x{height}")

            # Save image after validation to avoid writing corrupt/partial outputs.
            rgb.save(image_abs_path, format="PNG")

            prompt = f"<image>\n{UITARS_USR_PROMPT_NOTHOUGHT.format(instruction=instruction)}"
            op = row.get("op") or row.get("action") or infer_op_from_instruction(instruction)
            prediction = generate_action_prediction(
                op=op,
                instruction=instruction,
                screenshot_path=image_abs_path,
                click_x=cx,
                click_y=cy,
                target_coordinates=row.get("target_coordinates"),
                type_action_value=row.get("type_action_value"),
            )
            print(f"prediction: {prediction}")

            entry = {
                "id": f"salesforce_{safe_dataset_id}_{uuid}",
                "image": [image_rel_path],
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": prediction},
                ],
            }
            entries.append(entry)
            done_ids.add(entry["id"])

            if args.save_every > 0 and (len(entries) - last_saved_len) >= args.save_every:
                _atomic_write_json(output_json_path, entries)
                last_saved_len = len(entries)
                print(f"[save] Wrote {len(entries)} entries to: {output_json_path}")

            if args.print_examples > 0 and idx < args.print_examples:
                print(json.dumps(entry, indent=2)[:2000])

            if (idx + 1) % args.progress_every == 0:
                print(f"Processed {idx + 1}/{args.num_samples}")

            print("=" * 80)
        except Exception as e:
            # If we already wrote a screenshot for this row, remove it so JSON/images stay aligned.
            if image_abs_path is not None and os.path.exists(image_abs_path):
                try:
                    os.remove(image_abs_path)
                except Exception:
                    pass
            row_id = f"{dataset_id}_{uuid}" if dataset_id is not None or uuid is not None else "<unknown>"
            print(f"[skip] Failed to process index={idx} id={row_id}: {type(e).__name__}: {e}")
            continue

    print(f"Writing JSON to: {output_json_path}")
    _atomic_write_json(output_json_path, entries)

    print("Done.")
    print(f"Total entries: {len(entries)}")
    print(f"Screenshots dir: {screenshots_dir}")


if __name__ == "__main__":
    main()
