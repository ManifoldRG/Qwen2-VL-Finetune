"""
Visualize Salesforce dataset examples (UI-TARS JSON format).

This script:
- loads a generated JSON file (list of entries)
- displays each screenshot (smart-resized to training space)
- prints the instruction (tail of the human prompt)
- overlays the action coordinates parsed from the GPT action string

Example:
  python eval/visualize_salesforce_dataset.py \
    --json_path /mnt/disks/sca-data/salesforce-25k/salesforce_1.json \
    --dataset_root /mnt/disks/sca-data/salesforce-25k
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from typing import Any, Iterable, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image as PIL_Image
    from PIL import ImageDraw as PIL_ImageDraw
    from PIL import ImageFont as PIL_ImageFont


IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    *,
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Match the training-time smart resize (copied from src/prepare_salesforce_dataset_full.py).
    Returns (smart_h, smart_w).
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


def _resolve_image_path(dataset_root: str, image_field: Any) -> str:
    if isinstance(image_field, str):
        rel = image_field
    elif isinstance(image_field, Sequence) and image_field and isinstance(image_field[0], str):
        rel = image_field[0]
    else:
        raise ValueError(f"Invalid 'image' field: {image_field!r}")

    return rel if os.path.isabs(rel) else os.path.join(dataset_root, rel)


def extract_instruction_from_human_prompt(human_value: str) -> str:
    """
    Extract the instruction at the end of the human prompt.
    Expected format includes:
      '## User Instruction\\n<instruction>\\n'
    """
    marker = "## User Instruction\n"
    if marker in human_value:
        tail = human_value.split(marker, 1)[1]
        return tail.strip()
    # Fallback: last non-empty line
    lines = [ln.strip() for ln in human_value.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _parse_coord_string(coord_str: str) -> tuple[float, float]:
    """
    Parse "(x, y)" or "x,y" or "[x, y]" into (x, y).
    """
    s = coord_str.strip()
    s = s.replace("<|box_start|>", "").replace("<|box_end|>", "").strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return float(parsed[0]), float(parsed[1])
    except Exception:
        pass

    # Fallback: try to extract two numbers.
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", s)
    if not m:
        raise ValueError(f"Could not parse coordinate string: {coord_str!r}")
    return float(m.group(1)), float(m.group(2))


def parse_action_coordinates(action_text: str) -> list[tuple[float, float]]:
    """
    Extract coordinates from UITARS action strings.
    Examples:
      Action: click(start_box='(1063, 786)') -> [(1063, 786)]
      Action: drag(start_box='(100,200)', end_box='(300,400)') -> [(100,200), (300,400)]
    """
    coords: list[tuple[float, float]] = []
    for key in ("start_box", "end_box"):
        matches = re.findall(rf"{key}\s*=\s*['\"]([^'\"]+)['\"]", action_text)
        for match in matches:
            coords.append(_parse_coord_string(match))
    return coords


def _import_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:  # pragma: no cover
        raise SystemExit(
            "Pillow is required for image visualization. Install with: pip install Pillow\n"
            "Tip: You can still validate parsing with --dry_run."
        )
    return Image, ImageDraw, ImageFont


def _load_font(size: int):
    _, _, ImageFont = _import_pillow()
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_marker(
    img,
    coords: Sequence[tuple[float, float]],
    *,
    label_prefix: str = "GT",
) -> Any:
    _, ImageDraw, _ = _import_pillow()
    out = img.copy()
    draw = ImageDraw.Draw(out)

    font = _load_font(14)
    point_radius = 10
    crosshair = 18

    for i, (x, y) in enumerate(coords):
        xi, yi = int(round(x)), int(round(y))
        color = "lime" if i == 0 else "orange"
        outline = "black"

        bbox = [xi - point_radius, yi - point_radius, xi + point_radius, yi + point_radius]
        draw.ellipse(bbox, fill=color, outline=outline, width=2)
        draw.line([(xi - crosshair, yi), (xi + crosshair, yi)], fill=color, width=2)
        draw.line([(xi, yi - crosshair), (xi, yi + crosshair)], fill=color, width=2)

        label = f"{label_prefix}{i+1}" if len(coords) > 1 else label_prefix
        text_pos = (xi + point_radius + 6, yi - 10)
        tb = draw.textbbox(text_pos, label, font=font)
        draw.rectangle(tb, fill="black", outline="white", width=1)
        draw.text(text_pos, label, fill="white", font=font)

    return out


def visualize_entry(
    *,
    entry: dict[str, Any],
    dataset_root: str,
    show: bool,
    pause: bool,
    save_dir: Optional[str],
    show_original: bool,
) -> None:
    Image, _, _ = _import_pillow()
    image_abs_path = _resolve_image_path(dataset_root, entry.get("image"))
    conversations = entry.get("conversations") or []
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError(f"Invalid conversations for entry id={entry.get('id')!r}")

    human_value = conversations[0].get("value", "")
    gpt_value = conversations[1].get("value", "")

    instruction = extract_instruction_from_human_prompt(human_value)
    coords = parse_action_coordinates(gpt_value)

    img = Image.open(image_abs_path)
    if getattr(img, "mode", None) != "RGB":
        img = img.convert("RGB")

    orig_w, orig_h = img.size
    smart_h, smart_w = smart_resize(height=orig_h, width=orig_w)

    resized = img.resize((smart_w, smart_h), resample=Image.Resampling.LANCZOS)
    annotated = draw_marker(resized, coords, label_prefix="GT")

    # Overlay a short text header (instruction + coords)
    _, ImageDraw, _ = _import_pillow()
    draw = ImageDraw.Draw(annotated)
    font = _load_font(14)
    header = f"{instruction}   |   {coords}"
    tb = draw.textbbox((10, 10), header, font=font)
    pad = 6
    bg = [tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad]
    draw.rectangle(bg, fill="black", outline="white", width=1)
    draw.text((10, 10), header, fill="white", font=font)

    print("=" * 80)
    print(f"id: {entry.get('id')}")
    print(f"image: {image_abs_path}")
    print(f"orig_size: {orig_w}x{orig_h}  smart_size: {smart_w}x{smart_h}")
    print(f"instruction: {instruction}")
    print(f"action: {gpt_value}")
    print(f"coords: {coords}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_id = str(entry.get("id", "entry")).replace(os.sep, "_")
        out_path = os.path.join(save_dir, f"{safe_id}.png")
        annotated.save(out_path)
        print(f"saved: {out_path}")

        if show_original:
            # Map training coords back to original image space and draw there too.
            mapped = []
            for x, y in coords:
                mapped.append((x * orig_w / smart_w, y * orig_h / smart_h))
            orig_annotated = draw_marker(img, mapped, label_prefix="GT(orig)")
            out_path2 = os.path.join(save_dir, f"{safe_id}.orig.png")
            orig_annotated.save(out_path2)
            print(f"saved: {out_path2}")

    if show:
        try:
            annotated.show()
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to display image (headless environment?): {e}")

        if pause:
            resp = input("Enter for next (or 'q' to quit): ").strip().lower()
            if resp == "q":
                raise StopIteration


def load_entries(json_path: str) -> list[dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of entries")
    return data


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize Salesforce UI-TARS dataset JSON entries.")
    parser.add_argument(
        "--json_path",
        default="/mnt/disks/sca-data/salesforce-25k/salesforce.json",
        help="Path to generated JSON (list of entries).",
    )
    parser.add_argument(
        "--dataset_root",
        default="/mnt/disks/sca-data/salesforce-25k",
        help="Root dir containing screenshots/ (used to resolve relative image paths).",
    )
    parser.add_argument("--index", type=int, default=None, help="View only a single entry by index.")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit number of examples shown.")
    parser.add_argument("--no_show", action="store_true", help="Do not call Image.show().")
    parser.add_argument("--no_pause", action="store_true", help="Do not wait for Enter between examples.")
    parser.add_argument("--save_dir", default=None, help="If set, save annotated images here.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse entries and verify paths/coords without importing Pillow or opening images.",
    )
    parser.add_argument(
        "--show_original",
        action="store_true",
        help="Also save an overlay drawn in original image space (mapped back from training coords). Requires --save_dir.",
    )
    args = parser.parse_args(argv)

    entries = load_entries(args.json_path)

    indices: Iterable[int]
    if args.index is not None:
        if args.index < 0 or args.index >= len(entries):
            raise ValueError(f"--index out of range (0..{len(entries)-1})")
        indices = [args.index]
    else:
        indices = range(len(entries))

    shown = 0
    try:
        for i in indices:
            entry = entries[i]
            if args.dry_run:
                image_abs_path = _resolve_image_path(args.dataset_root, entry.get("image"))
                conversations = entry.get("conversations") or []
                human_value = conversations[0].get("value", "") if conversations else ""
                gpt_value = conversations[1].get("value", "") if len(conversations) > 1 else ""
                instruction = extract_instruction_from_human_prompt(human_value)
                coords = parse_action_coordinates(gpt_value)
                print("=" * 80)
                print(f"index: {i}")
                print(f"id: {entry.get('id')}")
                print(f"image_exists: {os.path.exists(image_abs_path)}")
                print(f"image: {image_abs_path}")
                print(f"instruction: {instruction}")
                print(f"action: {gpt_value}")
                print(f"coords: {coords}")
            else:
                visualize_entry(
                    entry=entry,
                    dataset_root=args.dataset_root,
                    show=not args.no_show,
                    pause=not args.no_pause,
                    save_dir=args.save_dir,
                    show_original=args.show_original,
                )

            shown += 1
            if args.max_examples is not None and shown >= args.max_examples:
                break
    except StopIteration:
        return


if __name__ == "__main__":
    main()

