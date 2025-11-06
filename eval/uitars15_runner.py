"""
Run UITARSAgent evaluation on trajectory episodes.

Usage:
    python -m eval.uitars15_runner \\
        --episode_dir outputs/run_X/episode_Y \\
        --model your-model \\
        --output_jsonl results.jsonl

Set environment variables before running:
    export DOUBAO_API_URL="https://your-endpoint.com/v1"
    export DOUBAO_API_KEY="your-api-key"
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from eval.episode_loader import load_episode
from eval.uitars15_v1 import UITARSAgent


def build_runtime_conf(args: argparse.Namespace) -> Dict[str, Any]:
    """Build runtime_conf dict for UITARSAgent."""
    runtime_conf: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "infer_mode": args.infer_mode,
        "prompt_style": args.prompt_style,
        "input_swap": args.input_swap,
        "language": args.language,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "callusr_tolerance": args.callusr_tolerance,
    }
    if args.history_n is not None:
        runtime_conf["history_n"] = args.history_n
    return runtime_conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UITARSAgent over a recorded episode.")
    parser.add_argument("--episode_dir", type=str, required=True, help="Path to outputs/<run>/<episode> directory.")
    parser.add_argument("--model", type=str, required=True, help="Model name/id for the OpenAI-compatible endpoint.")
    parser.add_argument("--instruction_source", type=str, choices=["step", "global"], default="step")
    parser.add_argument(
        "--observation_type",
        type=str,
        choices=["screenshot", "screenshot_a11y_tree"],
        default="screenshot",
    )
    parser.add_argument("--infer_mode", type=str, default="qwen25vl_normal")
    parser.add_argument("--prompt_style", type=str, default="qwen25vl_normal")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument(
        "--history_n",
        type=int,
        default=None,
        help="How many past screenshots to include. Default None means use class default (5).",
    )
    parser.add_argument(
        "--reset_each_step",
        action="store_true",
        help="If set, reset agent state before each step (stateless per-step prompts).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_pixels", type=int, default=16384 * 28 * 28)
    parser.add_argument("--min_pixels", type=int, default=100 * 28 * 28)
    parser.add_argument("--input_swap", action="store_true", help="Use clipboard paste for typing.")
    parser.add_argument("--no-input_swap", dest="input_swap", action="store_false")
    parser.set_defaults(input_swap=True)
    parser.add_argument("--callusr_tolerance", type=int, default=1)
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional path to write per-step predictions as JSONL.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    
    episode_dir = Path(args.episode_dir)
    if not episode_dir.is_dir():
        raise NotADirectoryError(f"Invalid episode_dir: {episode_dir}")

    runtime_conf = build_runtime_conf(args)
    agent = UITARSAgent(
        model=args.model,
        runtime_conf=runtime_conf,
        observation_type=args.observation_type,
        model_type="qwen25vl",
    )

    jsonl_file: Optional[Any] = None
    if args.output_jsonl is not None:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = out_path.open("w")

    agent.reset()

    step_index = 0
    for instruction, obs, metadata in load_episode(
        str(episode_dir), instruction_source=args.instruction_source
    ):
        if args.reset_each_step:
            agent.reset()

        try:
            prediction, actions = agent.predict(instruction, obs)
        except Exception as e:
            print(f"[runner] ERROR at step={step_index}: {e}")
            if jsonl_file is not None:
                record = {
                    "step_index": step_index,
                    "instruction": instruction,
                    "error": str(e),
                    "metadata": metadata,
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_file.flush()
            raise

        is_terminal = False
        if prediction == "client error" or actions in [["DONE"], ["FAIL"]]:
            is_terminal = True
            if prediction == "client error":
                print(f"[runner] WARNING: Client error at step={step_index}")
            elif actions == ["DONE"]:
                print(f"[runner] Task completed at step={step_index}")
            elif actions == ["FAIL"]:
                print(f"[runner] Task failed at step={step_index}")
        print(f"[runner] step={step_index}")
        print(f"[runner] instruction={instruction}")
        print(f"[runner] prediction={prediction}")
        print(f"[runner] actions={actions}")
        if jsonl_file is not None:
            record = {
                "step_index": step_index,
                "instruction": instruction,
                "prediction": prediction,
                "actions": actions,
                "metadata": metadata,
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_file.flush()

        step_index += 1
        if is_terminal:
            print(f"[runner] Stopping evaluation due to terminal state.")
            break

    print(f"[runner] Completed {step_index} steps.")
    
    if jsonl_file is not None:
        jsonl_file.close()


if __name__ == "__main__":
    main()


