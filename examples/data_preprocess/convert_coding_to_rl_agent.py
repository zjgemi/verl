#!/usr/bin/env python3
"""
Convert scientific coding practice data into verl agentic RL format.

This script reads the raw problem data (generator_output.json + log.json) and
produces Parquet files suitable for verl's ToolAgentLoop multi-turn training.

The output data format includes:
- agent_name: "tool_agent" — selects the ToolAgentLoop
- tools_kwargs: per-sample sandbox configuration (image, domain)
- reward_model.ground_truth: JSON-serialized ground truth answer

Usage:
  python examples/data_preprocess/convert_coding_to_rl_agent.py \
    --input_dir /personal/sp2/qwen_trajs/cc_deepseek_1w/split_data/train \
    --output_dir ~/data/coding_practice \
    --system_prompt_path /personal/sp2/qwen_trajs/cc_deepseek_1w/prompt.md \
    --images_path /personal/sp2/qwen_trajs/cc_deepseek_1w/images.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def load_system_prompt(path: str) -> str:
    """Load the system prompt from a markdown file."""
    with open(path) as f:
        return f.read().strip()


def load_images(images_path: str) -> dict[str, str]:
    """Load images.json mapping from domain -> Docker image."""
    with open(images_path) as f:
        return json.load(f)


def find_image_for_domain(domain: str, images: dict[str, str]) -> str | None:
    """Find the matching Docker image for a given domain name."""
    domain_lower = domain.lower().replace("_", "").replace("-", "")
    for key, image in images.items():
        normalized_key = key.lower().replace("_", "").replace("-", "")
        if normalized_key in domain_lower:
            return image
    # Fallback: match first word
    for key, image in images.items():
        first_word = key.lower().split("_")[0]
        if first_word in domain_lower:
            return image
    return None


def load_sample(sample_dir: str, domain: str, image: str | None, system_prompt: str) -> dict | None:
    """Load a single sample from a directory and return verl-format dict."""
    gen_path = os.path.join(sample_dir, "generator_output.json")
    log_path = os.path.join(sample_dir, "log.json")

    if not os.path.exists(gen_path) or not os.path.exists(log_path):
        return None

    with open(gen_path) as f:
        gen = json.load(f)
    with open(log_path) as f:
        log = json.load(f)

    problem = gen.get("problem", "")
    if not problem:
        return None

    # Extract ground truth
    verify_status = log.get("verify_status", [])
    if not verify_status:
        return None
    ground_truth = verify_status[0].get("ground_truth_answer")
    if ground_truth is None:
        return None

    task_dir = os.path.basename(os.path.dirname(sample_dir))
    sample_id = os.path.basename(sample_dir)

    return {
        "data_source": "coding_practice",
        "agent_name": "tool_agent",
        "prompt": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": problem,
            },
        ],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
        },
        "extra_info": {
            "split": "train",
            "index": int(sample_id) if sample_id.isdigit() else abs(hash(sample_id)) % (10**9),
            "task_name": task_dir,
            "sample_id": sample_id,
            "domain": domain,
            "task_id": gen.get("task_id", -1),
            "field": log.get("field", ""),
            "difficulty": log.get("difficulty", ""),
            "answer_type": log.get("answer_type", "code"),
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "execute_command": {
                    "create_kwargs": {
                        "domain": domain,
                        "image": image or "",
                    },
                },
            },
        },
    }


def convert_dataset(
    input_dir: str,
    output_dir: str,
    images: dict[str, str],
    system_prompt: str,
    split: str = "train",
    domain_filter: str | None = None,
) -> str:
    """Traverse all samples and convert to verl format.

    Args:
        input_dir: Path to split_data/<split>/ directory.
        output_dir: Where to save output parquet files.
        images: Images mapping dict.
        split: "train" / "test" / "validation".
        domain_filter: If set, only process this domain (e.g., "pymatgen_coding_practice_in_Materials").

    Returns:
        Path to the output parquet file.
    """
    samples = []
    failed = 0
    total_domains = 0

    # input_dir/<domain>/raw/<task_name>/<sample_id>/
    for domain in sorted(os.listdir(input_dir)):
        if domain_filter and domain != domain_filter:
            continue

        domain_path = os.path.join(input_dir, domain, "raw")
        if not os.path.isdir(domain_path):
            continue

        image = find_image_for_domain(domain, images)
        if not image:
            print(f"  [WARNING] No image found for domain '{domain}', skipping sandbox image")
        else:
            print(f"  Domain '{domain}' -> image: {image.split('/')[-1].split(':')[0]}")

        for task_dir in sorted(os.listdir(domain_path)):
            task_path = os.path.join(domain_path, task_dir)
            if not os.path.isdir(task_path):
                continue

            for sample_dir in sorted(os.listdir(task_path)):
                sample_path = os.path.join(task_path, sample_dir)
                if not os.path.isdir(sample_path):
                    continue

                sample = load_sample(sample_path, domain, image, system_prompt)
                if sample is None:
                    failed += 1
                    continue

                sample["extra_info"]["split"] = split
                samples.append(sample)

        total_domains += 1

    print(f"\nProcessed {total_domains} domains, {len(samples)} samples loaded, {failed} failed")

    if not samples:
        print("ERROR: No valid samples found")
        return ""

    # Save as Parquet
    df = pd.DataFrame(samples)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(samples)} samples to {output_path}")

    # Also save JSONL for inspection
    jsonl_path = os.path.join(output_dir, f"{split}.jsonl")
    with open(jsonl_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved JSONL copy to {jsonl_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert coding practice data to verl agentic RL format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to split_data/<split>/ directory (contains domain subdirectories).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for parquet files.",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="/personal/sp2/qwen_trajs/cc_deepseek_1w/images.json",
        help="Path to images.json (Docker image mapping).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="Dataset split name.",
    )
    parser.add_argument(
        "--system_prompt_path",
        type=str,
        default=None,
        help="Path to system prompt markdown file (e.g., prompt.md). If not set, uses a minimal default.",
    )
    parser.add_argument(
        "--domain_filter",
        type=str,
        default=None,
        help="Only process this domain (e.g., 'pymatgen_coding_practice_in_Materials').",
    )
    args = parser.parse_args()

    images = load_images(args.images_path)
    print(f"Loaded {len(images)} image mappings")

    if args.system_prompt_path:
        system_prompt = load_system_prompt(args.system_prompt_path)
        print(f"Loaded system prompt from {args.system_prompt_path} ({len(system_prompt)} chars)")
    else:
        system_prompt = (
            "You are an expert Python programmer specializing in scientific computing. "
            "Write Python code to solve the given problem. "
            "Your final answer MUST be printed as a Python literal on the LAST line of stdout. "
            "All debug output must go to earlier lines or stderr."
        )
        print("Using default system prompt (no --system_prompt_path provided)")

    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        images=images,
        system_prompt=system_prompt,
        split=args.split,
        domain_filter=args.domain_filter,
    )


if __name__ == "__main__":
    main()