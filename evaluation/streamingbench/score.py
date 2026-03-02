#!/usr/bin/env python3
"""
StreamingBench result handling:
- Merge multi-GPU outputs (if multiple files present)
- Score accuracy by task and overall
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def extract_answer(response):
    """Extract answer letter from model response"""
    if not response:
        return None

    response = response.strip()

    patterns = [
        r"answer is ([A-D])",
        r"^([A-D])[\.:\s]",
        r"\(([A-D])\)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    if response and response[0] in 'ABCD':
        return response[0]

    match = re.search(r"[A-D]", response)
    return match.group(0) if match else None


def merge_results(result_dir: Path) -> Path:
    """Merge gpu_*.jsonl under result_dir/output into merged_results.jsonl (if multiple files)."""
    output_dir = result_dir / "output"
    gpu_files = sorted(output_dir.glob("gpu_*.jsonl"))
    if not gpu_files:
        # fall back to any jsonl under result_dir
        any_files = sorted(result_dir.glob("**/*.jsonl"))
        if not any_files:
            raise FileNotFoundError(f"No JSONL files found in {result_dir}")
        return any_files[0]

    if len(gpu_files) == 1:
        return gpu_files[0]

    merged = result_dir / "merged_results.jsonl"
    with open(merged, "w") as fout:
        for jf in gpu_files:
            with open(jf) as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
    return merged


def calculate_scores(jsonl_file: Path):
    """Calculate accuracy scores from results"""
    stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    with open(jsonl_file) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task_type = data.get('task_type', 'unknown')
            stats['overall']['total'] += 1
            stats[task_type]['total'] += 1
            if data.get('correct'):
                stats['overall']['correct'] += 1
                stats[task_type]['correct'] += 1
            elif 'predicted_answer' in data and 'answer' in data:
                if str(data['predicted_answer']).strip().lower() == str(data['answer']).strip().lower():
                    stats['overall']['correct'] += 1
                    stats[task_type]['correct'] += 1

    results = {}
    for task_type, counts in stats.items():
        if counts['total'] > 0:
            results[task_type] = {
                'total': counts['total'],
                'correct': counts['correct'],
                'accuracy': counts['correct'] / counts['total']
            }
    return results


def main():
    parser = argparse.ArgumentParser(description='StreamingBench Merge + Scoring')
    parser.add_argument('--result_dir', required=True, help='Result directory')
    parser.add_argument('--model_name', required=True, help='Model name')
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_file = merge_results(result_dir)
    print(f"Scoring file: {result_file}")

    scores = calculate_scores(result_file)

    print("\nStreamingBench Results")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print("=" * 50)

    for task_type in sorted(scores.keys()):
        if task_type != 'overall':
            s = scores[task_type]
            print(f"{task_type:30} {s['correct']:4d}/{s['total']:4d} = {s['accuracy']*100:6.2f}%")

    if 'overall' in scores:
        print("-" * 50)
        s = scores['overall']
        print(f"{'Overall':30} {s['correct']:4d}/{s['total']:4d} = {s['accuracy']*100:6.2f}%")

    output_file = result_dir / f"{args.model_name}_scores.json"
    with open(output_file, 'w') as f:
        json.dump({'model': args.model_name, 'scores': scores}, f, indent=2)

    print(f"\nScores saved to: {output_file}")


if __name__ == '__main__':
    main()
