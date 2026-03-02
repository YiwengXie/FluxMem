#!/usr/bin/env python3
"""
StreamingBench evaluation for Qwen2.5-VL-FluxMem.

Single file handles both single-GPU and multi-GPU execution.
"""

import argparse
import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from qwen_vl_utils_fluxmem import process_vision_info
from qwen2_5_vl_fluxmem import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


class StreamingBenchEvaluator:
    """Evaluator for StreamingBench using Qwen2.5-VL-FluxMem."""

    PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {question}

Options:
{options}

The best option is:"""

    PROMPT_TEMPLATE_WITHOUT_OPTIONS = """You are an advanced video question-answering AI assistant. You have been provided with a video and a question related to the video. Your task is to carefully analyze the video and provide the answer to the question. 

Question: {question}

Answer:"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_logging()
        self._load_model()

    def _setup_logging(self):
        log_handlers = []
        if self.args.log_path:
            Path(self.args.log_path).parent.mkdir(parents=True, exist_ok=True)
            log_handlers.append(logging.FileHandler(self.args.log_path))
        log_handlers.append(logging.StreamHandler())
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=log_handlers,
        )
        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        self.logger.info(f"Loading FluxMem model: {self.args.ckpt_path}")
        torch.manual_seed(1234)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.args.ckpt_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            self.args.ckpt_path,
            min_pixels=16 * 28 * 28,
            max_pixels=self.args.max_pixels,
        )
        self.logger.info("FluxMem model loaded successfully")

    @staticmethod
    def _time_to_seconds(time_str):
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        raise ValueError(f"Invalid time format: {time_str}")

    def _format_prompt(self, question, options_str):
        if not options_str or options_str == 'nan' or pd.isna(options_str):
            return self.PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(question=question)
        try:
            options_list = eval(options_str)
        except Exception:
            options_list = options_str.split('\n')
        formatted_options = []
        for i, opt in enumerate(options_list):
            letter = chr(65 + i)
            formatted_options.append(opt if opt.startswith(f"{letter}.") else f"{letter}. {opt}")
        return self.PROMPT_TEMPLATE.format(question=question, options='\n'.join(formatted_options))

    def _extract_answer(self, response):
        import re
        response = response.strip()
        patterns = [
            r"option\s*([A-D])",
            r"([A-D])\s*is\s*the\s*best",
            r"answer\s*is\s*([A-D])",
            r"([A-D])\s*\)",
            r"^([A-D])$",
            r"option is\s*([A-D])",
            r"\(([A-D])\)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        match = re.search(r"[A-D]", response)
        return match.group(0) if match else ""

    def _process_video(self, video_path, prompt, timestamp_sec):
        fps = float(self.args.fps)
        start_time = 0
        end_time = timestamp_sec
        anchor_end = False
        if getattr(self.args, 'time_window_size', None) is not None and self.args.time_window_size > 0:
            start_time = max(0.0, end_time - float(self.args.time_window_size))
            anchor_end = True

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": 16 * 28 * 28,
                    "max_pixels": int(self.args.max_pixels),
                    "max_frames": int(self.args.max_num_frames),
                    "min_frames": 2,
                    "fps": fps,
                    "video_start": start_time,
                    "video_end": end_time,
                    "anchor_end": anchor_end,
                },
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                use_fluxmem=self.args.use_fluxmem,
                memory_drop_method=self.args.frame_sampling,
                short_frames=self.args.short_frames,
                medium_frames=self.args.medium_frames,
                pair_sim_threshold=self.args.pair_sim_threshold,
                save_path=self.args.save_path,
            )

        output_ids = generated_ids[0][inputs.input_ids.shape[1]:]
        response = self.processor.decode(output_ids, skip_special_tokens=True)
        return response

    def evaluate(self):
        df = pd.read_csv(self.args.task_csv)
        self.logger.info(f"Loaded {len(df)} questions")

        stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        Path(self.args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

        with open(self.args.output_jsonl, 'w') as f:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
                sample_id = row.question_id.split('_')[-2]
                video_path = os.path.join(self.args.video_dir, f"sample_{sample_id}", "video.mp4")
                if not os.path.exists(video_path):
                    self.logger.warning(f"Video not found: {video_path}")
                    continue

                timestamp_sec = self._time_to_seconds(row.time_stamp)
                prompt = self._format_prompt(row.question, row.options)

                try:
                    response = self._process_video(video_path, prompt, timestamp_sec)
                    has_options = not (pd.isna(row.options) or row.options == 'nan' or not row.options)

                    if has_options:
                        predicted = self._extract_answer(response)
                        correct = predicted == row.answer
                    else:
                        predicted = response.strip()
                        correct = None

                    result = {
                        'question_id': row.question_id,
                        'task_type': row.task_type,
                        'question': row.question,
                        'answer': row.answer,
                        'predicted_answer': predicted,
                        'response': response,
                        'has_options': has_options,
                        'correct': correct,
                    }

                    f.write(json.dumps(result) + '\n')
                    f.flush()

                    stats['overall']['total'] += 1
                    stats[row.task_type]['total'] += 1
                    if has_options and result['correct']:
                        stats['overall']['correct'] += 1
                        stats[row.task_type]['correct'] += 1
                    elif not has_options:
                        stats.setdefault('open_ended', {'total': 0, 'correct': 0})
                        stats['open_ended']['total'] += 1

                except Exception as e:
                    self.logger.error(f"Error processing {row.question_id}: {e}")
                    continue

        self._log_results(stats)
        self._save_summary(stats)

    def _log_results(self, stats):
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION RESULTS (Memory)")
        self.logger.info("=" * 50)
        for task_type, counts in stats.items():
            if counts['total'] > 0:
                acc = counts['correct'] / counts['total'] * 100
                self.logger.info(f"{task_type}: {counts['correct']}/{counts['total']} = {acc:.2f}%")

    def _save_summary(self, stats):
        summary = {
            'model': self.args.ckpt_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': {},
            'memory_args': {
                'use_fluxmem': self.args.use_fluxmem,
                'frame_sampling': self.args.frame_sampling,
                'short_frames': self.args.short_frames,
                'medium_frames': self.args.medium_frames,
            },
        }
        for task_type, counts in stats.items():
            if counts['total'] > 0:
                summary['results'][task_type] = {
                    'total': counts['total'],
                    'correct': counts['correct'],
                    'accuracy': counts['correct'] / counts['total'],
                }
        summary_path = self.args.output_jsonl.replace('.jsonl', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Summary saved to: {summary_path}")


def build_argparser():
    parser = argparse.ArgumentParser(description='StreamingBench Evaluation (Qwen2.5-VL-Memory)')
    parser.add_argument('--ckpt_path', required=True, help='Model checkpoint path')
    parser.add_argument('--task_csv', required=True, help='Task CSV file')
    parser.add_argument('--video_dir', required=True, help='Video directory')
    parser.add_argument('--result_dir', type=str, default=None, help='Result directory (optional)')
    parser.add_argument('--run_name', type=str, default='streamingbench', help='Run name for result directory')
    parser.add_argument('--output_jsonl', type=str, default=None, help='Output JSONL (auto-set if omitted)')
    parser.add_argument('--log_path', type=str, default=None, help='Log path (auto-set if omitted)')

    # Memory flags
    parser.add_argument('--use_fluxmem', action='store_true', help='Enable memory during generation')
    parser.add_argument('--frame_sampling', choices=['uniform', 'tail'], default='uniform', help='Frame sampling method')
    parser.add_argument('--short_frames', type=int, default=8, help='Number of short-term frames to keep fully')
    parser.add_argument('--medium_frames', type=int, default=16, help='Mid-term queue upper bound')
    parser.add_argument('--pair_sim_threshold', type=float, default=None, help='Fixed similarity threshold to bypass Otsu')

    # Video flags
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second for sampling')
    parser.add_argument('--max_num_frames', '--max-num-frames', dest='max_num_frames', type=int, default=256, help='Maximum frames to sample')
    parser.add_argument('--max_pixels', '--max-pixels', dest='max_pixels', type=int, default=256 * 28 * 28, help='Maximum pixels per frame')
    parser.add_argument('--time_window_size', type=float, default=None, help='Seconds to look back from timestamp; None = full history')
    parser.add_argument('--save_path', type=str, default=None, help='Optional jsonl to append memory/drop stats per sample')

    # Multi-GPU flags
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU mode')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs for multi-GPU mode')
    parser.add_argument('--dry_run', action='store_true', help='Show GPU splits without running (multi-GPU)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)  # internal use
    return parser


def _default_paths(args):
    ts = time.strftime('%Y%m%d_%H%M%S')
    result_dir = Path(args.result_dir) if args.result_dir else Path(f"eval_results/streamingbench/{args.run_name}_{ts}")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "log").mkdir(exist_ok=True)
    (result_dir / "output").mkdir(exist_ok=True)
    output_jsonl = args.output_jsonl or str(result_dir / "output" / f"results_{ts}.jsonl")
    log_path = args.log_path or str(result_dir / "log" / f"eval_{ts}.log")
    return result_dir, output_jsonl, log_path


def run_single(args):
    if args.output_jsonl and args.log_path:
        result_dir = Path(args.result_dir) if args.result_dir else Path(args.output_jsonl).parent.parent
        output_jsonl = args.output_jsonl
        log_path = args.log_path
    else:
        result_dir, output_jsonl, log_path = _default_paths(args)
        args.output_jsonl = output_jsonl
        args.log_path = log_path
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    evaluator = StreamingBenchEvaluator(args)
    evaluator.evaluate()
    return result_dir


def _split_dataframe(df, num_gpus):
    if num_gpus <= 1:
        return [df]
    chunk = len(df) // num_gpus
    splits = []
    for i in range(num_gpus):
        start = i * chunk
        end = start + chunk if i < num_gpus - 1 else len(df)
        splits.append(df.iloc[start:end])
    return splits


def run_multi(args):
    df = pd.read_csv(args.task_csv)
    splits = _split_dataframe(df, args.num_gpus)
    result_dir, _, _ = _default_paths(args)
    temp_dir = result_dir / "tmp"
    temp_dir.mkdir(exist_ok=True)

    def launch_worker(gpu_id, split_df):
        if len(split_df) == 0:
            return True
        split_csv = temp_dir / f"split_{gpu_id}.csv"
        split_df.to_csv(split_csv, index=False)
        output_jsonl = result_dir / "output" / f"gpu_{gpu_id}.jsonl"
        log_path = result_dir / "log" / f"gpu_{gpu_id}.log"

        cmd = [
            "python", "evaluation/streamingbench/streamingbench.py",
            "--ckpt_path", args.ckpt_path,
            "--task_csv", str(split_csv),
            "--video_dir", args.video_dir,
            "--output_jsonl", str(output_jsonl),
            "--log_path", str(log_path),
            "--worker",
        ]

        if args.use_fluxmem:
            cmd.append("--use_fluxmem")
        cmd += ["--frame_sampling", args.frame_sampling]
        cmd += ["--short_frames", str(args.short_frames), "--medium_frames", str(args.medium_frames)]
        if args.pair_sim_threshold is not None:
            cmd += ["--pair_sim_threshold", str(args.pair_sim_threshold)]
        cmd += [
            "--fps", str(args.fps),
            "--max-num-frames", str(args.max_num_frames),
            "--max-pixels", str(args.max_pixels),
        ]
        if args.time_window_size is not None:
            cmd += ["--time_window_size", str(args.time_window_size)]
        if args.save_path:
            cmd += ["--save_path", str(args.save_path)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {gpu_id}] start {len(split_df)} samples")
        res = subprocess.run(cmd, env=env)
        return res.returncode == 0

    if args.dry_run:
        for gid, split in enumerate(splits):
            print(f"GPU {gid}: {len(split)} samples")
        return result_dir

    with ThreadPoolExecutor(max_workers=args.num_gpus) as pool:
        results = list(pool.map(lambda p: launch_worker(*p), enumerate(splits)))

    if not all(results):
        raise RuntimeError("Some GPU workers failed")

    merged_file = result_dir / "merged_results.jsonl"
    with open(merged_file, "w") as fout:
        for jsonl_file in sorted((result_dir / "output").glob("gpu_*.jsonl")):
            with open(jsonl_file) as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
    print(f"Merged results -> {merged_file}")
    return result_dir


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if args.multi_gpu and not args.worker:
        result_dir = run_multi(args)
    else:
        result_dir = run_single(args)
    print(f"Finished. Results in: {result_dir}")


if __name__ == '__main__':
    main()
