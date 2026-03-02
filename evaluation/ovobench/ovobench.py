import torch
import json
import os
import os.path as osp
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import subprocess

import argparse
import logging
import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm import tqdm
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from qwen_vl_utils_fluxmem import process_vision_info
from qwen2_5_vl_fluxmem import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

# Parameters
RUN_NAME = "OVO-Bench"
CKPT_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
TASK_NAME = ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"]
TASK_JSON = "/data/junke_dont_remove/03_Evaluations/OVO-Bench/data/ovo_bench_new.json"
VIDEO_DIR = "/data/junke_dont_remove/03_Evaluations/OVO-Bench/data"
RESULT_DIR = "eval_results/ovobench"
LOG_PATH = "log/{run_name}_{curr_time}_{task}.log"
OUTPUT_JSONL = "outputs/{run_name}_{curr_time}_{task}.jsonl"
MIN_PIXELS = 16*28*28
MAX_PIXELS = 256*28*28
MIN_FRAMES = 4
MAX_FRAMES = 256
FPS = 1
# NFRAMES = 64

backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)

# helper functions
def build_prompt(task, question, options, _anno_, index):
    if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
        prompt = f"""
            Question: {question}
            Options:
            {formatted_options}
            Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
            Do not include any additional text or explanation in your response.
        """
    elif task == "REC":
        activity = _anno_["activity"]
        question = "How many times did they " + activity + "?"
        prompt = f""" 
            You're watching a video in which people may perform a certain type of action repetively. 
            The person performing this kind of action are referred to as 'they' in the following statement.
            You're task is to count how many times have different people in the video perform this kind of action in total.
            One complete motion counts as one. 
            Now, answer the following question: {question}
            Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
            Do not include any additional text or explanation in your response.
        """
    elif task == "SSR":
        step = _anno_["test_info"][index]["step"]
        prompt = f"""
            You're watching a tutorial video which contain a sequential of steps. 
            The following is one step from the whole procedures: 
            {step}
            Your task is to determine if the man or woman in the video is currently performing this step.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """

    elif task == "CRR":
        question = _anno_["question"]
        answer = _anno_["answer"]
        prompt = f"""
            You're responsible of answering questions based on the video content. 
            The following question are relevant to the latest frames, i.e. the end of the video.
            {question}
            Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """
    return prompt


def load_task_data(task_json_path, selected_tasks):
    with open(task_json_path, 'r') as f:
        task_list = json.load(f)
    filtered_tasks = [item for item in task_list if item['task'] in selected_tasks]
    return filtered_tasks


def distribute_samples(samples, num_gpus):
    """Distribute samples across GPUs by sample count."""
    items = []
    for item in samples:
        task = item['task']
        item_id = item['id']
        if task in ["REC", "SSR", "CRR"]:
            sample_count = len(item.get('test_info', []))
        else:
            sample_count = 1
        items.append({
            'id': item_id,
            'task': task,
            'sample_count': sample_count,
            'is_forward': task in ["REC", "SSR", "CRR"],
        })

    items.sort(key=lambda x: x['sample_count'], reverse=True)

    gpu_assignments = [[] for _ in range(num_gpus)]
    gpu_sample_counts = [0] * num_gpus

    for sample in items:
        min_idx = gpu_sample_counts.index(min(gpu_sample_counts))
        gpu_assignments[min_idx].append(sample)
        gpu_sample_counts[min_idx] += sample['sample_count']

    print("\nSample assignment:")
    total_samples = sum(gpu_sample_counts)
    for i, (samples_on_gpu, count) in enumerate(zip(gpu_assignments, gpu_sample_counts)):
        task_counts = defaultdict(int)
        id_counts = defaultdict(int)
        for s in samples_on_gpu:
            task_counts[s['task']] += 1
            id_counts[s['task']] += s['sample_count']
        task_str = ', '.join(f"{task}:{count}({id_counts[task]} samples)" for task, count in sorted(task_counts.items()))
        print(f"GPU {i}: {len(samples_on_gpu)} IDs, {count} total samples - {task_str}")
    if gpu_sample_counts and max(gpu_sample_counts) > 0:
        min_count = min(gpu_sample_counts) if min(gpu_sample_counts) > 0 else 1
        balance_ratio = max(gpu_sample_counts) / min_count
        print(f"\nBalance ratio: {balance_ratio:.2f} (closer to 1 is better)")
        print(f"Avg samples per GPU: {total_samples/num_gpus:.1f}")
    return gpu_assignments


def _list_jsonl_outputs(result_dir: Path, run_name: str):
    """List valid evaluation JSONL outputs for a given run."""
    output_dir = Path(result_dir) / "outputs"
    if not output_dir.exists():
        return []
    return [
        p for p in output_dir.glob("*.jsonl")
        if p.is_file()
        and p.name.startswith(run_name)
        and "token_drop_stats" not in p.name
        and "token_memory_stats" not in p.name
    ]


def get_response(
    prompt,
    video_path,
    model,
    processor,
    video_start=None,
    video_end=None,
    anchor_end=False,
    sample_id=None,
    use_fluxmem: bool = False,
    short_frames: int = 8,
    medium_frames: int = 16,
    save_path: str | None = None,
    video_chunk_t: int | None = None,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "min_frames": MIN_FRAMES,
                    "max_frames": MAX_FRAMES,
                    "fps": FPS,
                    "anchor_end": anchor_end,
                    "sample_id": sample_id,
                    "video_start": video_start,
                    "video_end": video_end,
                    # "nframes": NFRAMES,
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(torch.device('cuda'))
    generate_kwargs = dict(max_new_tokens=128)
    if use_fluxmem:
        generate_kwargs.update(use_fluxmem=True, short_frames=short_frames, medium_frames=medium_frames)
        if save_path is not None:
            generate_kwargs["save_path"] = save_path
    if video_chunk_t is not None:
        generate_kwargs["video_chunk_t"] = int(video_chunk_t)
    generated_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]
    return response

### Main script
def run_single(args):
    global RUN_NAME, CKPT_PATH, RESULT_DIR, TASK_JSON, VIDEO_DIR
    global LOG_PATH, OUTPUT_JSONL, MIN_PIXELS, MAX_PIXELS, MIN_FRAMES, MAX_FRAMES, FPS

    RUN_NAME = args.run_name
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TASK_JSON = args.task_json
    VIDEO_DIR = args.video_dir
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MIN_FRAMES = args.min_frames
    MAX_FRAMES = args.max_frames
    FPS = args.fps

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'outputs'), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, 'log'), exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = args.log_path or osp.join(RESULT_DIR, "log", f"{RUN_NAME}_{timestamp}.log")
    OUTPUT_JSONL = args.output_jsonl or osp.join(RESULT_DIR, "outputs", f"{RUN_NAME}_{timestamp}.jsonl")

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    lib_logger = logging.getLogger("qwen_vl_utils")
    lib_logger.setLevel(logging.INFO)
    lib_logger.addHandler(file_handler)
    lib_logger.propagate = False
    logger.propagate = False

    logger.info(f"Running {RUN_NAME} on OVO-Bench")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"Task json: {TASK_JSON}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Output jsonl: {OUTPUT_JSONL}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Max frames: {MAX_FRAMES}")
    logger.info(f"Min frames: {MIN_FRAMES}")
    logger.info(f"Time window size: {args.time_window_size}")
    logger.info(f"FPS: {FPS}")
    logger.info(f"Video chunk T: {args.video_chunk_t}")

    logger.info("=" * 50)
    logger.info("Memory Configuration:")
    logger.info(f"  Enable Memory: {args.use_fluxmem}")
    if args.use_fluxmem:
        logger.info(f"  Short Frames: {args.short_frames}")
        logger.info(f"  Medium Frames: {args.medium_frames}")
    else:
        logger.info("  Memory mechanism is disabled")
    logger.info(f"  Statistics Path: {args.save_path}")
    logger.info("=" * 50)

    torch.manual_seed(1234)
    logger.info("Set manual seed to 1234")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    original_tokenizer_max_len = processor.tokenizer.model_max_length
    processor.tokenizer.model_max_length = 32768
    logger.info(f"Expanded tokenizer model_max_length from {original_tokenizer_max_len} to {processor.tokenizer.model_max_length}")
    logger.info(f"Load model and processor from {CKPT_PATH}")

    with open(TASK_JSON, 'r') as f:
        task_list = json.load(f)

    start_time = time.time()
    filtered_task_list = [item for item in task_list if item['task'] in args.task]

    if args.sample_ids:
        sample_ids_set = set(int(sid) for sid in args.sample_ids)
        filtered_task_list = [item for item in filtered_task_list if item['id'] in sample_ids_set]
        logger.info(f"Processing {len(filtered_task_list)} samples with IDs: {args.sample_ids}")
    else:
        logger.info(f"Processing all {len(filtered_task_list)} samples in tasks: {args.task}")

    for item in tqdm(filtered_task_list):
        if item['task'] in backward_tasks or item['task'] in realtime_tasks:
            id, video, task, question, options, realtime, gt = \
                item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], item['gt']
            prompt = build_prompt(
                task=task,
                question=question,
                options=options,
                _anno_=None,
                index=None,
            )

            chunk_video_path = osp.join(VIDEO_DIR, "chunked_videos", f"{id}.mp4")
            if args.time_window_size is not None:
                clip = VideoFileClip(chunk_video_path)
                video_duration = clip.duration
                clip.close()
                video_start = max(0, video_duration - args.time_window_size)
                video_end = None
            else:
                video_start = None
                video_end = None
            response = get_response(
                prompt=prompt,
                video_path=chunk_video_path,
                model=model,
                processor=processor,
                video_start=video_start,
                video_end=video_end,
                anchor_end=(args.time_window_size is not None),
                sample_id=id,
                use_fluxmem=args.use_fluxmem,
                short_frames=args.short_frames,
                medium_frames=args.medium_frames,
                save_path=args.save_path,
                video_chunk_t=args.video_chunk_t,
            )

            output_dict = {
                'id': id,
                'video': video,
                'task': task,
                'question': question,
                'response': response,
                'ground_truth': chr(65 + gt),
            }

        elif item['task'] in forward_tasks:
            id, video, task, test_info = \
                item['id'], item['video'], item['task'], item['test_info']
            for i in range(len(test_info)):
                prompt = build_prompt(
                    task=task,
                    question=None,
                    options=None,
                    _anno_=item,
                    index=i,
                )
                chunk_video_path = osp.join(VIDEO_DIR, "chunked_videos", f"{id}_{i}.mp4")

                if args.time_window_size is not None:
                    clip = VideoFileClip(chunk_video_path)
                    video_duration = clip.duration
                    clip.close()
                    video_start = max(0, video_duration - args.time_window_size)
                    video_end = None
                else:
                    video_start = None
                    video_end = None
                response = get_response(
                    prompt=prompt,
                    video_path=chunk_video_path,
                    model=model,
                    processor=processor,
                    video_start=video_start,
                    video_end=video_end,
                    anchor_end=(args.time_window_size is not None),
                    sample_id=id,
                    use_fluxmem=args.use_fluxmem,
                    short_frames=args.short_frames,
                    medium_frames=args.medium_frames,
                    save_path=args.save_path,
                    video_chunk_t=args.video_chunk_t,
                )
                item['test_info'][i]['response'] = response

            output_dict = item
        else:
            continue

        with open(OUTPUT_JSONL, 'a' if osp.exists(OUTPUT_JSONL) else 'w') as f:
            f.write(json.dumps(output_dict) + '\n')

    end_time = time.time()
    cost_time = int(end_time - start_time)
    logger.info(f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")
    jsonl_files = _list_jsonl_outputs(RESULT_DIR, RUN_NAME)
    if not jsonl_files:
        logger.warning("No JSONL outputs found, skipping scoring.")
        return 1

    score_cmd = [
        "python", "evaluation/ovobench/score.py",
        "--result_dir", RESULT_DIR,
        "--run_name", RUN_NAME,
    ]
    score_res = subprocess.run(score_cmd, env=os.environ.copy())
    if score_res.returncode != 0:
        logger.error(f"Scoring failed with exit code {score_res.returncode}")
        return score_res.returncode
    logger.info("Scoring completed.")
    return 0


def run_multi_gpu(args):
    result_dir = Path(args.result_dir)
    (result_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (result_dir / "log").mkdir(exist_ok=True)

    samples = load_task_data(args.task_json, args.task)
    gpu_assignments = distribute_samples(samples, args.num_gpus)

    if args.dry_run:
        print("\nDry run: not executing workers")
        for gpu_id, items in enumerate(gpu_assignments):
            if items:
                sample_ids = [s['id'] for s in items]
                msg = f"GPU {gpu_id} sample IDs: {sample_ids[:5]}..." if len(sample_ids) > 5 else f"GPU {gpu_id} sample IDs: {sample_ids}"
                print(msg)
        return 0

    def launch_worker(gpu_id, gpu_samples):
        if not gpu_samples:
            return True
        sample_ids = sorted({str(s['id']) for s in gpu_samples})
        gpu_tasks = sorted({s['task'] for s in gpu_samples})

        log_path = result_dir / "log" / f"{args.run_name}_gpu{gpu_id}.log"
        output_jsonl = result_dir / "outputs" / f"{args.run_name}_gpu{gpu_id}.jsonl"

        cmd = [
            "python", "evaluation/ovobench/ovobench.py",
            "--run_name", args.run_name,
            "--task", *gpu_tasks,
            "--sample_ids", *sample_ids,
            "--ckpt_path", args.ckpt_path,
            "--task_json", args.task_json,
            "--video_dir", args.video_dir,
            "--min_pixels", str(args.min_pixels),
            "--max_pixels", str(args.max_pixels),
            "--min_frames", str(args.min_frames),
            "--max_frames", str(args.max_frames),
            "--fps", str(args.fps),
            "--result_dir", args.result_dir,
            "--log_path", str(log_path),
            "--output_jsonl", str(output_jsonl),
        ]

        if args.video_chunk_t is not None:
            cmd += ["--video_chunk_t", str(args.video_chunk_t)]
        if args.use_fluxmem:
            cmd.append("--use_fluxmem")
            cmd += ["--short_frames", str(args.short_frames)]
            cmd += ["--medium_frames", str(args.medium_frames)]
            if args.save_path:
                cmd += ["--save_path", args.save_path]
        if args.time_window_size is not None:
            cmd += ["--time_window_size", str(args.time_window_size)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - GPU {gpu_id}: {len(sample_ids)} IDs, tasks {gpu_tasks}")
        result = subprocess.run(cmd, env=env)
        return result.returncode == 0

    print("Starting parallel evaluation...")
    with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = [executor.submit(launch_worker, gpu_id, gpu_samples)
                   for gpu_id, gpu_samples in enumerate(gpu_assignments)]
        results = [f.result() for f in futures]

    all_ok = bool(results) and all(results)
    jsonl_files = _list_jsonl_outputs(result_dir, args.run_name)
    if not jsonl_files:
        print("No output JSONL found, skip scoring")
        return 1

    if not all_ok:
        print("Some GPU workers failed; attempting to score completed results")
    else:
        print("========== Evaluation finished ==========")
        print(f"========== Scoring {args.run_name} ==========")

    score_cmd = [
        "python", "evaluation/ovobench/score.py",
        "--result_dir", args.result_dir,
        "--run_name", args.run_name,
    ]
    score_res = subprocess.run(score_cmd, env=os.environ.copy())
    if score_res.returncode != 0:
        print(f"Scoring failed with exit code {score_res.returncode}")
        return score_res.returncode
    print("========== Scoring complete ==========")
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--task", type=str, nargs="+", choices=TASK_NAME, default=TASK_NAME)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--sample_ids", type=str, nargs="+", default=None,
                        help="Specific sample IDs to process. If not provided, process all samples in the specified tasks.")
    parser.add_argument("--time_window_size", type=float, default=None,
                        help="Time window size in seconds from video end. None means use full video.")
    parser.add_argument("--video_chunk_t", type=int, default=None,
                        help="Explicit video chunk size (frames) to avoid OOM; None disables chunking.")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU balanced mode.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for multi-GPU mode.")
    parser.add_argument("--dry_run", action="store_true", help="Show task split without running workers (multi_gpu mode).")
    parser.add_argument("--use_fluxmem", action="store_true", default=False,
                        help="Enable memory mechanism for video processing")
    parser.add_argument("--short_frames", type=int, default=8,
                        help="Number of short-term frames (full tokens)")
    parser.add_argument("--medium_frames", type=int, default=16,
                        help="Mid-term queue upper bound")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save memory statistics (jsonl format)")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.result_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.result_dir = f"eval_results/ovobench/{args.run_name}_{ts}"
    if args.multi_gpu:
        exit_code = run_multi_gpu(args)
    else:
        exit_code = run_single(args)
    sys.exit(exit_code if exit_code is not None else 0)
