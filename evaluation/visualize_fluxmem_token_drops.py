#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from qwen_vl_utils_fluxmem import process_vision_info
from qwen2_5_vl_fluxmem import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

def build_inputs(processor, args, device):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(args.video_path),
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                    "min_frames": args.min_frames,
                    "max_frames": args.max_frames,
                    "fps": args.fps,
                    "video_start": args.video_start,
                    "video_end": args.video_end,
                },
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(device)
    frames = video_inputs[0]
    if isinstance(frames, torch.Tensor):
        frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    else:
        raise TypeError("Expected tensor video frames from process_vision_info.")
    return inputs, frames


def load_records(jsonl_path, batch_idx):
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if int(record["batch_idx"]) == batch_idx:
                records.append(record)
    return sorted(records, key=lambda record: int(record["frame_idx"]))

def render_mask(frame, coords, grid_hw, alpha):
    if str(frame.dtype) != "uint8":
        frame = frame.clip(0, 255).astype("uint8")
    image = Image.fromarray(frame)
    if not coords or grid_hw is None:
        return image

    grid_h, grid_w = grid_hw
    cell_w = image.size[0] / grid_w
    cell_h = image.size[1] / grid_h
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for h, w in sorted({tuple(coord) for coord in coords}):
        x0 = int(w * cell_w)
        y0 = int(h * cell_h)
        x1 = int((w + 1) * cell_w)
        y1 = int((h + 1) * cell_h)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, alpha))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def generate_drop_jsonl(args, drop_jsonl_path):
    model_kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"
        model_kwargs["device_map"] = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.ckpt_path, **model_kwargs).eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model.model.fluxmem.drop_vis_path = str(drop_jsonl_path)

    inputs, frames = build_inputs(processor, args, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_fluxmem": True,
        "short_frames": args.short_frames,
        "medium_frames": args.medium_frames,
    }
    if args.pair_sim_threshold is not None:
        generate_kwargs["pair_sim_threshold"] = args.pair_sim_threshold

    with torch.no_grad():
        model.generate(**inputs, **generate_kwargs)
    return frames


def render_video_frames(args, drop_jsonl_path, frames, frame_dir):
    records = load_records(drop_jsonl_path, args.batch_idx)
    if not records:
        raise ValueError("No drop records were written.")

    frame_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        start = int(record["frame_idx"]) * args.temporal_patch_size
        stop = min(start + args.temporal_patch_size, len(frames))
        for sample_idx in range(start, stop):
            rendered = render_mask(
                frame=frames[sample_idx],
                coords=record.get("final_drop", []),
                grid_hw=record.get("grid_hw"),
                alpha=args.alpha,
            )
            rendered.save(frame_dir / f"frame_{sample_idx:04d}.jpg")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize FluxMem final token drops on sampled video frames.")
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Describe the video briefly.")
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--video_start", type=float, default=None)
    parser.add_argument("--video_end", type=float, default=None)
    parser.add_argument("--short_frames", type=int, default=8)
    parser.add_argument("--medium_frames", type=int, default=64)
    parser.add_argument("--pair_sim_threshold", type=float, default=None)
    parser.add_argument("--temporal_patch_size", type=int, default=2)
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=210)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video_path.stem
    drop_jsonl_path = args.output_dir / f"{stem}_fluxmem_drop.jsonl"
    frame_dir = args.output_dir / f"{stem}_drop_frames"

    if drop_jsonl_path.exists():
        drop_jsonl_path.unlink()

    frames = generate_drop_jsonl(args, drop_jsonl_path)
    render_video_frames(args, drop_jsonl_path, frames, frame_dir)
    print(f"drop_jsonl: {drop_jsonl_path}")
    print(f"frames_dir: {frame_dir}")


if __name__ == "__main__":
    main()
