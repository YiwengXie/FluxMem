import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
from torchcodec.decoders import VideoDecoder
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()
        self.failed_videos = []  # Store failed video information

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file, video_time=None):
        """Process video file with automatic format detection and decoder fallback."""
        video_file = self._find_video_file(video_file)
        if not video_file:
            raise FileNotFoundError(f"Video file not found: {video_file}")
        
        # Try decord first (usually faster)
        for attempt in range(3):
            try:
                return self._read_video_decord(video_file, video_time)
            except Exception as e:
                if attempt == 2:
                    print(f"Decord failed after 3 attempts: {e}")
                time.sleep(0.5)
        
        # Fallback to torchcodec
        try:
            return self._read_video_torchcodec(video_file, video_time)
        except Exception as e:
            raise RuntimeError(f"All video decoders failed for {video_file}: {e}")
    
    def _find_video_file(self, video_file):
        """Find video file with common extensions."""
        if os.path.exists(video_file):
            return video_file
        
        base = os.path.splitext(video_file)[0]
        for ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
            path = f"{base}{ext}"
            if os.path.exists(path):
                return path
        return None
    def _calculate_frame_range(self, start_time, end_time, total_frames, fps):
        """Calculate frame range from time range."""
        if fps <= 0 or total_frames <= 0:
            raise ValueError("Invalid fps or total_frames")
        
        max_duration = total_frames / fps
        
        # Default to full video if no time specified
        if start_time is None and end_time is None:
            return 0, total_frames - 1
        
        # Clamp times to valid range
        start_time = max(0, min(start_time or 0, max_duration))
        end_time = max(0, min(end_time or max_duration, max_duration))
        
        # Convert to frames
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Ensure valid range
        start_frame = min(start_frame, total_frames - 1)
        end_frame = min(max(start_frame, end_frame), total_frames - 1)
        
        return start_frame, end_frame
     

    def _read_video_decord(self, video_file, video_time=None):
        """Read video using decord library."""
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        
        # Calculate frame range
        start_time, end_time = video_time if video_time else (None, None)
        start_frame, end_frame = self._calculate_frame_range(start_time, end_time, total_frames, fps)
        
        # Apply context time constraint if specified
        start_frame, end_frame = self._apply_context_time_constraint(
            start_frame, end_frame, fps
        )
        
        # Sample frames
        frame_indices = self._sample_frame_indices(start_frame, end_frame, fps)
        video = vr.get_batch(frame_indices).asnumpy()
        
        # Calculate region length for effective fps
        # More accurate: use actual sampled frames to calculate duration
        if len(frame_indices) > 1:
            # Duration from first to last sampled frame
            actual_duration = (frame_indices[-1] - frame_indices[0]) / fps
            # Add one frame duration to include the last frame's display time
            region_length = actual_duration + 1.0 / fps
        else:
            # Single frame: duration is one frame period
            raise ValueError("Single frame video is not supported")
        
        return self._process_video_frames(video, frame_indices, fps, region_length)

    def _read_video_torchcodec(self, video_file, video_time=None):
        """Read video using torchcodec library."""
        decoder = VideoDecoder(video_file, device="cpu")
        total_frames = decoder.metadata.num_frames
        fps = decoder.metadata.average_fps
        
        # Calculate frame range
        start_time, end_time = video_time if video_time else (None, None)
        start_frame, end_frame = self._calculate_frame_range(start_time, end_time, total_frames, fps)
        
        # Apply context time constraint if specified
        start_frame, end_frame = self._apply_context_time_constraint(
            start_frame, end_frame, fps
        )
        
        # Sample frames
        frame_indices = self._sample_frame_indices(start_frame, end_frame, fps)
        frame_batch = decoder.get_frames_at(indices=frame_indices.tolist())
        video = frame_batch.data.cpu().numpy()
        
        # Calculate region length for effective fps
        # More accurate: use actual sampled frames to calculate duration
        if len(frame_indices) > 1:
            # Duration from first to last sampled frame
            actual_duration = (frame_indices[-1] - frame_indices[0]) / fps
            # Add one frame duration to include the last frame's display time
            region_length = actual_duration + 1.0 / fps
        else:
            # Single frame: duration is one frame period
            raise ValueError("Single frame video is not supported")
        
        return self._process_video_frames(video, frame_indices, fps, region_length)

    def _apply_context_time_constraint(self, start_frame, end_frame, fps):
        """Apply context time constraint to limit video duration."""
        context_time = getattr(self.data_args, "video_context_time", None)
        if context_time is None or context_time <= 0:
            return start_frame, end_frame
        
        duration = (end_frame - start_frame + 1) / fps
        if duration > context_time:
            # Keep the end, adjust the start
            max_frames = int(context_time * fps)
            start_frame = max(start_frame, end_frame - max_frames + 1)
        
        return start_frame, end_frame
    
    def _sample_frame_indices(self, start_frame, end_frame, fps):
        """Sample frame indices based on interval and constraints."""
        duration = (end_frame - start_frame + 1) / fps
        interval = getattr(self.data_args, "base_interval", 4)
        
        # Calculate target number of frames
        num_frames = round(duration / interval)
        min_frames = getattr(self.data_args, "video_min_frames", 4)
        max_frames = getattr(self.data_args, "video_max_frames", 8)
        num_frames = max(min_frames, min(num_frames, max_frames))
        
        # Generate indices
        indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        return np.unique(indices)
    
    def _process_video_frames(self, video, frame_indices, fps, region_length):
        """Process video frames with the image processor."""
        # Configure processor for video
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        
        # Process video
        processed = processor.preprocess(images=None, videos=video, return_tensors="pt")
        video_tensor = processed["pixel_values_videos"]
        grid_thw = processed["video_grid_thw"][0]
        
        # Calculate temporal information
        # Calculate effective fps: number of sampled frames / actual duration
        effective_fps = len(frame_indices) / region_length
        temporal_patch_size = self.data_args.image_processor.temporal_patch_size
        second_per_grid = [temporal_patch_size / effective_fps] * len(grid_thw)
        
        return video_tensor, grid_thw, second_per_grid

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Try the current sample first (only 1 retry)
        for attempt_idx in range(2):  # 0 and 1, total 2 attempts
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                if attempt_idx == 0:
                    print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception: {e}")
                    time.sleep(0.5)

        # Try up to 5 subsequent samples
        max_other_samples = 50
        for offset in range(1, max_other_samples + 1):
            try:
                next_index = (i + offset) % len(self.list_data_dict)
                sample = self._get_item(next_index)
                print(f"Successfully loaded alternative sample {next_index} after sample {i} failed")
                return sample
            except Exception as e:
                print(f"[Try other sample] Failed to fetch sample {next_index}. Exception: {e}")
                continue

        # Final attempt on the original sample
        try:
            print(f"Final attempt on sample {i}")
            sample = self._get_item(i)
            return sample
        except Exception as e:
            print(f"All attempts failed. Original sample {i} error: {e}")
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        if "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
        if "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            video_time = None
            if "start_time" in self.list_data_dict[i] and "end_time" in self.list_data_dict[i]:
                video_time = [self.list_data_dict[i]["start_time"], self.list_data_dict[i]["end_time"]]
            try:
                if isinstance(video_file, List):
                    if len(video_file) > 1:
                        video_file = [
                            os.path.join(video_folder, file) for file in video_file
                        ]
                        results = [self.process_video(file, video_time) for file in video_file]
                        video, video_grid_thw, second_per_grid_ts = zip(*results)
                    else:
                        video_file = video_file[0]
                        video_file = os.path.join(video_folder, video_file)
                        video, video_grid_thw, second_per_grid_ts = self.process_video(
                            video_file, video_time
                        )
                        video = [video]
                else:
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file, video_time
                    )
                    video = [video]
            except Exception as e:
                # Record failed video info
                failed_info = {
                    "index": i,
                    "video_file": video_file,
                    "error": str(e),
                    "data_dict": self.list_data_dict[i],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.failed_videos.append(failed_info)
                print(f"Failed to process video {video_file}: {str(e)}")
                raise  # Re-raise to maintain original behavior
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass