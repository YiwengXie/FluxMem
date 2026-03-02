import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from qwen2_5_vl_fluxmem import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from qwen_vl_utils_fluxmem import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils_fluxmem; install via `pip install qwen-vl-utils-fluxmem`")


@register_model("qwen2_5_vl_fluxmem")
class Qwen2_5_VL_FluxMem(lmms):
    """Qwen2.5-VL-FluxMem model wrapper for lmms-eval."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = True,
        min_pixels: int = 16 * 28 * 28,
        max_pixels: int = 256 * 28 * 28,
        max_num_frames: int = 128,
        fps: int = 1,
        # Memory-specific flags
        use_fluxmem: bool = True,
        frame_sampling: str = "uniform",
        short_frames: int = 8,
        medium_frames: int = 16,
        save_path: Optional[str] = None,
        # Optional time window (seconds) for video clipping
        clip_start_sec: Optional[float] = None,
        clip_duration_sec: Optional[float] = None,
        # Optional tail window (seconds): keep only the last N seconds
        tail_window_sec: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Store memory flags
        self.mem_use_fluxmem = bool(use_fluxmem)
        self.mem_frame_sampling = frame_sampling
        self.mem_short_frames = int(short_frames)
        self.mem_medium_frames = int(medium_frames)
        self.fps = int(fps)

        self.max_num_frames = int(max_num_frames)
        self.max_pixels = int(max_pixels)
        self.min_pixels = int(min_pixels)
        # Store optional clip/tail window
        self.clip_start_sec = None if clip_start_sec is None else float(clip_start_sec)
        self.clip_duration_sec = None if clip_duration_sec is None else float(clip_duration_sec)
        self.tail_window_sec = None if tail_window_sec is None else float(tail_window_sec)

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        if use_flash_attention_2:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map).eval()

        self.processor = Qwen2_5_VLProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.mem_save_path = save_path

        # Shallow routing removed from wrapper

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError("`until` must be str or list")
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": "You are a helpful assistant."}]
                processed_visuals = []
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        # Video: keep dynamic sampling; lmms-eval task loader provides path
                        vid_dict = {
                            "type": "video",
                            "video": visual,
                            "max_pixels": self.max_pixels,
                            "min_pixels": self.min_pixels,
                            "max_frames": self.max_num_frames,
                            "fps": self.fps,
                        }
                        # Apply head clip window only if tail window is not requested
                        if self.tail_window_sec is None and self.clip_start_sec is not None:
                            vid_dict["video_start"] = self.clip_start_sec
                            if self.clip_duration_sec is not None:
                                vid_dict["video_end"] = self.clip_start_sec + self.clip_duration_sec
                        processed_visuals.append(vid_dict)
                    elif isinstance(visual, Image.Image):
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        processed_visuals.append({
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{base64_string}",
                            "max_pixels": self.max_pixels,
                            "min_pixels": self.min_pixels,
                        })
                message.append({"role": "user", "content": processed_visuals + [{"type": "text", "text": context}]})
                batched_messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs, video_kwargs = process_vision_info(batched_messages, return_video_kwargs=True)
            video_fps = None
            if video_inputs is not None:
                # Tail window: keep only the last N seconds of frames (if requested)
                if self.tail_window_sec is not None and 'fps' in video_kwargs and len(video_kwargs['fps']) > 0:
                    fps_eff = float(video_kwargs['fps'][0])
                    keep_frames = max(1, int(round(fps_eff * self.tail_window_sec)))
                    total_frames_before = int(video_inputs[0].shape[0])
                    start_idx = max(0, total_frames_before - keep_frames)
                    video_inputs[0] = video_inputs[0][start_idx:]

                # Then uniformly sample up to max_num_frames, always include the last frame
                total_frames = int(video_inputs[0].shape[0])
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
                if 'fps' in video_kwargs and len(video_kwargs['fps']) > 0 and total_frames > 0:
                    original_fps = float(video_kwargs['fps'][0])
                    video_fps = original_fps * (len(indices) / max(1, total_frames))
            if video_fps is not None:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    fps=video_fps,
                )
            else:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if current_gen_kwargs["temperature"] > 0 else False,
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
                # Memory-specific args forwarded to model.forward via generate
                use_fluxmem=self.mem_use_fluxmem,
                memory_drop_method=self.mem_frame_sampling,
                short_frames=self.mem_short_frames,
                medium_frames=self.mem_medium_frames,
                save_path=self.mem_save_path,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        """Satisfy abstract interface; multi-round not used here."""
        raise NotImplementedError("generate_until_multi_round is not implemented for Qwen2.5-VL-Memory")
