<p align="center">
  <img src="assets/images/title.png" width="450" alt="FluxMem logo">
</p>

<h1 align="center" style="font-weight:600;">
  [CVPR 26] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding
</h1>

<p align="center">
  <a href="https://YiwengXie.github.io" target="_blank">Yiweng Xie</a>,
  <a href="https://boheumd.github.io" target="_blank">Bo He</a>,
  <a href="https://wdrink.github.io" target="_blank">Junke Wang</a>,
  <a href="#" target="_blank">Xiangyu Zheng</a>,
  <a href="https://yeziyi1998.github.io" target="_blank">Ziyi Ye</a>,
  <a href="https://zxwu.azurewebsites.net" target="_blank">Zuxuan Wu</a>
</p>

<p align="center">
  🌐 <a href="https://yiwengxie.github.io/FluxMem/" target="_blank"><b>Homepage</b></a> &nbsp;|&nbsp; 
  📄 <a href="https://arxiv.org/abs/2603.02096" target="_blank"><b>Paper</b></a><br>
</p>


> FluxMem uses a training-free hierarchical memory with temporal (mid-term) and spatial (long-term) compression to adaptively prune redundant visual tokens in streaming video, enabling efficient real-time reasoning for large multimodal models.

![FluxMem teaser](assets/images/teaserfigure.png)

## <img src="assets/icons/highlights.png" width="22" alt="Highlights icon" style="vertical-align:middle; position: relative; top: -0.2em;"> Highlights
- 🧠 **Hierarchical memory**: short-term keeps the freshest frames, mid-term filters temporal redundancy, long-term further removes spatial redundancy by anchoring salient tokens.
- 🪄 **Training-free**: drop-in gains without extra finetuning; if you do fine-tune, the gap just gets wider.
- 🧩 **Plug-and-play**: slips into Qwen2.5-VL as a memory add-on—no model surgery, no code rewrites.
- ⚡ **Efficient**: trims 60–70% visual tokens while lifting performance on both online and offline long-video benchmarks.

![Framework](assets/images/framework.png)

## <img src="assets/icons/repo_layout.png" width="22" alt="Repository layout icon" style="vertical-align:middle; position: relative; top: -0.2em;"> Repository Layout
```
FluxMem
├── models/
│   ├── qwen2-5-vl/         # FluxMem-patched Qwen2.5-VL model & processor
│   └── qwen-vl-utils/      # Vision preprocessing
├── qwen-vl-finetune/       # Training pipeline, data configs, SFT scripts
├── evaluation/             # StreamingBench, OVO-Bench, lmms-eval recipes
└── assets/                 # Figures used in README
```

## <img src="assets/icons/installation.png" width="22" alt="Installation icon" style="vertical-align:middle; position: relative; top: -0.2em;"> Installation
Create venv:
```bash
uv venv --python=python3.11
source .venv/bin/activate
```

- **Inference essentials**
  ```bash
  uv pip install -e models/qwen2-5-vl
  uv pip install -e models/qwen-vl-utils
  ```
- **Training**
  ```bash
  uv pip install -e "qwen-vl-finetune[train]"
  ```
- **Evaluation**
  ```bash
  uv pip install -e evaluation/lmms-eval      # for VideoMME / MLVU / LongVideoBench
  uv pip install ffmpeg-python==0.2.0 moviepy==1.0.3   # for StreamingBench / OVO-Bench
  ```
- **Flash-Attn 2**: download the matching wheel, then
  ```bash
  uv pip install ./flash_attn-*.whl --no-build-isolation
  ```
  e.g. `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl` (CUDA 12, torch 2.6, Python 3.11).


## <img src="assets/icons/quick_start.png" width="22" alt="Quick start icon" style="vertical-align:middle; position: relative; top: -0.2em;"> Quick Start
```python
import torch

from qwen_vl_utils_fluxmem import process_vision_info
from qwen2_5_vl_fluxmem import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = Qwen2_5_VLProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
)

video_path = 'PATH_TO_YOUR_VIDEO'
prompt = 'Describe this video.'

max_pixels = 256 * 28 * 28 
fps = 1

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"video": video_path, 'fps': fps, "max_pixels": max_pixels},
    ],
}]

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
inputs = inputs.to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.0,
    use_fluxmem=True,   # enable FluxMem
    short_frames=8,
    medium_frames=64,
)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

## <img src="assets/icons/training.png" width="22" alt="Training icon" style="vertical-align:middle; position: relative; top: -0.2em;"> Training
1) Configure dataset paths in `qwen-vl-finetune/qwenvl/data/__init__.py` (`annotation_path`, `data_path`).
2) Run the default SFT script:
```bash
cd qwen-vl-finetune
bash scripts/sft.sh
```
(Adjust hyperparameters/paths in scripts as needed; deepspeed & flash-attn are supported.)

## <img src="assets/icons/evaluation.png" width="22" alt="Evaluation icon" style="vertical-align:middle; position: relative; top: -0.2em;"> Evaluation
- StreamingBench: `bash evaluation/streamingbench/streamingbench.sh`.
- OVO-Bench: `bash evaluation/ovobench/ovobench.sh`.
- VideoMME / MLVU / LongVideoBench: `bash evaluation/lmms-eval/qwen25vl_fluxmem_*.sh`.
- Datasets: You can download the evaluation benchmarks from the corresponding link: [StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench); [OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench).

## <img src="assets/icons/visualizations.png" width="22" alt="Visualizations icon" style="vertical-align:middle; position: relative; top: -0.2em;"> Visualizations
To visualize which tokens FluxMem drops on a specific video:
```bash
python evaluation/visualize_fluxmem_token_drops.py \
    --video_path PATH_TO_VIDEO \
    --output_dir vis_outputs/ \
    --ckpt_path Qwen/Qwen2.5-VL-7B-Instruct \
    --fps 1 \
    --max_frames 256 \
    --short_frames 8 \
    --medium_frames 64
```

![Token Flow Visualization 1](assets/images/visualize-1.png)
![Token Flow Visualization 2](assets/images/visualize-2.png)

## <img src="assets/icons/license.png" width="22" alt="License icon" style="vertical-align:middle; position: relative; top: -0.2em;"> License
Apache-2.0. Please also follow upstream model and dataset licenses. 

## <img src="assets/icons/citations.png" width="22" alt="Citation icon" style="vertical-align:middle; position: relative; top: -0.2em;"> Citation
If you find FluxMem useful, please cite:
```bibtex
@inproceedings{xie2026fluxmem,
  title={FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding},
  author={Xie, Yiweng and He, Bo and Wang, Junke and Zheng, Xiangyu and Ye, Ziyi and Wu, Zuxuan},
  booktitle={CVPR},
  year={2026}
}
```

## <img src="assets/icons/acknowledgements.png" width="22" alt="Acknowledgements icon" style="vertical-align:middle; position: relative; top: -0.2em;"> Acknowledgements
**We thank the following projects for their contributions and inspiration**: [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL), [TimeChat-online](https://github.com/yaolinli/TimeChat-Online), [OVOBench](https://github.com/joeleelyf/ovo-bench), [StreamingBench](https://github.com/THUNLP-MT/StreamingBench), [LMMS-Eval](https://github.com/THUDM/LMMS-Eval).
