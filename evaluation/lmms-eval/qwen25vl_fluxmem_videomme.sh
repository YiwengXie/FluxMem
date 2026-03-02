#!/usr/bin/env bash
# FluxMem + lmms-eval: VideoMME

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12346
export DECORD_EOF_RETRY_MAX=20480

PRETRAINED="Qwen/Qwen2.5-VL-7B-Instruct"
MAX_PIXELS=100352
MAX_NUM_FRAMES=1024
FPS=1
USE_FLUXMEM=true
SHORT_FRAMES=8
MEDIUM_FRAMES=512
SAVE_PATH="eval_results/videomme/memory_stats.jsonl"

MODEL_ARGS_ARR=(
  "pretrained=${PRETRAINED}"
  "max_pixels=${MAX_PIXELS}"
  "max_num_frames=${MAX_NUM_FRAMES}"
  "fps=${FPS}"
  "use_fluxmem=${USE_FLUXMEM}"
  "short_frames=${SHORT_FRAMES}"
  "medium_frames=${MEDIUM_FRAMES}"
  "use_flash_attention_2=True"
  "save_path=${SAVE_PATH}"
)
MODEL_ARGS=$(IFS=, ; echo "${MODEL_ARGS_ARR[*]}")

accelerate launch --num_processes=8 --main_process_port=${MASTER_PORT} -m lmms_eval \
    --model qwen2_5_vl_fluxmem \
    --model_args "$MODEL_ARGS" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_5_vl_fluxmem \
    --output_path ./eval_results/videomme/
