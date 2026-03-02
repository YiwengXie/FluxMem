#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# Paths
DEEPSPEED_CONFIG=./scripts/zero3.json
ENTRY_FILE=qwenvl/train/train_qwen.py

# Model & data
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
DATASETS="your_dataset1,your_dataset2"

# Output
RUN_NAME="qwen2.5vl-fluxmem"
OUTPUT_DIR="./output/${RUN_NAME}"

# Training hyperparams
LEARNING_RATE=1e-5
MLP_LR=1e-5
BATCH_SIZE=1
GRAD_ACCUM_STEPS=16
NUM_TRAIN_EPOCHS=1
VIDEO_MIN_FRAMES=2
VIDEO_MIN_FRAME_PIXELS=12544   # 16*28*28
VIDEO_MAX_FRAME_PIXELS=$((256*28*28))
VIDEO_MAX_FRAMES=256
BASE_INTERVAL=1
VIDEO_CONTEXT_TIME=128

# FluxMem settings
FRAME_SAMPLING="uniform"
SHORT_FRAMES=8
MEDIUM_FRAMES=64

ARGS="
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_use ${DATASETS} \
    --data_flatten True \
    --data_packing False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size $((BATCH_SIZE*2)) \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --model_max_length 33792 \
    --video_context_time ${VIDEO_CONTEXT_TIME} \
    --base_interval ${BASE_INTERVAL} \
    --video_max_frames ${VIDEO_MAX_FRAMES} \
    --video_min_frames ${VIDEO_MIN_FRAMES} \
    --video_max_frame_pixels ${VIDEO_MAX_FRAME_PIXELS} \
    --video_min_frame_pixels ${VIDEO_MIN_FRAME_PIXELS} \
    --eval_strategy \"no\" \
    --save_strategy \"steps\" \
    --save_steps 500 \
    --save_total_limit 1 \
    --optim adamw_torch \
    --learning_rate ${LEARNING_RATE} \
    --mm_projector_lr ${MLP_LR} \
    --vision_tower_lr 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type \"cosine\" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${RUN_NAME} \
    --use_fluxmem \
    --frame_sampling ${FRAME_SAMPLING} \
    --short_frames ${SHORT_FRAMES} \
    --medium_frames ${MEDIUM_FRAMES}"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} ${ARGS}
