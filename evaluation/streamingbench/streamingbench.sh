#!/bin/bash
trap "kill 0" SIGINT

TASK_CSV="StreamingBench-csv/Real_Time_Visual_Understanding.csv"
VIDEO_DIR="StreamingBench/data/real"
CKPT_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
MAX_PIXELS=401408          # 512*28*28
MAX_NUM_FRAMES=256
FPS=1
TIMEWINDOW=256

# Multi-GPU
MULTI_GPU=true
NUM_GPUS=8

# FluxMem toggles
USE_FLUXMEM=true
SHORT_FRAMES=8
MEDIUM_FRAMES=64


# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-gpu) MULTI_GPU=true; shift ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --task-csv) TASK_CSV="$2"; shift 2 ;;
        --video-dir) VIDEO_DIR="$2"; shift 2 ;;
        --ckpt-path) CKPT_PATH="$2"; shift 2 ;;
        --max-pixels) MAX_PIXELS="$2"; shift 2 ;;
        --max-num-frames) MAX_NUM_FRAMES="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --timewindow) TIMEWINDOW="$2"; shift 2 ;;
        --save-path) SAVE_PATH="$2"; shift 2 ;;
        # Memory flags
        --use-memory) USE_FLUXMEM=true; shift ;;
        --short-frames) SHORT_FRAMES="$2"; shift 2 ;;
        --medium-frames) MEDIUM_FRAMES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="eval_results/streamingbench_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"/{log,output}

SAVE_PATH="$RESULT_DIR/log/memory_stats.jsonl"

echo "StreamingBench Evaluation (Memory)"
echo "================================="
echo "Model: $CKPT_PATH"
echo "Mode: $([ "$MULTI_GPU" = true ] && echo "Multi-GPU ($NUM_GPUS GPUs)" || echo "Single-GPU")"
echo "Output: $RESULT_DIR"
echo "Memory: USE_FLUXMEM=$USE_FLUXMEM SHORT_FRAMES=$SHORT_FRAMES MEDIUM_FRAMES=$MEDIUM_FRAMES MAX_PIXELS=$MAX_PIXELS MAX_NUM_FRAMES=$MAX_NUM_FRAMES FPS=$FPS TIMEWINDOW=${TIMEWINDOW:-none}"
if [ -n "$PAIR_SIM_THRESHOLD" ]; then
  echo "Pair similarity threshold provided (bypass Otsu): $PAIR_SIM_THRESHOLD"
fi
if [ -n "$SAVE_PATH" ]; then
    echo "Will save memory stats to: $SAVE_PATH"
fi

# Build common memory args
MEMORY_ARGS=( )
if [ "$USE_FLUXMEM" = true ]; then MEMORY_ARGS+=("--use_fluxmem"); fi
MEMORY_ARGS+=("--short_frames" "$SHORT_FRAMES" "--medium_frames" "$MEDIUM_FRAMES")

# Video sampling/processing args to forward
VIDEO_ARGS=("--max-pixels" "$MAX_PIXELS" "--max-num-frames" "$MAX_NUM_FRAMES" "--fps" "$FPS")
# Pass time window (in seconds) to python as --time_window_size if provided
if [ -n "$TIMEWINDOW" ]; then
    VIDEO_ARGS+=("--time_window_size" "$TIMEWINDOW")
fi
# If fixed threshold is provided, forward to Python to bypass Otsu
if [ -n "$PAIR_SIM_THRESHOLD" ]; then
    VIDEO_ARGS+=("--pair_sim_threshold" "$PAIR_SIM_THRESHOLD")
fi
if [ -n "$SAVE_PATH" ]; then
    VIDEO_ARGS+=("--save_path" "$SAVE_PATH")
fi

if [ "$MULTI_GPU" = true ]; then
    python evaluation/streamingbench/streamingbench.py \
        --ckpt_path "$CKPT_PATH" \
        --task_csv "$TASK_CSV" \
        --video_dir "$VIDEO_DIR" \
        --result_dir "$RESULT_DIR" \
        --run_name "streamingbench" \
        --multi_gpu \
        --num_gpus "$NUM_GPUS" \
        "${MEMORY_ARGS[@]}" \
        "${VIDEO_ARGS[@]}"
else
    OUTPUT_JSONL="$RESULT_DIR/output/results_${TIMESTAMP}.jsonl"
    LOG_PATH="$RESULT_DIR/log/eval_${TIMESTAMP}.log"

    python evaluation/streamingbench/streamingbench.py \
        --ckpt_path "$CKPT_PATH" \
        --task_csv "$TASK_CSV" \
        --video_dir "$VIDEO_DIR" \
        --output_jsonl "$OUTPUT_JSONL" \
        --log_path "$LOG_PATH" \
        --result_dir "$RESULT_DIR" \
        --run_name "streamingbench" \
        "${MEMORY_ARGS[@]}" \
        "${VIDEO_ARGS[@]}"
fi

PYTHON_EXIT=$?

# Run scoring if we have any JSONL results; warn if the evaluation exited non‑zero.
if find "$RESULT_DIR" -maxdepth 2 -name '*.jsonl' | grep -q .; then
    if [ $PYTHON_EXIT -ne 0 ]; then
        echo "Warning: evaluation exited with code $PYTHON_EXIT; attempting to score existing results..."
    else
        echo "Running scoring..."
    fi
    python evaluation/streamingbench/score.py \
        --result_dir "$RESULT_DIR" \
        --model_name "$(basename "$CKPT_PATH")"
else
    echo "No JSONL results found in $RESULT_DIR; skipping scoring."
fi

echo "Evaluation complete. Results in: $RESULT_DIR"
