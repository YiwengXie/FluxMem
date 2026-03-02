import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
# hf_home="/share/junjie/shuyan/lmms-eval/~/.cache/huggingface"
base_cache_dir = os.path.expanduser(hf_home)


with open(Path(__file__).parent / "mlvu_dev.yaml", "r") as f:
    raw_data_dev = f.readlines()
    safe_data_dev = []
    for i, line in enumerate(raw_data_dev):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_dev.append(line)
cache_name_dev = yaml.safe_load("".join(safe_data_dev))["dataset_kwargs"]["cache_dir"]
cache_dir_dev = os.path.join(base_cache_dir, cache_name_dev)


with open(Path(__file__).parent / "mlvu_test.yaml", "r") as f:
    raw_data_test = f.readlines()
    safe_data_test = []
    for i, line in enumerate(raw_data_test):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_test.append(line)
cache_name_test = yaml.safe_load("".join(safe_data_test))["dataset_kwargs"]["cache_dir"]
cache_dir_test = os.path.join(base_cache_dir, cache_name_test)


def mlvu_doc_to_visual_dev(doc):
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir_dev, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def mlvu_doc_to_visual_test(doc):
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir_test, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    full_prompt = pre_prompt + question + post_prompt
    return full_prompt


def extract_characters_regex(s: str) -> str:
    """Extract choice letter A-D from the model output in a robust way.

    Handles formats like:
    - "Best option: (A)"
    - "A. Delivers a product"
    - "Answer: A"
    - "(C)" or Chinese parentheses "（C）"
    Falls back to original string if nothing is found.
    """
    s = (s or "").strip()
    if not s:
        return s

    # Common patterns
    patterns = [
        r"best\s*option\s*:\s*[\(（]?\s*([A-D])\s*[\)）]?",
        r"^\s*[\(（]?\s*([A-D])\s*[\)）\.\s]",
        r"answer\s*[:：]\s*[\(（]?\s*([A-D])\s*[\)）]?",
        r"\boption\s*([A-D])\b",
    ]

    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Standalone A-D not inside a word
    m = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", s)
    if m:
        return m.group(1).upper()

    # Legacy fallback: char before first closing parenthesis
    for close_p in (")", "）"):
        if close_p in s:
            idx = s.index(close_p)
            if idx - 1 >= 0:
                return s[idx - 1 : idx].upper()
            break

    return s


def mlvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]

    pred_ans = extract_characters_regex(pred)

    task_type = doc["task_type"]
    data_dict = {"question_id": doc["question"], "task_type": task_type, "pred_answer": pred_ans, "answer": doc["answer"]}

    return {f"mlvu_percetion_score": data_dict}


def mlvu_aggregate_results_dev(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "needle", "order", "plotQA", "topic_reasoning"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    # Calculate and log accuracy for each task category
    covered_categories = []
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        if total_answered > 0:
            accuracy = 100 * total_correct / total_answered
            task_category_scores[task_cate] = accuracy
            covered_categories.append(task_cate)
            eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}% (n={total_answered})")
        else:
            # Skip categories with no samples to avoid misleading 0.0%
            eval_logger.info(f"Evaluation on Task Categories: {task_cate}: skipped (no samples in this run)")

    # Calculate and log average accuracy across covered task categories only
    if covered_categories:
        average_accuracy = sum(task_category_scores.values()) / len(covered_categories)
    else:
        average_accuracy = 0

    eval_logger.info(
        f"Average Performance Across Covered Task Categories ({len(covered_categories)}/{len(TASK_TYPES)}): {average_accuracy:.1f}%"
    )

    return average_accuracy


def mlvu_aggregate_results_test(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "needleQA", "order", "plotQA", "sportsQA", "topic_reasoning", "tutorialQA"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    # Calculate and log accuracy for each task category
    covered_categories = []
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        if total_answered > 0:
            accuracy = 100 * total_correct / total_answered
            task_category_scores[task_cate] = accuracy
            covered_categories.append(task_cate)
            eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}% (n={total_answered})")
        else:
            # Skip categories with no samples to avoid misleading 0.0%
            eval_logger.info(f"Evaluation on Task Categories: {task_cate}: skipped (no samples in this run)")

    # Calculate and log average accuracy across covered task categories only
    if covered_categories:
        average_accuracy = sum(task_category_scores.values()) / len(covered_categories)
    else:
        average_accuracy = 0

    eval_logger.info(
        f"Average Performance Across Covered Task Categories ({len(covered_categories)}/{len(TASK_TYPES)}): {average_accuracy:.1f}%"
    )

    return average_accuracy
