#!/bin/bash

# bash run_evaluation test.json none false
testset_path=${1:-test.json}
retrieval=${2:-bm25}
use_rag=${3:-True}

project_dir=$(dirname $(dirname $(pwd)))

python $project_dir/evaluation/run_evaluation_vllm.py \
    --model_path /share/luoqi/models/deepseek-coder-1.3b-instruct \
    --model_name deepseek \
    --max_new_tokens 128 \
    --testset_path $testset_path \
    --total_budget 4096 \
    --sample_num -1 \
    --use_vllm True \
    --use_rag $use_rag \
    --gpus 8 \
    --gpu_memory_utilization 0.82 \
    --temperature 0 \
    --retrieval $retrieval \
    --group_key type \
    --max_rag_num 1 \
