#!/bin/bash

# ================================================
# Usage:
#   bash advllm_selftuning.sh <start_idx> <end_idx> <model_name>
#
# Example:
#   bash advllm_selftuning.sh 0 5 vicuna
# This will run iterations 0 to 4 (5 not included) for model "vicuna".
#
# Notes:
# - This script is designed for 8 A100 GPUs (80GB).
# - If you have fewer GPUs, modify --num_processes in the accelerate launch command.
#   For example, change '--num_processes 8' to '--num_processes 4' if using 4 GPUs.
#
# Supported model names: vicuna, guanaco, mistral, phi3, llama2, llama3
# ================================================

if [ $# -ne 3 ]; then
    echo "Error: Incorrect number of arguments"
    echo "Usage: bash advllm_selftuning.sh <start_idx> <end_idx> <model_name>"
    exit 1
fi

export start_i=$1
export end_i=$2
export models=$3

# Trap Ctrl+C (SIGINT) and kill child processes
trap "echo 'Interrupted! Killing processes...'; pkill -P $$; exit 1" SIGINT

for ((i=start_i; i<end_i; i++)); do
    LOGFILE="advllm_${models}_${i}.log"

    echo "=== Iteration $i | Model: $models ==="
    echo accelerate launch --num_processes 8 adv_llm/suffix_sampling.py --current_iteration "$i" --model "$models" --target_models "$models" > "$LOGFILE" 2>&1'
    python adv_llm/knowledge_updating.py --current_iteration "$i" --model "$models" --target_models "$models" >> "$LOGFILE" 2>&1
done
