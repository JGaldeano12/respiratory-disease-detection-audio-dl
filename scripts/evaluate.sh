#!/bin/bash
cd /app || exit 1
export PYTHONPATH=/app
export TF_CPP_MIN_LOG_LEVEL=1

# Path to the model to evaluate. Defaults to the best Custom CNN checkpoint.
MODEL_PATH="${1:-models/CustomCNN/best_model_CustomCNN.keras}"

echo "Starting evaluation of $MODEL_PATH..."

python src/evaluation/evaluate.py \
    --model_path "$MODEL_PATH" \
    --data_dir data/processed \
    --output_dir experiments

echo "Model evaluation complete!"