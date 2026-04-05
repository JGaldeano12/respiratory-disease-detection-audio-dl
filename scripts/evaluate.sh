#!/bin/bash

# This script evaluates the trained model on the test dataset and generates a report.
echo "Starting model evaluation..."

# Run the Python script to evaluate the model
python src/evaluation/evaluate.py \
    --model_dir models/checkpoints \
    --test_data_dir data/test \
    --output_dir experiments

echo "Model evaluation complete!"