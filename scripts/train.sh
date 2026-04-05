#!/bin/bash

# This script trains the model using the generated dataset.
echo "Starting model training..."

# Run the Python script to train the model
python src/training/train.py \
    --dataset_dir data/processed \
    --output_dir models/checkpoints \
    --config configs/training_config.yaml

echo "Model training complete!"