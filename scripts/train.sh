#!/bin/bash
cd /app
export PYTHONPATH=/app

LOG_DIR="src/testing"

# This script trains the model using the generated dataset.
echo "Starting model training..."

# Run the Python script to train the model
echo "Training the EfficientNet model with seed 12345..."
python src/training/train_efficient.py --random_seed 12345 2>&1 | tee "$LOG_DIR/seed12345_efficientnet_8s_focal_loss_train.txt"

echo "Model training complete!"