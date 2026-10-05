#!/bin/bash
cd /app
export PYTHONPATH=/app
export TF_CPP_MIN_LOG_LEVEL=1

LOG_DIR="experiments"

echo "========== SEED 1234 =========="
echo "Training the Custom CNN model with changing seed..."
python src/training/train_custom_cnn.py --random_seed 1234 2>&1 | tee "$LOG_DIR/training_custom_cnn.txt"

echo "Training complete for seed 1234! Check the results to find the best performance."
echo "Model training complete!"