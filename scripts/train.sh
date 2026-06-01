#!/bin/bash
cd /app
export PYTHONPATH=/app

LOG_DIR="src/testing"

# This script trains the model using the generated dataset.
echo "Starting model training..."

# # Run the Python script to train the model
# echo "Training the EfficientNet model with seed 12345..."
# python src/training/train_efficient.py --random_seed 12345 2>&1 | tee "$LOG_DIR/seed12345_efficientnet_8s_focal_loss_train_replicability_v1.txt"

# # Run the Python script to train the model
# echo "Training the EfficientNet model with seed 12345..."
# python src/training/train_efficient.py --random_seed 12345 2>&1 | tee "$LOG_DIR/seed12345_efficientnet_8s_focal_loss_train_replicability_v2.txt"

# Now, do a loop to change the random seed and train the model multiple times, incrementing 
# one number each time, such as 123, 1234, 12345, etc. Starting from 123456 and increment 10 times.
for i in {1..15}
do
    SEED=$((123 * 10**i))
    echo "Training the EfficientNet model with seed $SEED..."
    python src/training/train_efficient.py --random_seed $SEED 2>&1 | tee "$LOG_DIR/seed${SEED}_efficientnet_8s_focal_loss_train_replicability.txt"
done

echo "Model training complete!"