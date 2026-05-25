#!/bin/bash
cd /app
export PYTHONPATH=/app

LOG_DIR="src/testing"

BASE="123456789012345"

# # First, generate and augment the dataset for the first seed
# python src/data/augment.py --seed 20260119

for i in $(seq 5 15)
do
    seed="${BASE:0:$((i+2))}"
    echo "========== SEED $seed =========="

    # echo "Training the Custom CNN model with seed $seed..."
    # python src/training/train.py --random_seed $seed \
    #     2>&1 | tee "$LOG_DIR/seed${seed}_cnn_focal_loss_train.txt"
    
    echo "Training the EfficientNet model with seed $seed..."
    python src/training/train_efficient.py --random_seed $seed \
        2>&1 | tee "$LOG_DIR/seed${seed}_efficientnet_7s_focal_loss_train.txt"

    echo "Training complete for seed $seed!"
done

echo "All seeds complete. Logs saved in $LOG_DIR/"