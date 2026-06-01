#!/bin/bash
cd /app
export PYTHONPATH=/app

LOG_DIR="src/testing"
mkdir -p $LOG_DIR

for seed in $(seq 20260114 20260131)
do
    echo "========== SEED $seed =========="

    echo "Generating dataset with seed $seed..."
    python src/data/generate_dataset.py --seed $seed

    echo "Augmenting dataset with seed $seed..."
    python src/data/augment.py --seed $seed

    echo "Training the EfficientNet model with seed 123456..."
    python src/training/train_efficient.py --random_seed 123456 2>&1 | tee "$LOG_DIR/seed${seed}_reproducibility_secured.txt"

    echo "Training complete for seed $seed! Check the results to find the best performance."

    echo "Cleaning up the dataset for the next seed..."
    python src/data/delete_dataset.py

done

echo "All seeds complete. Logs saved in $LOG_DIR/"