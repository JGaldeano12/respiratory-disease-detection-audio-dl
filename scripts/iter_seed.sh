#!/bin/bash
cd /app
export PYTHONPATH=/app

LOG_DIR="src/testing"
mkdir -p "$LOG_DIR"

for seed in \
202511 202512 \
202601 202602 202603 202604 202605 202606 \
202607 202608 202609 202610 202611 202612
do
    echo "========== SEED $seed =========="

    echo "Generating dataset with seed $seed..."
    python src/data/generate_dataset.py --seed "$seed"

    echo "Augmenting dataset with seed $seed..."
    python src/data/augment.py --seed "$seed"

    echo "Training the Custom CNN model with changing seed..."
    python src/training/train_residual.py --random_seed $seed 2>&1 | tee "$LOG_DIR/seed${seed}_last_try.txt"

    echo "Training complete for seed $seed! Check the results to find the best performance."

    echo "Cleaning up the dataset for the next seed..."
    python src/data/delete_dataset.py

done

echo "All seeds complete. Logs saved in $LOG_DIR/"