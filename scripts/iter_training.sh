#!/bin/bash
cd /app
export PYTHONPATH=/app

LOG_DIR="src/testing"

echo "Generating dataset with seed 202506..."
python src/data/generate_dataset.py --seed 202506

echo "Augmenting dataset with seed 202506..."
python src/data/augment.py --seed 202506

# Start iterating through different seeds for training.
# I'll do a loop starting from 12345 and adding numbers such as 123456, 1234567 until 10 experiments are done.

for seed in 123 1234 12345 123456 1234567 12345678 123456789 1234567890
do
    echo "========== SEED $seed =========="
    echo "Training the Custom CNN model with changing seed..."
    python src/training/train_residual.py --random_seed $seed 2>&1 | tee "$LOG_DIR/custom_cnn_residual_final_experiment_${seed}.txt"
    echo "Training complete for seed $seed! Check the results to find the best performance."
done

echo "Training complete for all seeds!"
echo "All seeds complete. Logs saved in $LOG_DIR/"