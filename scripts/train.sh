#!/bin/bash
cd /app
export PYTHONPATH=/app

LOG_DIR="experiments"

# echo "Generating dataset with seed 202506..."
# python src/data/generate_dataset.py --seed 202506

# echo "Augmenting dataset with seed 202506..."
# python src/data/augment.py --seed 202506

echo "========== SEED 1234 =========="
echo "Training the Custom CNN model with changing seed..."
python src/training/train_residual.py --random_seed 1234 2>&1 | tee "$LOG_DIR/custom_cnn_residual_less_lrate.txt"

echo "Training complete for seed 1234! Check the results to find the best performance."
echo "Model training complete!"