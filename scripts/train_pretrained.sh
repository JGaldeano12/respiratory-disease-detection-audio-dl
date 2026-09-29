#!/bin/bash

cd /app || exit 1
export PYTHONPATH=/app

LOG_DIR="experiments"
SEED=1234

mkdir -p "$LOG_DIR"

echo "========================================="
echo "        MODEL TRAINING"
echo "========================================="
echo
echo "Select the model you want to train:"
echo
echo "  1) ResNet"
echo "  2) EfficientNet"
echo "  3) VGG"
echo "  4) All models"
echo
read -rp "Enter your choice [1-4]: " MODEL

case "$MODEL" in

    1)
        echo "Training ResNet with seed $SEED..."
        python src/training/train_resnet.py --random_seed "$SEED" 2>&1 | tee "$LOG_DIR/training_resnet.txt"
        ;;

    2)
        echo "Training EfficientNet with seed $SEED..."
        python src/training/train_efficient.py  --random_seed "$SEED" 2>&1 | tee "$LOG_DIR/training_efficient.txt"
        ;;

    3)
        echo
        echo "Training VGG with seed $SEED..."
        python src/training/train_vgg.py --random_seed "$SEED" 2>&1 | tee "$LOG_DIR/training_vgg.txt"
        ;;

    4)
        echo
        echo "Training all models with seed $SEED..."
        echo

        echo "Training ResNet..."
        python src/training/train_resnet.py --random_seed "$SEED" 2>&1 | tee "$LOG_DIR/training_resnet.txt"

        echo
        echo "Training EfficientNet..."
        python src/training/train_efficient.py  --random_seed "$SEED" 2>&1 | tee "$LOG_DIR/training_efficient.txt"

        echo
        echo "Training VGG..."
        python src/training/train_vgg.py --random_seed "$SEED" 2>&1 | tee "$LOG_DIR/training_vgg.txt"
        ;;

    *)
        echo
        echo "Invalid option. Please select a number between 1 and 4."
        exit 1
        ;;

esac

echo
echo "Check the results in the '$LOG_DIR' directory."