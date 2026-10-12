#!/bin/bash
# Repeat the full pipeline (dataset generation, augmentation, training and test evaluation)
# for several patient-wise split seeds, to measure how much the results depend on the split.
#
# Usage:
#   bash scripts/run_split_seeds.sh                                   # Custom CNN, 8 s, default seeds
#   bash scripts/run_split_seeds.sh 202506 42 7                       # custom seeds
#   MODEL=custom_cnn_cosine DURATION=7 bash scripts/run_split_seeds.sh 202508 202509
#   MODEL=custom_cnn_cosine VARIANT=smooth3 TRAIN_ARGS="--smoothing_window 3" bash scripts/run_split_seeds.sh 202508
#
# Available models: custom_cnn, custom_cnn_cosine, efficientnet, vgg19, resnet50
# The training seed is kept fixed so that only the data split changes between runs.

cd "${APP_DIR:-/app}" || exit 1
export PYTHONPATH="${APP_DIR:-/app}"
export TF_CPP_MIN_LOG_LEVEL=1

SPLIT_SEEDS=("${@:-202506 42 1234 2024 7}")
SPLIT_SEEDS=(${SPLIT_SEEDS[@]})
TRAIN_SEED=1234
MODEL="${MODEL:-custom_cnn}"
DURATION="${DURATION:-8}"
VARIANT="${VARIANT:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

# Training script and checkpoint name (as saved by the ICBHI callback) of each model.
case "$MODEL" in
    custom_cnn)        TRAIN_SCRIPT="src/training/train_custom_cnn.py";        MODEL_NAME="CustomCNN" ;;
    custom_cnn_cosine) TRAIN_SCRIPT="src/training/train_custom_cnn_cosine.py"; MODEL_NAME="CustomCNNCosine" ;;
    efficientnet)      TRAIN_SCRIPT="src/training/train_efficient.py";         MODEL_NAME="EfficientNet" ;;
    vgg19)             TRAIN_SCRIPT="src/training/train_vgg.py";               MODEL_NAME="VGG19" ;;
    resnet50)          TRAIN_SCRIPT="src/training/train_resnet.py";            MODEL_NAME="ResNet" ;;
    *) echo "Unknown model: $MODEL"; exit 1 ;;
esac

# The original configuration (Custom CNN, 8 s) keeps its original folder; any other one gets its own.
if [ "$MODEL" = "custom_cnn" ] && [ "$DURATION" = "8" ] && [ -z "$VARIANT" ]; then
    OUT_DIR="experiments/split_seeds"
    MODEL_DIR="models/split_seeds"
else
    OUT_DIR="experiments/split_seeds_${MODEL_NAME}_${DURATION}s${VARIANT:+_$VARIANT}"
    MODEL_DIR="models/split_seeds_${MODEL_NAME}_${DURATION}s${VARIANT:+_$VARIANT}"
fi
RESULTS="$OUT_DIR/results.csv"
mkdir -p "$OUT_DIR" "$MODEL_DIR" models/custom_cnn models/pretrained

[ -f "$RESULTS" ] || echo "split_seed,best_val_icbhi,test_sensitivity,test_specificity,test_icbhi" > "$RESULTS"

for SEED in "${SPLIT_SEEDS[@]}"; do
    # Skip seeds whose results are already in the table, so the script can be relaunched safely.
    if grep -q "^$SEED," "$RESULTS"; then
        echo "Skipping split seed $SEED (already completed)."
        continue
    fi

    RUN_DIR="$OUT_DIR/seed_$SEED"
    mkdir -p "$RUN_DIR"
    echo "========== SPLIT SEED $SEED | $MODEL_NAME${VARIANT:+ | $VARIANT} | ${DURATION} s | $(date '+%F %T') =========="

    # 1. Delete the previous dataset (keeping .gitkeep) and regenerate it with this split seed.
    find data/processed -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
    python -m src.data.generate_dataset --seed "$SEED" --duration "$DURATION" > "$RUN_DIR/generate.log" 2>&1 || { echo "Generation failed for seed $SEED"; continue; }

    # 2. Augment the training patients of this split (same seed, so the split is identical).
    python src/data/augment.py --seed "$SEED" --sample_rate 8000 --duration "$DURATION" --test_train_split 80 > "$RUN_DIR/augment.log" 2>&1 || { echo "Augmentation failed for seed $SEED"; continue; }

    # 3. Train with a fixed training seed. Model selection uses the validation set only.
    # The previous best checkpoint is removed first, so a failed run can never reuse another seed's model.
    rm -f "models/$MODEL_NAME/best_model_$MODEL_NAME.keras"
    python "$TRAIN_SCRIPT" --random_seed "$TRAIN_SEED" $TRAIN_ARGS > "$RUN_DIR/train.log" 2>&1 || { echo "Training failed for seed $SEED"; continue; }

    # Keep a copy of the best checkpoint of this split, since the next run overwrites it.
    MODEL_PATH="$MODEL_DIR/best_model_${MODEL_NAME}_seed_$SEED.keras"
    cp "models/$MODEL_NAME/best_model_$MODEL_NAME.keras" "$MODEL_PATH" || { echo "No checkpoint found for seed $SEED"; continue; }

    # 4. Evaluate the best checkpoint once on the test set.
    python src/evaluation/evaluate.py --model_path "$MODEL_PATH" --data_dir data/processed --output_dir "$RUN_DIR" > "$RUN_DIR/evaluate.log" 2>&1 || { echo "Evaluation failed for seed $SEED"; continue; }

    # 5. Collect the metrics into the results table.
    VAL=$(grep "Best ICBHI Score during training" "$RUN_DIR/train.log" | tail -1 | awk '{print $NF}')
    SE=$(grep "^Sensitivity:" "$RUN_DIR/evaluate.log" | awk '{print $2}')
    SP=$(grep "^Specificity:" "$RUN_DIR/evaluate.log" | awk '{print $2}')
    SC=$(grep "^ICBHI Score:" "$RUN_DIR/evaluate.log" | awk '{print $3}')
    echo "$SEED,$VAL,$SE,$SP,$SC" >> "$RESULTS"
    echo "Seed $SEED -> val ICBHI: $VAL | test Se: $SE | Sp: $SP | ICBHI: $SC"
done

# Summary across all split seeds: mean and standard deviation of each metric.
python - "$RESULTS" << 'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
print("\n========== RESULTS PER SPLIT SEED ==========")
print(df.to_string(index=False))
print("\n========== MEAN ± STD ==========")
for col in df.columns[1:]:
    print(f"{col}: {df[col].mean():.4f} ± {df[col].std():.4f}")
PY