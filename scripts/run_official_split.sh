#!/bin/bash
# Run the full pipeline (dataset generation, augmentation, training and test evaluation)
# on the official ICBHI 2017 train/test partition. The validation set is drawn patient-wise
# from the official training set, and the official testing set is evaluated only once.
#
# Usage:
#   bash scripts/run_official_split.sh                        # Custom CNN, validation seed 202506
#   bash scripts/run_official_split.sh 202506 efficientnet    # EfficientNet, validation seed 202506
#   bash scripts/run_official_split.sh 202506 custom_cnn 5    # Custom CNN with 5-second segments
#   VARIANT=smooth3 TRAIN_ARGS="--smoothing_window 3" bash scripts/run_official_split.sh 202506 custom_cnn_cosine 8
#
# Optional environment variables:
#   TRAIN_ARGS  extra options passed to the training script (e.g. "--freq_norm --band_mask")
#   VARIANT     name of the variant, added to the output folder and checkpoint names
#   SPLIT_MODE  "official" (default: official ICBHI 2017 partition), "random" (seeded patient-wise
#               80:20 split; the first argument is then the split seed) or "respirenet" (RespireNet's
#               patient-wise folds, fold 4 as the testing set; the first argument only selects the
#               validation patients). Runs are stored in experiments/<mode>_split and models/<mode>_split.
#   TRAIN_SEED  training seed (weights initialization and data order); defaults to 1234
#   PROTOCOL    "standard" (default: validation set, checkpoint selected on validation) or
#               "oracle" (RespireNet-like: no validation set, its patients go back to training,
#               and the checkpoint is selected on the TEST set; results are optimistic)
#
# If data/processed already holds the dataset of this split and duration (see .dataset_config),
# generation and augmentation are skipped, since they do not depend on the training options.
#
# Available models: custom_cnn, custom_cnn_cosine, efficientnet, vgg19, resnet50

cd /app || exit 1
export PYTHONPATH=/app
export TF_CPP_MIN_LOG_LEVEL=1

VAL_SEED="${1:-202506}"
MODEL="${2:-custom_cnn}"
DURATION="${3:-8}"
VARIANT="${VARIANT:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
PROTOCOL="${PROTOCOL:-standard}"
VAL_SPLIT=15
SPLIT_MODE="${SPLIT_MODE:-official}"
case "$SPLIT_MODE" in
    official) SPLIT_NAME="official"; SPLIT_LABEL="OFFICIAL SPLIT"; SEED_LABEL="validation seed" ;;
    random)   SPLIT_NAME="random";   SPLIT_LABEL="RANDOM 80:20 SPLIT"; SEED_LABEL="split seed" ;;
    respirenet) SPLIT_NAME="respirenet"; SPLIT_LABEL="RESPIRENET FOLD 4"; SEED_LABEL="validation seed" ;;
    *) echo "Unknown split mode: $SPLIT_MODE (use official, random or respirenet)"; exit 1 ;;
esac
if [ "$PROTOCOL" = "oracle" ]; then
    VAL_SPLIT=0
    TRAIN_ARGS="$TRAIN_ARGS --select_on test"
elif [ "$PROTOCOL" != "standard" ]; then
    echo "Unknown protocol: $PROTOCOL (use standard or oracle)"; exit 1
fi
TRAIN_SEED="${TRAIN_SEED:-1234}"
OFFICIAL_FILE="src/resources/ICBHI_challenge_train_test.txt"

# Training script and checkpoint name (as saved by the ICBHI callback) of each model.
case "$MODEL" in
    custom_cnn)   TRAIN_SCRIPT="src/training/train_custom_cnn.py"; MODEL_NAME="CustomCNN" ;;
    custom_cnn_cosine) TRAIN_SCRIPT="src/training/train_custom_cnn_cosine.py"; MODEL_NAME="CustomCNNCosine" ;;
    efficientnet) TRAIN_SCRIPT="src/training/train_efficient.py";  MODEL_NAME="EfficientNet" ;;
    vgg19)        TRAIN_SCRIPT="src/training/train_vgg.py";        MODEL_NAME="VGG19" ;;
    resnet50)     TRAIN_SCRIPT="src/training/train_resnet.py";     MODEL_NAME="ResNet" ;;
    *) echo "Unknown model: $MODEL (use custom_cnn, custom_cnn_cosine, efficientnet, vgg19 or resnet50)"; exit 1 ;;
esac

# Runs with the default 8-second segments keep their original names; other durations add a suffix.
SUFFIX=""
[ "$DURATION" != "8" ] && SUFFIX="_${DURATION}s"
[ -n "$VARIANT" ] && SUFFIX="${SUFFIX}_${VARIANT}"
[ "$PROTOCOL" = "oracle" ] && SUFFIX="${SUFFIX}_oracle"

OUT_DIR="experiments/${SPLIT_NAME}_split/${MODEL_NAME}_val_seed_${VAL_SEED}${SUFFIX}"
MODEL_PATH="models/${SPLIT_NAME}_split/best_model_${MODEL_NAME}_${SPLIT_NAME}_val_seed_${VAL_SEED}${SUFFIX}.keras"

# The training scripts save their final model in these folders, so they must exist.
mkdir -p "$OUT_DIR" "models/${SPLIT_NAME}_split" models/custom_cnn models/pretrained

[ -f "$OFFICIAL_FILE" ] || { echo "Official partition file not found: $OFFICIAL_FILE"; exit 1; }

echo "========== $SPLIT_LABEL | $MODEL_NAME${VARIANT:+ | $VARIANT} | $SEED_LABEL $VAL_SEED | ${DURATION} s | $PROTOCOL | $(date '+%F %T') =========="
[ -n "$TRAIN_ARGS" ] && echo "Training options: $TRAIN_ARGS"

DATA_TAG="${SPLIT_NAME}_${VAL_SEED}_${DURATION}s"
[ "$PROTOCOL" = "oracle" ] && [ "$SPLIT_MODE" = "official" ] && DATA_TAG="official_oracle_${DURATION}s"
[ "$PROTOCOL" = "oracle" ] && [ "$SPLIT_MODE" = "random" ] && DATA_TAG="random_${VAL_SEED}_oracle_${DURATION}s"
[ "$PROTOCOL" = "oracle" ] && [ "$SPLIT_MODE" = "respirenet" ] && DATA_TAG="respirenet_oracle_${DURATION}s"
DATA_MARKER="data/processed/.dataset_config"

if [ "$(cat "$DATA_MARKER" 2>/dev/null)" = "$DATA_TAG" ]; then
    echo "Reusing the existing dataset ($DATA_TAG)."
else
    # 1. Delete the previous dataset (keeping .gitkeep) and regenerate it with the official partition.
    find data/processed -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
    python -m src.data.generate_dataset --seed "$VAL_SEED" --duration "$DURATION" --val_split "$VAL_SPLIT" --test_train_split 80 --split_mode "$SPLIT_MODE" --official_split_file "$OFFICIAL_FILE" > "$OUT_DIR/generate.log" 2>&1 || { echo "Generation failed"; exit 1; }
    grep -E "Warning|not found" "$OUT_DIR/generate.log"

    # 2. Augment only the official training patients that are not in the validation set.
    python src/data/augment.py --seed "$VAL_SEED" --sample_rate 8000 --duration "$DURATION" --val_split "$VAL_SPLIT" --test_train_split 80 --split_mode "$SPLIT_MODE" --official_split_file "$OFFICIAL_FILE" > "$OUT_DIR/augment.log" 2>&1 || { echo "Augmentation failed"; exit 1; }

    # Record which dataset is now in data/processed, so later runs with the same split and duration can reuse it.
    echo "$DATA_TAG" > "$DATA_MARKER"
fi

# 3. Train with a fixed training seed. Model selection uses the validation set only.
# The previous best checkpoint is removed first, so a failed run can never reuse another run's model.
rm -f "models/$MODEL_NAME/best_model_$MODEL_NAME.keras"
python "$TRAIN_SCRIPT" --random_seed "$TRAIN_SEED" $TRAIN_ARGS > "$OUT_DIR/train.log" 2>&1 || { echo "Training failed"; exit 1; }

# Keep a copy of the best checkpoint, since other runs overwrite it.
cp "models/$MODEL_NAME/best_model_$MODEL_NAME.keras" "$MODEL_PATH" || { echo "No checkpoint found"; exit 1; }

# 4. Evaluate the best checkpoint once on the official testing set, overall and per recording device.
python src/evaluation/evaluate.py --model_path "$MODEL_PATH" --data_dir data/processed --output_dir "$OUT_DIR" > "$OUT_DIR/evaluate.log" 2>&1 || { echo "Evaluation failed"; exit 1; }
python -m src.evaluation.evaluate_by_device --model_path "$MODEL_PATH" --data_dir data/processed --output_dir "$OUT_DIR" > "$OUT_DIR/evaluate_by_device.log" 2>&1 || echo "Per-device evaluation failed"

# 5. Summary.
echo "Best validation ICBHI: $(grep "Best ICBHI Score during training" "$OUT_DIR/train.log" | tail -1 | awk '{print $NF}')"
grep -E "^(Sensitivity|Specificity|ICBHI Score):" "$OUT_DIR/evaluate.log"
sed -n '/\[Test\]/,$p' "$OUT_DIR/evaluate_by_device.log"
echo "Logs and confusion matrix saved in $OUT_DIR"