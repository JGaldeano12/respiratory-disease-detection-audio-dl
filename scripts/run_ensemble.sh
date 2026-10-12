#!/bin/bash
# Ensemble of single-phase Custom CNNs (official split, 8 s, 3-epoch smoothed selection).
# The members differ only in their training seed and are fixed beforehand; their predicted
# probabilities are averaged with equal weights, so nothing is selected on the test set.
#
# The member trained with seed 1234 is the reference run of the improvement plan ("smooth3"),
# so only the remaining members are trained here. If the improvement plan is still running,
# this script waits until it finishes. Completed members are skipped, so it can be relaunched.
#
# Usage:
#   bash scripts/run_ensemble.sh

cd "${APP_DIR:-/app}" || exit 1
export PYTHONPATH="${APP_DIR:-/app}"
export TF_CPP_MIN_LOG_LEVEL=1

VAL_SEED=202506
DURATION=8
TRAIN_ARGS="--smoothing_window 3"
EXTRA_TRAIN_SEEDS="101 202 303 404"
REFERENCE_MODEL="models/official_split/best_model_CustomCNNCosine_official_val_seed_${VAL_SEED}_smooth3.keras"
OUT_DIR="experiments/ensemble"
SUMMARY="experiments/ensemble_summary.txt"

# Wait for the improvement plan, which uses the same data and model folders.
while pgrep -f run_improvements.sh > /dev/null; do
    echo "Waiting for run_improvements.sh to finish... ($(date '+%F %T'))"
    sleep 600
done

echo "########## ENSEMBLE PLAN STARTED: $(date '+%F %T') ##########"
mkdir -p "$OUT_DIR"

[ -f "$REFERENCE_MODEL" ] || { echo "Reference model not found: $REFERENCE_MODEL"; exit 1; }
MODEL_PATHS=("$REFERENCE_MODEL")

# Train the remaining members (the dataset is regenerated once and then reused).
for seed in $EXTRA_TRAIN_SEEDS; do
    variant="smooth3_trainseed$seed"
    model_path="models/official_split/best_model_CustomCNNCosine_official_val_seed_${VAL_SEED}_${variant}.keras"
    if [ -f "$model_path" ]; then
        echo "Skipping member with training seed $seed (already trained)."
    else
        VARIANT="$variant" TRAIN_ARGS="$TRAIN_ARGS" TRAIN_SEED="$seed" bash scripts/run_official_split.sh "$VAL_SEED" custom_cnn_cosine "$DURATION"
    fi
    [ -f "$model_path" ] && MODEL_PATHS+=("$model_path") || echo "Member with training seed $seed is missing; continuing without it."
done

# Make sure the official dataset is in place before evaluating (it is if any member was trained here).
if [ "$(cat data/processed/.dataset_config 2>/dev/null)" != "official_${VAL_SEED}_${DURATION}s" ]; then
    echo "Regenerating the official dataset for the evaluation..."
    find data/processed -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
    python -m src.data.generate_dataset --seed "$VAL_SEED" --duration "$DURATION" --split_mode official > "$OUT_DIR/generate.log" 2>&1 || { echo "Generation failed"; exit 1; }
fi

# Evaluate every member and the ensemble on the validation and test sets.
python -m src.evaluation.evaluate_ensemble --model_paths "${MODEL_PATHS[@]}" --data_dir data/processed --output_dir "$OUT_DIR" --name "CustomCNNCosine_smooth3_${#MODEL_PATHS[@]}models" > "$OUT_DIR/evaluate_ensemble.log" 2>&1 || { echo "Ensemble evaluation failed"; exit 1; }

sed -n '/^Ensemble:/,$p' "$OUT_DIR/evaluate_ensemble.log" | tee "$SUMMARY"
echo "########## ENSEMBLE PLAN FINISHED: $(date '+%F %T') ##########"