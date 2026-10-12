#!/bin/bash
# Oracle protocol (RespireNet-like), 8-second segments, on a seeded patient-wise random 80:20 split
# (SPLIT_MODE=random, the default) or on the official ICBHI 2017 split (SPLIT_MODE=official).
# There is no validation set: its patients go back to training, and the checkpoint of every run
# is selected on the TEST set. The resulting scores are optimistic and must be reported as such.
#
#   1. Custom CNN, three phases, training seed 1234.
#   2. Custom CNN, single cosine phase, training seed 1234.
#   3. The variant with the best oracle score is trained with four more training seeds (dates).
#   4. The five models of that variant are averaged as an ensemble and evaluated on the test set.
#
# Completed runs are skipped, so the script can be relaunched safely.
#
# Usage:
#   bash scripts/run_oracle.sh                                # random 80:20 split, split seed 202506
#   SPLIT_SEED=42 bash scripts/run_oracle.sh                  # random 80:20 split, split seed 42
#   SPLIT_MODE=official bash scripts/run_oracle.sh            # official split
#   SPLIT_MODE=respirenet bash scripts/run_oracle.sh          # RespireNet folds (fold 4 as the testing set)

cd "${APP_DIR:-/app}" || exit 1
export PYTHONPATH="${APP_DIR:-/app}"
export TF_CPP_MIN_LOG_LEVEL=1

SPLIT_MODE="${SPLIT_MODE:-random}"
VAL_SEED="${SPLIT_SEED:-202506}"
DURATION=8
EXTRA_TRAIN_SEEDS="${EXTRA_TRAIN_SEEDS:-202501 202502 202503 202504}"
RUN_OFFICIAL="${RUN_OFFICIAL:-bash scripts/run_official_split.sh}"
OUT_DIR="experiments/oracle_${SPLIT_MODE}_${VAL_SEED}"
SUMMARY="experiments/oracle_summary_${SPLIT_MODE}_${VAL_SEED}.txt"
mkdir -p "$OUT_DIR"

# Output folder and checkpoint of an oracle run, as named by run_official_split.sh.
run_dir() { echo "experiments/${SPLIT_MODE}_split/${1}_val_seed_${VAL_SEED}${2:+_$2}_oracle"; }  # $1 = model name, $2 = variant
model_path() { echo "models/${SPLIT_MODE}_split/best_model_${1}_${SPLIT_MODE}_val_seed_${VAL_SEED}${2:+_$2}_oracle.keras"; }

# Best (test-selected) ICBHI score of a run, or -1 if it is missing or incomplete.
oracle_score() {
    local score
    score=$(grep "Best ICBHI Score during training" "$1/train.log" 2>/dev/null | head -1 | awk '{print $NF}')
    echo "${score:--1}"
}

# Run an oracle experiment unless it has already been completed.
run_oracle() {  # $1 = model option, $2 = model name, $3 = variant, $4 = training seed
    if [ -f "$(run_dir "$2" "$3")/evaluate.log" ]; then
        echo "Skipping $2 ${3:-seed 1234} (already completed)."
    else
        SPLIT_MODE="$SPLIT_MODE" PROTOCOL=oracle VARIANT="$3" TRAIN_SEED="$4" $RUN_OFFICIAL "$VAL_SEED" "$1" "$DURATION"
    fi
}

echo "########## ORACLE PLAN STARTED ($SPLIT_MODE split, seed $VAL_SEED): $(date '+%F %T') ##########"

# ---------- 1-2. Both Custom CNN variants ----------
run_oracle custom_cnn CustomCNN "" 1234
run_oracle custom_cnn_cosine CustomCNNCosine "" 1234

SCORE_PHASES=$(oracle_score "$(run_dir CustomCNN)")
SCORE_COSINE=$(oracle_score "$(run_dir CustomCNNCosine)")
if awk -v a="$SCORE_COSINE" -v b="$SCORE_PHASES" 'BEGIN { exit !(a > b) }'; then
    BEST_OPTION=custom_cnn_cosine; BEST_NAME=CustomCNNCosine
else
    BEST_OPTION=custom_cnn; BEST_NAME=CustomCNN
fi
echo ">>> Oracle scores: three phases $SCORE_PHASES, cosine $SCORE_COSINE -> $BEST_NAME"

# ---------- 3. More training seeds of the best variant ----------
MODEL_PATHS=("$(model_path "$BEST_NAME")")
for seed in $EXTRA_TRAIN_SEEDS; do
    run_oracle "$BEST_OPTION" "$BEST_NAME" "trainseed$seed" "$seed"
    path=$(model_path "$BEST_NAME" "trainseed$seed")
    [ -f "$path" ] && MODEL_PATHS+=("$path") || echo "Model with training seed $seed is missing; continuing without it."
done

# ---------- 4. Ensemble ----------
DATA_TAG="official_oracle_${DURATION}s"
[ "$SPLIT_MODE" = "random" ] && DATA_TAG="random_${VAL_SEED}_oracle_${DURATION}s"
[ "$SPLIT_MODE" = "respirenet" ] && DATA_TAG="respirenet_oracle_${DURATION}s"
if [ "$(cat data/processed/.dataset_config 2>/dev/null)" != "$DATA_TAG" ]; then
    echo "Regenerating the oracle dataset for the ensemble evaluation..."
    find data/processed -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
    python -m src.data.generate_dataset --seed "$VAL_SEED" --duration "$DURATION" --val_split 0 --test_train_split 80 --split_mode "$SPLIT_MODE" > "$OUT_DIR/generate.log" 2>&1 || { echo "Generation failed"; exit 1; }
    echo "$DATA_TAG" > data/processed/.dataset_config
fi
python -m src.evaluation.evaluate_ensemble --model_paths "${MODEL_PATHS[@]}" --data_dir data/processed --output_dir "$OUT_DIR" --name "${BEST_NAME}_oracle_${#MODEL_PATHS[@]}models" > "$OUT_DIR/evaluate_ensemble.log" 2>&1 || echo "Ensemble evaluation failed"

# ---------- Summary ----------
{
    echo "ORACLE SUMMARY ($(date '+%F %T')) - $SPLIT_MODE split, seed $VAL_SEED"
    echo "Checkpoints selected on the TEST set (RespireNet-like protocol); scores are optimistic."
    echo
    printf "%-44s %9s %9s %9s\n" "Run" "Test Se" "Test Sp" "Test ICBHI"
    for dir in experiments/${SPLIT_MODE}_split/*_val_seed_${VAL_SEED}*_oracle; do
        [ -f "$dir/evaluate.log" ] || continue
        printf "%-44s %9s %9s %9s\n" "$(basename "$dir" | sed "s/_val_seed_${VAL_SEED}//")" \
            "$(grep '^Sensitivity:' "$dir/evaluate.log" | awk '{print $2}')" \
            "$(grep '^Specificity:' "$dir/evaluate.log" | awk '{print $2}')" \
            "$(grep '^ICBHI Score:' "$dir/evaluate.log" | awk '{print $3}')"
    done
    echo
    sed -n '/^Ensemble:/,$p' "$OUT_DIR/evaluate_ensemble.log"
} | tee "$SUMMARY"

echo "########## ORACLE PLAN FINISHED: $(date '+%F %T') ##########"