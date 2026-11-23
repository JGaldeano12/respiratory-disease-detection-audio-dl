#!/bin/bash
# Custom CNN (single cosine phase) on the OFFICIAL ICBHI 2017 split with the oracle protocol
# (no validation set; checkpoint selected on the TEST set), trained with five training seeds,
# plus the ensemble of the five models. This is the protocol most comparable with published
# CNNs trained without pretraining on the official split, which report a single run.
#
# Results: per-seed metrics, mean ± std over seeds, and the ensemble, in
# experiments/official_oracle_seeds_summary.txt.
#
# It waits for any experiment that is still running, skips completed runs, and can be relaunched.
#
# Usage:
#   bash scripts/run_official_oracle_seeds.sh

cd "${APP_DIR:-/app}" || exit 1
export PYTHONPATH="${APP_DIR:-/app}"

RUN_OFFICIAL="${RUN_OFFICIAL:-bash scripts/run_official_split.sh}"
SEEDS="1234 202501 202502 202503 202504"
OUT_DIR="experiments/official_oracle_seeds"
SUMMARY="experiments/official_oracle_seeds_summary.txt"
mkdir -p "$OUT_DIR"

while pgrep -f "run_official_split\.sh|run_oracle\.sh|run_oracle_splits\.sh|run_night_[a-z0-9_]*\.sh|run_pretrained_imgnorm\.sh" > /dev/null; do
    echo "Waiting for a running experiment to finish... ($(date '+%F %T'))"
    sleep 300
done

echo "########## OFFICIAL ORACLE SEEDS PLAN STARTED: $(date '+%F %T') ##########"

completed() { grep -q "^ICBHI Score:" "$1/evaluate.log" 2>/dev/null; }
run_dir() { echo "experiments/official_split/CustomCNNCosine_val_seed_202506_trainseed${1}_oracle"; }
model_path() { echo "models/official_split/best_model_CustomCNNCosine_official_val_seed_202506_trainseed${1}_oracle.keras"; }

# ---------- Five training seeds ----------
for seed in $SEEDS; do
    if completed "$(run_dir "$seed")"; then
        echo "Skipping seed $seed (already completed)."
    else
        SPLIT_MODE=official PROTOCOL=oracle VARIANT="trainseed$seed" TRAIN_SEED="$seed" $RUN_OFFICIAL 202506 custom_cnn_cosine 8 \
            || echo "Seed $seed failed; continuing."
    fi
done

# ---------- Ensemble ----------
MODEL_PATHS=()
for seed in $SEEDS; do
    path=$(model_path "$seed")
    [ -f "$path" ] && MODEL_PATHS+=("$path") || echo "Missing model: $path"
done
if [ "$(cat data/processed/.dataset_config 2>/dev/null)" != "official_oracle_8s" ]; then
    echo "Regenerating the official dataset for the ensemble evaluation..."
    find data/processed -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
    python -m src.data.generate_dataset --seed 202506 --duration 8 --val_split 0 --split_mode official > "$OUT_DIR/generate.log" 2>&1 || { echo "Generation failed"; exit 1; }
    echo "official_oracle_8s" > data/processed/.dataset_config
fi
if [ "${#MODEL_PATHS[@]}" -ge 2 ]; then
    python -m src.evaluation.evaluate_ensemble --model_paths "${MODEL_PATHS[@]}" --data_dir data/processed --output_dir "$OUT_DIR" \
        --name "CustomCNNCosine_official_oracle_${#MODEL_PATHS[@]}models" > "$OUT_DIR/evaluate_ensemble.log" 2>&1 || echo "Ensemble evaluation failed"
fi

# ---------- Summary ----------
{
    echo "OFFICIAL SPLIT, ORACLE PROTOCOL ($(date '+%F %T')) - Custom CNN (cosine), checkpoint selected on the TEST set"
    echo
    printf "%-10s %9s %9s %9s\n" "Seed" "Test Se" "Test Sp" "Test ICBHI"
    for seed in $SEEDS; do
        dir=$(run_dir "$seed")
        if completed "$dir"; then
            printf "%-10s %9s %9s %9s\n" "$seed" \
                "$(grep '^Sensitivity:' "$dir/evaluate.log" | awk '{print $2}')" \
                "$(grep '^Specificity:' "$dir/evaluate.log" | awk '{print $2}')" \
                "$(grep '^ICBHI Score:' "$dir/evaluate.log" | awk '{print $3}')"
        else
            printf "%-10s %29s\n" "$seed" "no results"
        fi
    done
    for seed in $SEEDS; do grep "^ICBHI Score:" "$(run_dir "$seed")/evaluate.log" 2>/dev/null | awk '{print $3}'; done \
        | awk '{s+=$1; ss+=$1*$1; n++} END {if (n>1) printf "\nMean ± std over %d seeds: %.4f ± %.4f\n", n, s/n, sqrt((ss-s*s/n)/(n-1))}'
    echo
    sed -n '/^Ensemble:/,$p' "$OUT_DIR/evaluate_ensemble.log" 2>/dev/null
} | tee "$SUMMARY"

echo "########## OFFICIAL ORACLE SEEDS PLAN FINISHED: $(date '+%F %T') ##########"