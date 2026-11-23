#!/bin/bash
# Custom CNN (single cosine phase) on the OFFICIAL ICBHI 2017 split with the oracle protocol
# (no validation set; checkpoint selected on the TEST set), five training seeds plus their ensemble,
# with a different feature extraction (sampling rate, bandpass cutoff and highest Mel frequency).
# It is the same plan as run_official_oracle_seeds.sh, so the results compare directly with it.
#
# Results in experiments/official_oracle_<features>_summary.txt, followed by the 8 kHz baseline.
#
# It waits for any experiment that is still running, skips completed runs, and can be relaunched.
#
# Usage:
#   bash scripts/run_official_oracle_features.sh                                   # 4000 Hz, 50-1900 Hz, Mel up to 2000 Hz
#   SAMPLE_RATE=4000 HIGHCUT=1450 FMAX=1500 bash scripts/run_official_oracle_features.sh

cd "${APP_DIR:-/app}" || exit 1
export PYTHONPATH="${APP_DIR:-/app}"

export SAMPLE_RATE="${SAMPLE_RATE:-4000}"
export HIGHCUT="${HIGHCUT:-1900}"
export FMAX="${FMAX:-2000}"
FEATURES="sr${SAMPLE_RATE}_hc${HIGHCUT}_fm${FMAX}"

RUN_OFFICIAL="${RUN_OFFICIAL:-bash scripts/run_official_split.sh}"
SEEDS="1234 202501 202502 202503 202504"
OUT_DIR="experiments/official_oracle_${FEATURES}"
SUMMARY="experiments/official_oracle_${FEATURES}_summary.txt"
DATA_TAG="official_oracle_8s_${FEATURES}"
mkdir -p "$OUT_DIR"

python3 -c "import sys; sys.exit(0 if float('$HIGHCUT') < float('$SAMPLE_RATE') / 2 else 1)" \
    || { echo "HIGHCUT ($HIGHCUT) must be below SAMPLE_RATE / 2 ($SAMPLE_RATE / 2)"; exit 1; }

while pgrep -f "run_official_split\.sh|run_oracle\.sh|run_oracle_splits\.sh|run_official_oracle_seeds\.sh|run_night_[a-z0-9_]*\.sh|run_pretrained_imgnorm\.sh" > /dev/null; do
    echo "Waiting for a running experiment to finish... ($(date '+%F %T'))"
    sleep 300
done

echo "########## OFFICIAL ORACLE $FEATURES PLAN STARTED: $(date '+%F %T') ##########"

completed() { grep -q "^ICBHI Score:" "$1/evaluate.log" 2>/dev/null; }
run_dir() { echo "experiments/official_split/CustomCNNCosine_val_seed_202506_${FEATURES}_trainseed${1}_oracle"; }
model_path() { echo "models/official_split/best_model_CustomCNNCosine_official_val_seed_202506_${FEATURES}_trainseed${1}_oracle.keras"; }

# ---------- Five training seeds ----------
for seed in $SEEDS; do
    if completed "$(run_dir "$seed")"; then
        echo "Skipping seed $seed (already completed)."
    else
        SPLIT_MODE=official PROTOCOL=oracle VARIANT="${FEATURES}_trainseed$seed" TRAIN_SEED="$seed" $RUN_OFFICIAL 202506 custom_cnn_cosine 8 \
            || echo "Seed $seed failed; continuing."
    fi
done

# ---------- Ensemble ----------
MODEL_PATHS=()
for seed in $SEEDS; do
    path=$(model_path "$seed")
    [ -f "$path" ] && MODEL_PATHS+=("$path") || echo "Missing model: $path"
done
if [ "$(cat data/processed/.dataset_config 2>/dev/null)" != "$DATA_TAG" ]; then
    echo "Regenerating the official dataset ($FEATURES) for the ensemble evaluation..."
    find data/processed -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
    python -m src.data.generate_dataset --seed 202506 --duration 8 --val_split 0 --split_mode official \
        --sample_rate "$SAMPLE_RATE" --highcut "$HIGHCUT" --fmax "$FMAX" > "$OUT_DIR/generate.log" 2>&1 || { echo "Generation failed"; exit 1; }
    echo "$DATA_TAG" > data/processed/.dataset_config
fi
if [ "${#MODEL_PATHS[@]}" -ge 2 ]; then
    python -m src.evaluation.evaluate_ensemble --model_paths "${MODEL_PATHS[@]}" --data_dir data/processed --output_dir "$OUT_DIR" \
        --name "CustomCNNCosine_official_oracle_${FEATURES}_${#MODEL_PATHS[@]}models" > "$OUT_DIR/evaluate_ensemble.log" 2>&1 || echo "Ensemble evaluation failed"
fi

# ---------- Summary ----------
{
    echo "OFFICIAL SPLIT, ORACLE PROTOCOL ($(date '+%F %T')) - Custom CNN (cosine), checkpoint selected on the TEST set"
    echo "Features: $SAMPLE_RATE Hz, bandpass 50-$HIGHCUT Hz, Mel 50-$FMAX Hz"
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
    if [ -f experiments/official_oracle_seeds_summary.txt ]; then
        echo
        echo "---------- Baseline: original features (8000 Hz, bandpass 50-2000 Hz, Mel 50-2500 Hz) ----------"
        cat experiments/official_oracle_seeds_summary.txt
    fi
} | tee "$SUMMARY"

echo "########## OFFICIAL ORACLE $FEATURES PLAN FINISHED: $(date '+%F %T') ##########"