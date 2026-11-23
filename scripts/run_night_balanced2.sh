#!/bin/bash
# Unattended plan (about 15 hours) on RespireNet's fold 4, oracle protocol (checkpoint selected on the TEST set):
#
#   A. Custom CNN (cosine) + balanced batches + random blanking of the 1.5-2 kHz band (Littmann 3200), one run.
#   B. Custom CNN (cosine) + balanced batches with three more training seeds, and the ensemble of the four models.
#   C. VGG19 (ImageNet-style input) + balanced batches with two more training seeds, and the ensemble of the three models.
#   D. Summary in experiments/night_balanced2_summary.txt.
#
# It waits for any experiment that is still running, skips completed runs, and can be relaunched.
#
# Usage:
#   bash scripts/run_night_balanced2.sh

cd "${APP_DIR:-/app}" || exit 1
export PYTHONPATH="${APP_DIR:-/app}"

RUN_OFFICIAL="${RUN_OFFICIAL:-bash scripts/run_official_split.sh}"
OUT_DIR="experiments/night_balanced2"
SUMMARY="experiments/night_balanced2_summary.txt"
mkdir -p "$OUT_DIR"

while pgrep -f "run_official_split\.sh|run_oracle\.sh|run_oracle_splits\.sh|run_night_fold4\.sh|run_pretrained_imgnorm\.sh|run_night_balanced\.sh" > /dev/null; do
    echo "Waiting for a running experiment to finish... ($(date '+%F %T'))"
    sleep 300
done

echo "########## BALANCED NIGHT PLAN 2 STARTED: $(date '+%F %T') ##########"

completed() { grep -q "^ICBHI Score:" "$1/evaluate.log" 2>/dev/null; }
run_dir() { echo "experiments/respirenet_split/${1}_val_seed_202506_${2}_oracle"; }               # $1 = model name, $2 = variant
model_path() { echo "models/respirenet_split/best_model_${1}_respirenet_val_seed_202506_${2}_oracle.keras"; }

# Run one fold-4 oracle experiment unless it has already been completed.
run_fold4() {  # $1 = model option, $2 = model name, $3 = variant, $4 = training seed, $5 = training options
    if completed "$(run_dir "$2" "$3")"; then
        echo "Skipping $2 $3 (already completed)."
    else
        SPLIT_MODE=respirenet PROTOCOL=oracle VARIANT="$3" TRAIN_SEED="$4" TRAIN_ARGS="$5" $RUN_OFFICIAL 202506 "$1" 8 \
            || echo "$2 $3 failed; continuing."
    fi
}

# Evaluate the ensemble of the given variants of a model on the fold-4 test set.
run_ensemble() {  # $1 = model name, $2 = ensemble name, remaining arguments = variants
    local name="$1" ens="$2"; shift 2
    local paths=() variant path
    for variant in "$@"; do
        path=$(model_path "$name" "$variant")
        [ -f "$path" ] && paths+=("$path") || echo "Missing model: $path"
    done
    if [ "$(cat data/processed/.dataset_config 2>/dev/null)" != "respirenet_oracle_8s" ]; then
        echo "Regenerating the fold-4 dataset for the ensemble evaluation..."
        find data/processed -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
        python -m src.data.generate_dataset --seed 202506 --duration 8 --val_split 0 --split_mode respirenet > "$OUT_DIR/generate.log" 2>&1 || { echo "Generation failed"; return 1; }
        echo "respirenet_oracle_8s" > data/processed/.dataset_config
    fi
    if [ "${#paths[@]}" -ge 2 ]; then
        python -m src.evaluation.evaluate_ensemble --model_paths "${paths[@]}" --data_dir data/processed --output_dir "$OUT_DIR" \
            --name "${ens}_${#paths[@]}models" > "$OUT_DIR/${ens}.log" 2>&1 || echo "Ensemble $ens failed"
    fi
}

# ---------- A. Custom CNN + balanced batches + band blanking ----------
run_fold4 custom_cnn_cosine CustomCNNCosine balanced_bandmask 1234 "--balanced_batches --band_mask"

# ---------- B. Custom CNN + balanced batches: more seeds and ensemble ----------
for seed in 202501 202502 202503; do
    run_fold4 custom_cnn_cosine CustomCNNCosine "balanced_trainseed$seed" "$seed" "--balanced_batches"
done
run_ensemble CustomCNNCosine CustomCNN_balanced_fold4 balanced balanced_trainseed202501 balanced_trainseed202502 balanced_trainseed202503

# ---------- C. VGG19 + balanced batches: more seeds and ensemble ----------
for seed in 202501 202502; do
    run_fold4 vgg19 VGG19 "imgnorm_balanced_trainseed$seed" "$seed" "--balanced_batches"
done
run_ensemble VGG19 VGG19_balanced_fold4 imgnorm_balanced imgnorm_balanced_trainseed202501 imgnorm_balanced_trainseed202502

# ---------- D. Summary ----------
metrics() {
    if completed "$1"; then
        printf "%9s %9s %9s" \
            "$(grep '^Sensitivity:' "$1/evaluate.log" | awk '{print $2}')" \
            "$(grep '^Specificity:' "$1/evaluate.log" | awk '{print $2}')" \
            "$(grep '^ICBHI Score:' "$1/evaluate.log" | awk '{print $3}')"
    else
        printf "%29s" "no results"
    fi
}
row() { printf "%-16s %-40s %s\n" "$1" "$2" "$(metrics "$(run_dir "$1" "$3")")"; }
{
    echo "RESPIRENET FOLD 4 ($(date '+%F %T')) - oracle protocol (checkpoint selected on the TEST set). RespireNet: 68.5%"
    echo
    printf "%-16s %-40s %9s %9s %9s\n" "Model" "Configuration" "Test Se" "Test Sp" "Test ICBHI"
    row CustomCNNCosine "balanced batches, seed 1234" balanced
    row CustomCNNCosine "balanced batches, seed 202501" balanced_trainseed202501
    row CustomCNNCosine "balanced batches, seed 202502" balanced_trainseed202502
    row CustomCNNCosine "balanced batches, seed 202503" balanced_trainseed202503
    row CustomCNNCosine "balanced batches + band blanking" balanced_bandmask
    row VGG19 "ImageNet input + balanced, seed 1234" imgnorm_balanced
    row VGG19 "ImageNet input + balanced, seed 202501" imgnorm_balanced_trainseed202501
    row VGG19 "ImageNet input + balanced, seed 202502" imgnorm_balanced_trainseed202502
    for log in "$OUT_DIR"/CustomCNN_balanced_fold4.log "$OUT_DIR"/VGG19_balanced_fold4.log; do
        echo
        sed -n '/^Ensemble:/,$p' "$log" 2>/dev/null
    done
} | tee "$SUMMARY"

echo "########## BALANCED NIGHT PLAN 2 FINISHED: $(date '+%F %T') ##########"