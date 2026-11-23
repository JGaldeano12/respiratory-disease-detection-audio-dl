#!/bin/bash
# Unattended plan (about 13 hours) on RespireNet's fold 4, oracle protocol (checkpoint selected on the TEST set):
#
#   A. VGG19 (ImageNet-style input) with class-balanced batches (RespireNet-like sampler, no focal loss weights).
#   B. Custom CNN (cosine) with class-balanced batches.
#   C. VGG19 (ImageNet-style input) with two more training seeds (202501, 202502), to measure the run-to-run noise.
#   D. Ensemble of the three VGG19 (ImageNet-style input) models: seeds 1234, 202501 and 202502.
#   E. Summary in experiments/night_balanced_summary.txt.
#
# It waits for any experiment that is still running, skips completed runs, and can be relaunched.
#
# Usage:
#   bash scripts/run_night_balanced.sh

cd "${APP_DIR:-/app}" || exit 1
export PYTHONPATH="${APP_DIR:-/app}"

RUN_OFFICIAL="${RUN_OFFICIAL:-bash scripts/run_official_split.sh}"
OUT_DIR="experiments/night_balanced"
SUMMARY="experiments/night_balanced_summary.txt"
mkdir -p "$OUT_DIR"

while pgrep -f "run_official_split\.sh|run_oracle\.sh|run_oracle_splits\.sh|run_night_fold4\.sh|run_pretrained_imgnorm\.sh" > /dev/null; do
    echo "Waiting for a running experiment to finish... ($(date '+%F %T'))"
    sleep 300
done

echo "########## BALANCED NIGHT PLAN STARTED: $(date '+%F %T') ##########"

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

# ---------- A-B. Class-balanced batches ----------
run_fold4 vgg19 VGG19 imgnorm_balanced 1234 "--balanced_batches"
run_fold4 custom_cnn_cosine CustomCNNCosine balanced 1234 "--balanced_batches"

# ---------- C. More training seeds of VGG19 (ImageNet-style input) ----------
for seed in 202501 202502; do
    run_fold4 vgg19 VGG19 "imgnorm_trainseed$seed" "$seed" ""
done

# ---------- D. Ensemble of the three VGG19 (ImageNet-style input) models ----------
MODEL_PATHS=()
for variant in imgnorm imgnorm_trainseed202501 imgnorm_trainseed202502; do
    path=$(model_path VGG19 "$variant")
    [ -f "$path" ] && MODEL_PATHS+=("$path") || echo "Missing model: $path"
done
if [ "$(cat data/processed/.dataset_config 2>/dev/null)" != "respirenet_oracle_8s" ]; then
    echo "Regenerating the fold-4 dataset for the ensemble evaluation..."
    find data/processed -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
    python -m src.data.generate_dataset --seed 202506 --duration 8 --val_split 0 --split_mode respirenet > "$OUT_DIR/generate.log" 2>&1 || { echo "Generation failed"; exit 1; }
    echo "respirenet_oracle_8s" > data/processed/.dataset_config
fi
if [ "${#MODEL_PATHS[@]}" -ge 2 ]; then
    python -m src.evaluation.evaluate_ensemble --model_paths "${MODEL_PATHS[@]}" --data_dir data/processed --output_dir "$OUT_DIR" \
        --name "VGG19_imgnorm_fold4_${#MODEL_PATHS[@]}models" > "$OUT_DIR/evaluate_ensemble.log" 2>&1 || echo "Ensemble evaluation failed"
fi

# ---------- E. Summary ----------
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
{
    echo "RESPIRENET FOLD 4 ($(date '+%F %T')) - oracle protocol (checkpoint selected on the TEST set). RespireNet: 68.5%"
    echo
    printf "%-16s %-34s %9s %9s %9s\n" "Model" "Configuration" "Test Se" "Test Sp" "Test ICBHI"
    printf "%-16s %-34s %s\n" "CustomCNNCosine" "class-weighted focal loss" "$(metrics "experiments/respirenet_split/CustomCNNCosine_val_seed_202506_oracle")"
    printf "%-16s %-34s %s\n" "CustomCNNCosine" "balanced batches" "$(metrics "$(run_dir CustomCNNCosine balanced)")"
    printf "%-16s %-34s %s\n" "VGG19" "dB input" "$(metrics "experiments/respirenet_split/VGG19_val_seed_202506_oracle")"
    printf "%-16s %-34s %s\n" "VGG19" "ImageNet input, seed 1234" "$(metrics "$(run_dir VGG19 imgnorm)")"
    printf "%-16s %-34s %s\n" "VGG19" "ImageNet input, seed 202501" "$(metrics "$(run_dir VGG19 imgnorm_trainseed202501)")"
    printf "%-16s %-34s %s\n" "VGG19" "ImageNet input, seed 202502" "$(metrics "$(run_dir VGG19 imgnorm_trainseed202502)")"
    printf "%-16s %-34s %s\n" "VGG19" "ImageNet input + balanced batches" "$(metrics "$(run_dir VGG19 imgnorm_balanced)")"
    printf "%-16s %-34s %s\n" "EfficientNet" "ImageNet input" "$(metrics "$(run_dir EfficientNet imgnorm)")"
    printf "%-16s %-34s %s\n" "ResNet" "ImageNet input" "$(metrics "$(run_dir ResNet imgnorm)")"
    echo
    sed -n '/^Ensemble:/,$p' "$OUT_DIR/evaluate_ensemble.log" 2>/dev/null
} | tee "$SUMMARY"

echo "########## BALANCED NIGHT PLAN FINISHED: $(date '+%F %T') ##########"