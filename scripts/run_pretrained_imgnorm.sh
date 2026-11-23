#!/bin/bash
# Pretrained models with ImageNet-style input (spectrogram scaled to [0, 255] per sample and, for
# ResNet-50 and VGG19, ImageNet channel means subtracted) on RespireNet's fold 4, oracle protocol.
# The runs are stored with the variant name "imgnorm", next to the previous runs without it, and
# the summary compares both.
#
# It waits for any experiment that is still running, skips completed runs, and can be relaunched.
#
# Usage:
#   bash scripts/run_pretrained_imgnorm.sh

cd "${APP_DIR:-/app}" || exit 1
export PYTHONPATH="${APP_DIR:-/app}"

RUN_OFFICIAL="${RUN_OFFICIAL:-bash scripts/run_official_split.sh}"
SUMMARY="experiments/pretrained_imgnorm_summary.txt"

while pgrep -f "run_official_split\.sh|run_oracle\.sh|run_oracle_splits\.sh|run_night_fold4\.sh" > /dev/null; do
    echo "Waiting for a running experiment to finish... ($(date '+%F %T'))"
    sleep 300
done

echo "########## PRETRAINED IMGNORM PLAN STARTED: $(date '+%F %T') ##########"

completed() { grep -q "^ICBHI Score:" "$1/evaluate.log" 2>/dev/null; }
run_dir() { echo "experiments/respirenet_split/${1}_val_seed_202506${2:+_$2}_oracle"; }  # $1 = model name, $2 = variant

for pair in vgg19:VGG19 efficientnet:EfficientNet resnet50:ResNet; do
    option=${pair%%:*}; name=${pair##*:}
    if completed "$(run_dir "$name" imgnorm)"; then
        echo "Skipping $name imgnorm (already completed)."
    else
        SPLIT_MODE=respirenet PROTOCOL=oracle VARIANT=imgnorm $RUN_OFFICIAL 202506 "$option" 8 || echo "$name imgnorm failed; continuing."
    fi
done

metrics() {  # $1 = run folder
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
    echo "PRETRAINED MODELS ON RESPIRENET FOLD 4 ($(date '+%F %T')) - oracle protocol (checkpoint selected on the TEST set)"
    echo "RespireNet reports 68.5% on this test fold."
    echo
    printf "%-14s %-22s %9s %9s %9s\n" "Model" "Input" "Test Se" "Test Sp" "Test ICBHI"
    printf "%-14s %-22s %s\n" "CustomCNN" "dB spectrogram" "$(metrics "$(run_dir CustomCNNCosine)")"
    for name in VGG19 EfficientNet ResNet; do
        printf "%-14s %-22s %s\n" "$name" "dB spectrogram" "$(metrics "$(run_dir "$name")")"
        printf "%-14s %-22s %s\n" "$name" "ImageNet-style" "$(metrics "$(run_dir "$name" imgnorm)")"
    done
} | tee "$SUMMARY"

echo "########## PRETRAINED IMGNORM PLAN FINISHED: $(date '+%F %T') ##########"