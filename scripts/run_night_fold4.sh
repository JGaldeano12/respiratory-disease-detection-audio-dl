#!/bin/bash
# Unattended plan (about 14-16 hours), oracle protocol (checkpoint selected on the TEST set):
#
#   A. Pretrained models on RespireNet's fold 4 (the same test patients as RespireNet's 80-20
#      results): EfficientNet, ResNet-50 and VGG19, one run each, 8-second segments.
#   B. Random-split sweep: the runs affected by degenerate epochs (an epoch with sensitivity or
#      specificity 0 that became the best one and cut the training short) are moved to *_old and
#      trained again with the fixed callback; the sweep summary is then rebuilt with all splits.
#   C. Summary of every result in experiments/night_fold4_summary.txt.
#
# It waits for any experiment that is still running, skips completed runs, and can be relaunched.
#
# Usage:
#   bash scripts/run_night_fold4.sh

cd "${APP_DIR:-/app}" || exit 1
export PYTHONPATH="${APP_DIR:-/app}"

RUN_OFFICIAL="${RUN_OFFICIAL:-bash scripts/run_official_split.sh}"
RUN_SPLITS="${RUN_SPLITS:-bash scripts/run_oracle_splits.sh}"
SUMMARY="experiments/night_fold4_summary.txt"

# Wait for any experiment that is still running (they share data/processed and the model folders).
while pgrep -f "run_official_split\.sh|run_oracle\.sh|run_oracle_splits\.sh" > /dev/null; do
    echo "Waiting for a running experiment to finish... ($(date '+%F %T'))"
    sleep 300
done

echo "########## NIGHT PLAN STARTED: $(date '+%F %T') ##########"

# A run is completed only if its evaluation log contains the final metrics.
completed() { grep -q "^ICBHI Score:" "$1/evaluate.log" 2>/dev/null; }

# ---------- A. Pretrained models on RespireNet's fold 4 ----------
for pair in efficientnet:EfficientNet resnet50:ResNet vgg19:VGG19; do
    option=${pair%%:*}; name=${pair##*:}
    if completed "experiments/respirenet_split/${name}_val_seed_202506_oracle"; then
        echo "Skipping $name on fold 4 (already completed)."
    else
        SPLIT_MODE=respirenet PROTOCOL=oracle $RUN_OFFICIAL 202506 "$option" 8 || echo "$name on fold 4 failed; continuing."
    fi
done

# ---------- B. Rerun the random-split runs affected by degenerate epochs ----------
python scripts/find_degenerate_runs.py experiments/random_split/CustomCNNCosine_val_seed_2025??_oracle/train.log | tee experiments/degenerate_runs.txt
for log in $(grep "^AFFECTED" experiments/degenerate_runs.txt | awk '{print $2}'); do
    dir=$(dirname "$log")
    target="${dir}_old"
    [ -e "$target" ] && target="${dir}_old_$(date '+%Y%m%d%H%M%S')"
    echo "Moving $dir to $target to train it again."
    mv "$dir" "$target"
done
$RUN_SPLITS || echo "Split sweep failed."

# ---------- C. Summary ----------
{
    echo "NIGHT SUMMARY ($(date '+%F %T')) - oracle protocol: checkpoints selected on the TEST set"
    echo
    echo "RespireNet fold 4 (RespireNet reports 68.5% on this test fold):"
    printf "%-18s %9s %9s %9s\n" "Model" "Test Se" "Test Sp" "Test ICBHI"
    for name in CustomCNNCosine EfficientNet ResNet VGG19; do
        dir="experiments/respirenet_split/${name}_val_seed_202506_oracle"
        if completed "$dir"; then
            printf "%-18s %9s %9s %9s\n" "$name" \
                "$(grep '^Sensitivity:' "$dir/evaluate.log" | awk '{print $2}')" \
                "$(grep '^Specificity:' "$dir/evaluate.log" | awk '{print $2}')" \
                "$(grep '^ICBHI Score:' "$dir/evaluate.log" | awk '{print $3}')"
        else
            printf "%-18s %9s\n" "$name" "no results"
        fi
    done
    echo
    echo "Runs trained again because of degenerate epochs:"
    grep "^AFFECTED" experiments/degenerate_runs.txt || echo "  none"
    echo
    cat experiments/oracle_splits_summary.txt 2>/dev/null
} | tee "$SUMMARY"

echo "########## NIGHT PLAN FINISHED: $(date '+%F %T') ##########"