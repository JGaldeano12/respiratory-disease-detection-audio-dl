#!/bin/bash
# Unattended improvement plan for the single-phase Custom CNN (official split, 8 s).
# Every decision is taken on the VALIDATION set; the test results are only collected and reported.
#
#   1. Reference: checkpoint selection with a 3-epoch moving average ("smooth3"), adopted a priori
#      because single-epoch peaks of the small validation set were not representative.
#   2. One change at a time on top of the reference:
#        freqnorm  per-band mean normalization of each spectrogram (device/channel signature)
#        bandmask  random blanking of the 1.5-2 kHz band during training (Littmann 3200)
#        alphasqrt milder class weighting in the focal loss (square root of the balanced weights)
#      A change is accepted if it improves the validation criterion by at least MIN_DELTA.
#   3. If two or more changes are accepted, they are combined in one more run.
#   4. The configuration with the best validation criterion is run on several random splits.
#
# Validation criterion: best 3-epoch mean of the validation ICBHI score, the same for every run.
# Completed runs are skipped, so the script can be relaunched safely.
#
# Usage:
#   bash scripts/run_improvements.sh

cd "${APP_DIR:-/app}" || exit 1

VAL_SEED=202506
DURATION=8
MODEL=custom_cnn_cosine
MODEL_NAME=CustomCNNCosine
MIN_DELTA=0.005
ROBUSTNESS_SEEDS="202508 202509 202510"
RUN_OFFICIAL="${RUN_OFFICIAL:-bash scripts/run_official_split.sh}"
RUN_SEEDS="${RUN_SEEDS:-bash scripts/run_split_seeds.sh}"
SUMMARY="experiments/improvements_summary.txt"

REFERENCE_ARGS="--smoothing_window 3"
declare -A CHANGE_ARGS=( [freqnorm]="--freq_norm" [bandmask]="--band_mask" [alphasqrt]="--alpha sqrt" )
CHANGES="freqnorm bandmask alphasqrt"

# Output folder of an official-split run of this model, as named by run_official_split.sh.
run_dir() {  # $1 = variant name
    echo "experiments/official_split/${MODEL_NAME}_val_seed_${VAL_SEED}_$1"
}

# Validation criterion of a run: best 3-epoch mean of the validation ICBHI score (-1 if unavailable).
criterion() {  # $1 = run folder
    python - "$1/train.log" << 'PY'
import re
import sys

import numpy as np

try:
    text = open(sys.argv[1]).read()
except OSError:
    print(-1)
    sys.exit()

scores = [float(s) for s in re.findall(r"^ICBHI Score: ([\d.]+) - Sensitivity", text, re.M)]
if len(scores) < 3 or "Best ICBHI Score during training" not in text:
    print(-1)
else:
    print(f"{max(np.mean(scores[i - 2:i + 1]) for i in range(2, len(scores))):.5f}")
PY
}

greater() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'; }

# Run a variant unless it has already been completed.
run_variant() {  # $1 = variant name, $2 = training options
    if [ -f "$(run_dir "$1")/evaluate.log" ]; then
        echo "Skipping $1 (already completed)."
    else
        VARIANT="$1" TRAIN_ARGS="$2" $RUN_OFFICIAL "$VAL_SEED" "$MODEL" "$DURATION"
    fi
}

echo "########## IMPROVEMENT PLAN STARTED: $(date '+%F %T') ##########"

# ---------- 1. Reference ----------
run_variant smooth3 "$REFERENCE_ARGS"
REF=$(criterion "$(run_dir smooth3)")
[ "$REF" = "-1" ] && { echo "Reference run failed; stopping."; exit 1; }
echo ">>> Reference (smooth3) validation criterion: $REF"

BEST_NAME=smooth3; BEST_ARGS="$REFERENCE_ARGS"; BEST_SCORE="$REF"
ACCEPTED_NAMES=""; ACCEPTED_ARGS=""

# ---------- 2. One change at a time ----------
for change in $CHANGES; do
    name="smooth3_$change"
    args="$REFERENCE_ARGS ${CHANGE_ARGS[$change]}"
    run_variant "$name" "$args"
    score=$(criterion "$(run_dir "$name")")
    if greater "$score" "$(awk -v r="$REF" -v d="$MIN_DELTA" 'BEGIN { print r + d }')"; then
        echo ">>> $change: $score -> accepted (reference $REF)"
        ACCEPTED_NAMES="${ACCEPTED_NAMES}_$change"; ACCEPTED_ARGS="$ACCEPTED_ARGS ${CHANGE_ARGS[$change]}"
    else
        echo ">>> $change: $score -> not accepted (reference $REF)"
    fi
    if greater "$score" "$BEST_SCORE"; then BEST_NAME="$name"; BEST_ARGS="$args"; BEST_SCORE="$score"; fi
done

# ---------- 3. Combination of the accepted changes ----------
if [ "$(echo "$ACCEPTED_NAMES" | tr -cd '_' | wc -c)" -ge 2 ]; then
    name="smooth3${ACCEPTED_NAMES}"
    args="$REFERENCE_ARGS$ACCEPTED_ARGS"
    run_variant "$name" "$args"
    score=$(criterion "$(run_dir "$name")")
    echo ">>> Combination $name: $score"
    if greater "$score" "$BEST_SCORE"; then BEST_NAME="$name"; BEST_ARGS="$args"; BEST_SCORE="$score"; fi
fi
echo ">>> Chosen configuration (by validation): $BEST_NAME ($BEST_ARGS), criterion $BEST_SCORE"

# ---------- 4. Robustness over random splits ----------
MODEL="$MODEL" DURATION="$DURATION" VARIANT="$BEST_NAME" TRAIN_ARGS="$BEST_ARGS" $RUN_SEEDS $ROBUSTNESS_SEEDS

# ---------- Summary ----------
{
    echo "IMPROVEMENT SUMMARY ($(date '+%F %T'))"
    echo "Chosen configuration (by validation): $BEST_NAME ($BEST_ARGS)"
    echo
    printf "%-44s %11s %9s %9s %9s\n" "Run" "Val (3-ep)" "Test Se" "Test Sp" "Test ICBHI"
    for dir in experiments/official_split/*_val_seed_${VAL_SEED}*; do
        [ -f "$dir/evaluate.log" ] || continue
        printf "%-44s %11s %9s %9s %9s\n" "$(basename "$dir" | sed "s/_val_seed_${VAL_SEED}//")" "$(criterion "$dir")" \
            "$(grep '^Sensitivity:' "$dir/evaluate.log" | awk '{print $2}')" \
            "$(grep '^Specificity:' "$dir/evaluate.log" | awk '{print $2}')" \
            "$(grep '^ICBHI Score:' "$dir/evaluate.log" | awk '{print $3}')"
    done
    echo
    echo "Robustness over random splits ($BEST_NAME):"
    cat "experiments/split_seeds_${MODEL_NAME}_${DURATION}s_${BEST_NAME}/results.csv" 2>/dev/null
} | tee "$SUMMARY"

echo "########## IMPROVEMENT PLAN FINISHED: $(date '+%F %T') ##########"