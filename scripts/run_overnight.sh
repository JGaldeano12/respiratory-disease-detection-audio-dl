#!/bin/bash
# Unattended experiment plan (about 18 hours). Every decision is taken on the VALIDATION set;
# the test results are only collected and reported.
#
#   A. Segment duration: Custom CNN (three phases) on the official split with 5, 6 and 7 s
#      (8 s is reused from the previous run). The best duration is chosen by validation ICBHI.
#   B. Training schedule: Custom CNN with a single cosine-decay phase at the chosen duration.
#      The best Custom CNN variant (three phases vs cosine) is chosen by validation ICBHI.
#   C. EfficientNet at the chosen duration (skipped if that run already exists, e.g. for 8 s).
#   D. Robustness: the chosen Custom CNN configuration on several random patient-wise splits.
#
# Completed runs (those with an evaluate.log) are skipped, so the script can be relaunched safely.
#
# Usage:
#   bash scripts/run_overnight.sh

cd "${APP_DIR:-/app}" || exit 1

VAL_SEED=202506
DURATIONS="8 7 6 5"            # On ties, the first one listed (the current default) is kept.
ROBUSTNESS_SEEDS="202508 202509 202510"
RUN_OFFICIAL="${RUN_OFFICIAL:-bash scripts/run_official_split.sh}"
RUN_SEEDS="${RUN_SEEDS:-bash scripts/run_split_seeds.sh}"
SUMMARY="experiments/overnight_summary.txt"

# Output folder of an official-split run, as named by run_official_split.sh.
run_dir() {  # $1 = model name (e.g. CustomCNN), $2 = duration
    local suffix=""
    [ "$2" != "8" ] && suffix="_${2}s"
    echo "experiments/official_split/${1}_val_seed_${VAL_SEED}${suffix}"
}

# Best validation ICBHI of a run, or -1 if the run is missing or incomplete.
val_score() {
    local log="$1/train.log"
    local score
    score=$(grep "Best ICBHI Score during training" "$log" 2>/dev/null | tail -1 | awk '{print $NF}')
    echo "${score:--1}"
}

# Run an official-split experiment unless it has already been completed.
run_official() {  # $1 = model option, $2 = model name, $3 = duration
    if [ -f "$(run_dir "$2" "$3")/evaluate.log" ]; then
        echo "Skipping $2 with $3 s (already completed)."
    else
        $RUN_OFFICIAL "$VAL_SEED" "$1" "$3"
    fi
}

echo "########## OVERNIGHT PLAN STARTED: $(date '+%F %T') ##########"

# ---------- A. Segment duration ----------
for d in $DURATIONS; do
    run_official custom_cnn CustomCNN "$d"
done

BEST_DURATION=""
BEST_VAL=-1
for d in $DURATIONS; do
    v=$(val_score "$(run_dir CustomCNN "$d")")
    echo "Custom CNN, $d s -> validation ICBHI: $v"
    if awk -v a="$v" -v b="$BEST_VAL" 'BEGIN { exit !(a > b) }'; then
        BEST_VAL="$v"; BEST_DURATION="$d"
    fi
done
[ -n "$BEST_DURATION" ] || { echo "No completed duration run; stopping."; exit 1; }
echo ">>> Chosen duration (by validation): $BEST_DURATION s"

# ---------- B. Training schedule ----------
run_official custom_cnn_cosine CustomCNNCosine "$BEST_DURATION"

VAL_PHASES=$(val_score "$(run_dir CustomCNN "$BEST_DURATION")")
VAL_COSINE=$(val_score "$(run_dir CustomCNNCosine "$BEST_DURATION")")
if awk -v a="$VAL_COSINE" -v b="$VAL_PHASES" 'BEGIN { exit !(a > b) }'; then
    BEST_CNN="custom_cnn_cosine"
else
    BEST_CNN="custom_cnn"
fi
echo ">>> Chosen Custom CNN variant (by validation): $BEST_CNN (three phases: $VAL_PHASES, cosine: $VAL_COSINE)"

# ---------- C. EfficientNet ----------
run_official efficientnet EfficientNet "$BEST_DURATION"

# ---------- D. Robustness over random splits ----------
MODEL="$BEST_CNN" DURATION="$BEST_DURATION" $RUN_SEEDS $ROBUSTNESS_SEEDS

# ---------- Summary ----------
{
    echo "OVERNIGHT SUMMARY ($(date '+%F %T'))"
    echo "Decisions taken on validation: duration = $BEST_DURATION s, Custom CNN variant = $BEST_CNN"
    echo
    printf "%-18s %8s %10s %9s %9s %9s\n" "Model" "Duration" "Val ICBHI" "Test Se" "Test Sp" "Test ICBHI"
    for dir in experiments/official_split/*_val_seed_${VAL_SEED}*; do
        [ -f "$dir/evaluate.log" ] || continue
        name=$(basename "$dir")
        model=${name%%_val_seed_*}
        duration=$(echo "$name" | grep -oE '_[0-9]+s$' | tr -d '_s')
        printf "%-18s %8s %10s %9s %9s %9s\n" "$model" "${duration:-8} s" "$(val_score "$dir")" \
            "$(grep '^Sensitivity:' "$dir/evaluate.log" | awk '{print $2}')" \
            "$(grep '^Specificity:' "$dir/evaluate.log" | awk '{print $2}')" \
            "$(grep '^ICBHI Score:' "$dir/evaluate.log" | awk '{print $3}')"
    done
} | tee "$SUMMARY"

echo "########## OVERNIGHT PLAN FINISHED: $(date '+%F %T') ##########"