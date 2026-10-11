#!/bin/bash
# Oracle protocol (RespireNet-like) over several patient-wise random 80:20 splits.
# For every split seed, one single-phase cosine Custom CNN is trained (fixed training seed, 8 s)
# with the checkpoint selected on the TEST set. ALL the results are recorded, so the report gives
# the distribution over splits (mean, standard deviation, best and worst), not a single split.
#
# The sweep can be stopped at any moment and relaunched with the same command: completed splits
# (including the split 202506 trained earlier with the same settings) are skipped, and an
# interrupted split starts again from the beginning of its training (its dataset is reused if it
# was already generated). A split counts as completed only when its test evaluation has finished.
#
# To stop it:
#   pkill -f run_oracle_splits.sh; pkill -f run_official_split.sh
#   pkill -f "src.data.generate_dataset|src/data/augment.py|src/training/|src/evaluation/"
#
# Usage:
#   bash scripts/run_oracle_splits.sh                          # split seeds 202501-202512
#   bash scripts/run_oracle_splits.sh 202601 202602 202603     # other split seeds

cd "${APP_DIR:-/app}" || exit 1
export PYTHONPATH="${APP_DIR:-/app}"

SPLIT_SEEDS=("${@:-202501 202502 202503 202504 202505 202506 202507 202508 202509 202510 202511 202512}")
SPLIT_SEEDS=(${SPLIT_SEEDS[@]})
TRAIN_SEED=1234
DURATION=8
RUN_OFFICIAL="${RUN_OFFICIAL:-bash scripts/run_official_split.sh}"
OUT_DIR="experiments/oracle_splits"
RESULTS="$OUT_DIR/results.csv"
SUMMARY="experiments/oracle_splits_summary.txt"
mkdir -p "$OUT_DIR"

run_dir() { echo "experiments/random_split/CustomCNNCosine_val_seed_${1}_oracle"; }

# A split is completed only if its evaluation log contains the final metrics
# (the log file is created as soon as the evaluation starts, so its mere existence is not enough).
completed() { grep -q "^ICBHI Score:" "$(run_dir "$1")/evaluate.log" 2>/dev/null; }

echo "########## ORACLE SPLIT SWEEP STARTED: $(date '+%F %T') ##########"

for seed in "${SPLIT_SEEDS[@]}"; do
    if completed "$seed"; then
        echo "Skipping split seed $seed (already completed)."
    else
        SPLIT_MODE=random PROTOCOL=oracle TRAIN_SEED="$TRAIN_SEED" $RUN_OFFICIAL "$seed" custom_cnn_cosine "$DURATION" \
            || echo "Split seed $seed failed; continuing with the next one."
    fi
done

# Collect every split (test metrics and test composition) and summarise the distribution.
python - "$RESULTS" "${SPLIT_SEEDS[@]}" << 'PY' | tee "$SUMMARY"
import contextlib
import io
import os
import re
import sys

import numpy as np
import pandas as pd

from src.data.divide import get_split

results_path, seeds = sys.argv[1], sys.argv[2:]
rows = []
for seed in seeds:
    log = f"experiments/random_split/CustomCNNCosine_val_seed_{seed}_oracle/evaluate.log"
    text = open(log).read() if os.path.exists(log) else ""
    if not re.search(r"^ICBHI Score: ", text, re.M):
        print(f"Split seed {seed}: no results (not run yet or interrupted)")
        continue
    metric = lambda name: float(re.search(rf"^{name}: ([\d.]+)", text, re.M).group(1))

    # Composition of the test set of this split (same split as the one used for training).
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, test = get_split('/app/src/resources/aux_respiratory_cycles.npy', int(seed), 80, 0, 'random')
    counts = {c: int((test[:, 6] == c).sum()) for c in ['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle']}

    rows.append({'split_seed': int(seed), 'test_se': metric('Sensitivity'), 'test_sp': metric('Specificity'),
                 'test_icbhi': metric('ICBHI Score'), 'test_patients': len(set(test[:, 0])),
                 'normal': counts['Healthy'], 'crackle': counts['Crackle'], 'wheeze': counts['Wheeze'], 'both': counts['Wheeze & Crackle']})

df = pd.DataFrame(rows)
df.to_csv(results_path, index=False)

print("ORACLE SPLIT SWEEP - Custom CNN (cosine), checkpoint selected on the TEST set, one model per split")
print()
print(df.to_string(index=False))
if len(df):
    print()
    for col in ['test_se', 'test_sp', 'test_icbhi']:
        print(f"{col}: mean {df[col].mean():.4f} ± {df[col].std():.4f} | min {df[col].min():.4f} | max {df[col].max():.4f}")
PY

echo "########## ORACLE SPLIT SWEEP FINISHED: $(date '+%F %T') ##########"