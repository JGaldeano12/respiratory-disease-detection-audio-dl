"""
Find training runs affected by degenerate epochs.

A run is affected when, at some point, an epoch with sensitivity 0 or specificity 0 (a trivial
classifier) had the best ICBHI score so far. Such an epoch could be selected as the checkpoint and
could stop the training early. Runs where this never happened are unaffected by the fix in the
ICBHI callback, and rerunning them would give the same result. Runs trained with the fixed
callback (their log contains "Degenerate epoch") are reported as fixed and never as affected.

Usage:
    python scripts/find_degenerate_runs.py experiments/random_split/*_oracle/train.log
"""
import re
import sys

PATTERN = re.compile(r"^ICBHI Score: ([\d.]+) - Sensitivity: ([\d.]+) - Specificity: ([\d.]+)", re.M)

affected = []
for path in sys.argv[1:]:
    text = open(path).read()
    if "Degenerate epoch" in text:
        print(f"fixed     {path}  (trained with the fixed callback)")
        continue
    epochs = [tuple(map(float, match)) for match in PATTERN.findall(text)]
    best = 0.0
    for number, (score, sensitivity, specificity) in enumerate(epochs, start=1):
        degenerate = sensitivity == 0.0 or specificity == 0.0
        if score > best:
            if degenerate:
                affected.append(path)
                print(f"AFFECTED  {path}  (epoch {number}: ICBHI {score:.5f}, Se {sensitivity:.5f}, Sp {specificity:.5f}; {len(epochs)} epochs trained)")
                break
            best = score
    else:
        print(f"ok        {path}  ({len(epochs)} epochs trained)")

print(f"\n{len(affected)} affected run(s) out of {len(sys.argv) - 1}.")