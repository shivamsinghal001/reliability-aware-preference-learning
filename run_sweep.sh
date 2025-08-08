#!/usr/bin/env bash
set -euo pipefail

# FOLDS=(4)
# LRS=(2e-6 5e-6 1e-5 2e-5 3e-5 5e-5)
# LRS=(2e-6 5e-6)
# EPOCHs=(1 2 3 5)

FOLDS=(0 1 2 3 4)
LRS=(2e-05)
EPOCHs=(1)


TS_MAX_JOBS=8

# Start ts with 6 slots
ts -S "$TS_MAX_JOBS"

for fold in "${FOLDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for epoch in "${EPOCHs[@]}"; do
      ts ./run_one.sh "$fold" "$lr" "$epoch"
    done
  done
done

echo "All jobs queued.  Monitor with: ts -l"
