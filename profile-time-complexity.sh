#!/bin/sh

set -e

SCRIPT="$(mktemp ./.time_bcdm.XXXX.py)"
cp ./time_bcdm.py "$SCRIPT"
trap 'rm -rf -- "$SCRIPT"' EXIT

for trim in --trim --no-trim; do
    for num_samples in $(seq 100 100 500) $(seq 1000 1000 10000); do
        uv run "$SCRIPT" \
            --output=complexity.json \
            --num-samples="$num_samples" \
            --num-loops=50 \
            "$trim" \
            e0c6c2d0ae80ca0e264a779b30b7ea02e0c0c5fb
    done
done
