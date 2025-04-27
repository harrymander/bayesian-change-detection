#!/bin/sh

set -e

for trim in --trim --no-trim; do
    for num_samples in 100 200 300 400 500 1000 2000 3000 4000 5000 10000; do
        uv run time_bcdm.py \
            --output=complexity.json \
            --num-samples="$num_samples" \
            "$trim" \
            HEAD
    done
done
