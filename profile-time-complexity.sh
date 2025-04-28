#!/bin/sh

set -e

SCRIPT="$(mktemp ./.time_bcdm.XXXX.py)"
cp ./time_bcdm.py "$SCRIPT"
trap 'rm -rf -- "$SCRIPT"' EXIT

profile_all_sample_sizes() {
    for num_samples in $(seq 100 100 500) $(seq 1000 1000 10000) $(seq 12500 2500 20000); do
        uv run "$SCRIPT" \
            --options="${1:-"{}"}" \
            --output=complexity.json \
            --num-samples="$num_samples" \
            --num-loops=50 \
            e0c6c2d0ae80ca0e264a779b30b7ea02e0c0c5fb
    done
}

profile_all_sample_sizes '{"trim": {}}'
profile_all_sample_sizes '{"trim": {"max_num_probs": 50, "min_prob": 1e-6}}'
profile_all_sample_sizes '{"trim": {"max_num_probs": 10, "min_prob": 1e-6}}'
