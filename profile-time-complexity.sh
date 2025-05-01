#!/bin/sh

set -e

SCRIPT="$(mktemp ./.time_bcdm.XXXX.py)"
cp ./time_bcdm.py "$SCRIPT"
trap 'rm -rf -- "$SCRIPT"' EXIT

COMMIT=HEAD

profile_all_sample_sizes() {
    for num_samples in $(seq 100 100 500) $(seq 1000 1000 10000) $(seq 12500 2500 20000) $(seq 22000 2000 30000); do
        uv run "$SCRIPT" \
            --options="${1:-"{}"}" \
            --output="complexity-$(git rev-parse --short "$COMMIT").json" \
            --num-samples="$num_samples" \
            --num-loops=10 \
            "$(git rev-parse "$COMMIT")"
    done
}

profile_all_sample_sizes '{}'
profile_all_sample_sizes '{"max_num_probs": 50, "min_prob": 1e-6}'
profile_all_sample_sizes '{"max_num_probs": 10, "min_prob": 1e-6}'
profile_all_sample_sizes '{"max_num_probs": 50, "min_prob": 0}'
profile_all_sample_sizes '{"max_num_probs": 10, "min_prob": 0}'
