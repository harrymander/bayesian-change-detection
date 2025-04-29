#!/bin/sh

set -e

SCRIPT="$(mktemp ./.time_bcdm.XXXX.py)"
cp ./time_bcdm.py "$SCRIPT"
trap 'rm -rf -- "$SCRIPT"' EXIT

profile_all_sample_sizes() {
    for num_samples in $(seq 100 100 500) $(seq 1000 1000 10000) $(seq 12500 2500 20000); do
        uv run "$SCRIPT" \
            --options="${1:-"{}"}" \
            --output=dtypes.json \
            --num-samples="$num_samples" \
            --num-loops=50 \
            1da8d41f25d310a922f92619454428117ad74efa
    done
}

profile_all_sample_sizes '{"max_num_probs": 50, "min_prob": 1e-6, "dtype": "float64"}'
profile_all_sample_sizes '{"max_num_probs": 50, "min_prob": 1e-6, "dtype": "float32"}'
