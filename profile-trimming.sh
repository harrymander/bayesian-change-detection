#!/bin/sh

set -e

REFS=$(git rev-list --reverse fc074abec112f6e42bd7369876f5ac28181c8d62~..e0c6c2d0ae80ca0e264a779b30b7ea02e0c0c5fb)
run_prof() {
    for ref in $REFS; do
        uv run time_bcdm.py --output=times.json "$@" -- "$ref"
    done
}

run_prof --no-trim
run_prof --trim
