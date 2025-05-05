#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from itertools import product
from typing import Any

import click


def get_short_rev(ref: str) -> str:
    return (
        subprocess.run(
            ("git", "rev-parse", "--short", ref, "--"),
            stdout=subprocess.PIPE,
            check=True,
        )
        .stdout.decode()
        .strip()
    )


def get_commit_summary(ref: str) -> str:
    return (
        subprocess.run(
            ("git", "log", "-1", "--oneline", "--no-decorate", ref, "--"),
            stdout=subprocess.PIPE,
            check=True,
        )
        .stdout.decode()
        .strip()
    )


def irange(start: int, stop: int, step: int) -> range:
    """Equivalent to `range(start, stop, step)`, but `stop` is included."""
    return range(start, stop + step, step)


def product_dicts(**kw: Iterable) -> Iterable[dict[str, Any]]:
    keys = kw.keys()
    yield from (
        dict(zip(keys, values, strict=True))
        for values in product(*kw.values())
    )


NUM_SAMPLES = [
    *irange(100, 500, 100),
    *irange(1000, 10000, 1000),
    *irange(12500, 20000, 2500),
    *irange(22000, 30000, 2000),
]
OPTIONS: list[dict] = [
    {},
    *product_dicts(min_prob=[1e-6, 0], max_num_probs=[50, 10]),
]


def profile(*, script: str, commits: Iterable[str], dry_run: bool):
    commits = set(get_short_rev(c) for c in commits)
    summaries = map(get_commit_summary, commits)
    print(
        f"Testing {len(commits)} commits:\n  {'\n  '.join(summaries)}",
        f"Testing over {len(NUM_SAMPLES)} sample sizes: {NUM_SAMPLES}",
        f"Testing options:\n  {'\n  '.join(map(json.dumps, OPTIONS))}",
        file=sys.stdout if dry_run else sys.stderr,
        sep="\n",
    )
    if dry_run:
        print(
            "Re-run without --dry-run/-n to perform timing.",
            file=sys.stderr,
        )
        return

    for commit, num_samples, options in product(commits, NUM_SAMPLES, OPTIONS):
        subprocess.run(
            (
                "uv",
                "run",
                script,
                f"--options={json.dumps(options)}",
                f"--num-samples={num_samples}",
                "--num-loops=10",
                "--output",
                f"complexity-{commit}.json",
                commit,
            ),
            check=True,
            env=os.environ | {"VIRTUAL_ENV": ""},
        )


@click.command()
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    help="Just print the configurations that would be tested and exit.",
)
@click.argument("commits", nargs=-1)
def main(commits: list[str], **kw) -> None:
    """
    Run time_bcdm.py over a range of sample sizes, options, and revisions.

    If COMMITS not passed, defaults to HEAD.
    """
    if not commits:
        commits = ["HEAD"]

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=".",
        prefix=".time_bcdm.",
        suffix=".py",
        delete=True,
        delete_on_close=False,
    ) as temp_script:
        with open("time_bcdm.py", "rb") as f:
            shutil.copyfileobj(f, temp_script)
        temp_script.close()
        profile(script=temp_script.name, commits=commits, **kw)


if __name__ == "__main__":
    main()
