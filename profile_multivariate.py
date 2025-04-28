import cProfile
import os.path
import shutil
import subprocess
import sys
from functools import partial
from tempfile import TemporaryDirectory

import click
import numpy as np

# Import these to avoid additional overhead from imports in profiling
import scipy.linalg.lapack
import scipy.special
import scipy.stats  # noqa: F401
from scipy.stats import multivariate_t  # noqa: F401

from bayesian_change_detection.multivariate import multivariate_bcdm
from multivariate_examples import generate_random_piecewise_data


def run_snakeviz(file: str) -> int:
    args = []
    cmd = "snakeviz"
    if shutil.which(cmd):
        args.append(cmd)
    elif shutil.which("uvx"):
        args.extend(("uvx", cmd))
    else:
        raise RuntimeError(f"{cmd} or uvx not installed")

    return subprocess.run([*args, "--", file]).returncode


@click.command()
@click.option(
    "--snakeviz",
    "-v",
    is_flag=True,
    help="Run snakeviz to visualize the profile in a browser.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    help="Path to write profile to.",
)
@click.option(
    "--trim/--no-trim",
    "trim_support",
    show_default=True,
    help="Whether to trim support.",
)
@click.option(
    "--num-samples",
    "-n",
    default=10_000,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of samples in the test data to perform change detection on.",
)
def main(
    output: str | None,
    snakeviz: bool,
    trim_support: bool,
    num_samples: int,
) -> None:
    def run_profile(filename: str | None) -> None:
        y = generate_random_piecewise_data(
            rng=np.random.default_rng(42),
            num_samples=num_samples,
            mean_mean=0,
            mean_var=2,
            var=1,
            hazard=0.1,
        )[0]

        kwargs: dict = dict(cov=2, hazard=0.1)
        if trim_support:
            kwargs.update(max_num_probs=20, min_prob=1e-6)

        cProfile.runctx(
            "run_segmentation(x, y)",
            globals={},
            locals={
                "run_segmentation": partial(multivariate_bcdm, **kwargs),
                "x": np.ones_like(y),
                "y": y,
            },
            filename=filename,
        )

    ret = 0
    if not snakeviz:
        run_profile(output)
    elif output:
        run_profile(output)
        ret = run_snakeviz(output)
    else:
        with TemporaryDirectory() as tempdir:
            output = os.path.join(tempdir, "profile.prof")
            run_profile(output)
            ret = run_snakeviz(output)

    sys.exit(ret)


if __name__ == "__main__":
    main()
