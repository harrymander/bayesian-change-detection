import cProfile
import os.path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import click
import numpy as np

# Import these to avoid additional overhead from imports in profiling
import scipy.linalg.lapack
import scipy.special
import scipy.stats  # noqa: F401
from scipy.stats import multivariate_t  # noqa: F401

from bayesian_change_detection import multivariate_bcdm


def run_segmentation(x: np.ndarray, y: np.ndarray) -> None:
    multivariate_bcdm(
        x,
        y,
        hazard=0.02,
        cov=1e6,
        shape=1e-3,
        scale=1e-6,
    )


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
@click.option("--snakeviz", "-v", is_flag=True)
@click.option("--output", "-o", type=click.Path(dir_okay=False, writable=True))
def main(output: str | None, snakeviz: bool) -> None:
    # Generate triangular data
    rng = np.random.default_rng(42)
    samples = 1000
    X = np.linspace(0, 3 * 2 * np.pi, samples)
    Y = -np.arcsin(np.sin(X)) * 2 / np.pi + 0.1 * rng.standard_normal(samples)
    X = np.c_[np.ones_like(X), X]

    def run_profile(filename: str | None) -> None:
        cProfile.runctx(
            "run_segmentation(x, y)",
            globals={},
            locals={
                "run_segmentation": run_segmentation,
                "x": X,
                "y": Y,
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
