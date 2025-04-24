import cProfile
import os.path
import shutil
import subprocess
import sys
from functools import partial
from tempfile import TemporaryDirectory

import click

# Import these to avoid additional overhead from imports in profiling
import scipy.linalg.lapack
import scipy.special
import scipy.stats  # noqa: F401
from scipy.stats import multivariate_t  # noqa: F401

from multivariate_examples import (
    load_well_data,
    well_data_multivarate_bcdm,
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
@click.option("--trim-support", "--trim", is_flag=True)
def main(output: str | None, snakeviz: bool, trim_support: bool) -> None:
    def run_profile(filename: str | None) -> None:
        x, y = load_well_data()

        run_segmentation = (
            partial(
                well_data_multivarate_bcdm, max_num_probs=20, min_prob=1e-6
            )
            if trim_support
            else well_data_multivarate_bcdm
        )
        cProfile.runctx(
            "run_segmentation(x, y)",
            globals={},
            locals={
                "run_segmentation": run_segmentation,
                "x": x,
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
