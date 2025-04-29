import cProfile
import os.path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import click

# Import these to avoid additional overhead from imports in profiling
import scipy.linalg.lapack
import scipy.special
import scipy.stats  # noqa: F401
from scipy.stats import multivariate_t  # noqa: F401

from time_bcdm import (
    Options,
    OptionsJsonOrPath,
    generate_profiling_data,
    get_profiling_function,
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
@click.option("--options", "-c", type=OptionsJsonOrPath())
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
    num_samples: int,
    options: Options | None,
) -> None:
    if options is None:
        options = Options()

    def run_profile(filename: str | None) -> None:
        data = generate_profiling_data(num_samples, options)
        cProfile.runctx(
            "f()",
            globals={},
            locals=dict(f=get_profiling_function(data, options)),
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
