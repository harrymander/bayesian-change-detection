import textwrap
import timeit

import click
import numpy as np
import scipy.special
from numpy.testing import assert_allclose


def logsumexp_sparse1(x):
    return scipy.special.logsumexp(x[~np.isneginf(x)])


def logsumexp_sparse2(x):
    return scipy.special.logsumexp(x[np.isfinite(x)])


@click.command(context_settings={"show_default": True})
@click.option(
    "--proportion-finite",
    "--finite",
    "-p",
    type=click.FloatRange(0, 1.0),
    default=0.1,
    help="""Proportion of the test array that is finite (lower number means
    sparser array).""",
)
@click.option(
    "--size",
    "-s",
    type=click.IntRange(1),
    default=100_000,
    help="""Size of the test array.""",
)
@click.option(
    "--number",
    "-n",
    type=click.IntRange(1),
    default=10_000,
    help="""Number of repeats for timing.""",
)
@click.option(
    "--seed",
    default="42",
    help="""Random seed; encoded as string. Pass empty string to use
    unpredictable entropy from system.""",
)
@click.option("--check/--no-check", default=True)
@click.option("--sort/--no-sort", default=True)
def main(
    proportion_finite: float,
    size: int,
    number: int,
    seed: str,
    check: bool,
    sort: bool,
) -> None:
    rng = np.random.default_rng(list(seed.encode()) if seed else None)
    x = np.log(rng.random(size))
    x[rng.random(size) > proportion_finite] = -np.inf

    funcs = (scipy.special.logsumexp, logsumexp_sparse1, logsumexp_sparse2)
    ref = funcs[0](x)
    for f in funcs[1:]:
        val = f(x)
        try:
            assert_allclose(
                val,
                ref,
                strict=True,  # type: ignore  # not supported on numpy <2
            )
        except AssertionError as e:
            name = f.__name__
            ref_name = funcs[0].__name__
            assertion_msg = textwrap.indent(str(e), "  ")
            msg = f"{name} output does not match {ref_name}:{assertion_msg}"
            if check:
                raise AssertionError(msg) from e
            click.secho(f"Warning: {msg}", fg="yellow", err=True)

    times = [
        (
            f.__name__,
            timeit.timeit("f(x)", globals={"f": f, "x": x}, number=number),
        )
        for f in funcs
    ]
    if sort:
        times.sort(key=lambda v: v[1])

    for name, time in times:
        click.echo(f"{name}: {time / number * 1e6:g} μs", err=True)


if __name__ == "__main__":
    main()
