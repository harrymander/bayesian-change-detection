from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage
from tqdm import tqdm

from bayesian_change_detection import MultivariateBcdm
from examples import triangle_wave

FIGSIZE = (20, 10)


def plot_posterior_probabilities(
    ax: Axes,
    log_posterior: np.ndarray,
    *,
    xlim=None,
    trim_zero: bool = True,
    **kwargs,
) -> tuple[np.ndarray, AxesImage]:
    samples = len(log_posterior)
    if xlim is None:
        x0, x1 = (0, samples - 1)
    else:
        x0, x1 = xlim

    posterior = np.exp(log_posterior)
    if trim_zero:
        zeros = np.where(np.isclose(posterior, 0).all(axis=1))[0]
        if zeros.size:
            zero_row = zeros[0]
            posterior = posterior[:zero_row]

    kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "cmap": "gray_r",
        "norm": LogNorm(vmin=1e-4, vmax=1),
        "extent": (x0, x1, 0, len(posterior)),
    } | kwargs
    return posterior, ax.imshow(posterior, **kwargs)  # type: ignore


def triangular(rng: np.random.Generator):
    """Simple example with triangular wave data."""

    samples = 500

    # Create input and outputs.
    X = np.linspace(0, 3 * 2 * np.pi, samples)
    noise = 0.1 * rng.standard_normal(samples)
    Y = triangle_wave(2 * np.pi, X - np.pi / 2) + noise

    # Determine location of true boundaries.
    true_boundaries = np.pi * np.arange(0, 6) + np.pi / 2
    true_boundaries = np.sort(true_boundaries[true_boundaries <= X.max()])

    rate = 0.001
    omega = 1.0e-3 * np.eye(2)
    sigma = 1.0e-6
    samples = 500

    model = MultivariateBcdm(
        2,
        hazard=rate,
        prior_cov=omega,
        prior_shape=sigma,
    )

    # Update the segment length hypotheses given the data.
    for x, y in tqdm(zip(X, Y, strict=True), total=len(X)):
        model.update(np.array([1, x]), y)

    axes = plt.subplots(3, sharex=True, figsize=FIGSIZE)[1]
    axes[0].plot(X, Y, "-o")
    posterior, im = plot_posterior_probabilities(
        axes[1], model.log_posterior(), xlim=X[[0, -1]]
    )
    axes[1].plot(X, posterior.argmax(axis=0), color="red")
    for ax in axes[1:]:
        ax.set_ylabel("Run length")
    for ax in axes:
        for boundary in true_boundaries:
            ax.axvline(boundary, color="red", linestyle="--")

    pred = model.log_predictive()
    plot_posterior_probabilities(
        axes[2],
        pred.T,
        xlim=X[[0, -1]],
        norm=None,
        origin="upper",
    )

    plt.tight_layout()
    plt.colorbar(im, ax=axes)


def generate_random_piecewise_data(rng, varx, mean0, var0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    data = []
    cps = []
    meanx = mean0
    for t in range(0, T):
        if rng.random() < cp_prob:
            meanx = rng.normal(mean0, var0)
            cps.append(t)
        data.append(rng.normal(meanx, varx))
    return data, cps


def random_piecewise(rng: np.random.Generator):
    T = 500  # Number of observations.
    hazard = 1 / 100  # Constant prior on changepoint probability.
    mean0 = 0  # The prior mean on the mean parameter.
    var0 = 2  # The prior variance for mean parameter.
    varx = 1  # The known variance of the data.

    data, cps = generate_random_piecewise_data(
        rng, varx, mean0, var0, T, hazard
    )
    model = MultivariateBcdm(1, prior_cov=var0, hazard=hazard)
    for y in tqdm(data):
        model.update(1, y)

    axes = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE)[1]
    axes[0].plot(data, "-o")
    for cp in cps:
        for ax in axes:
            ax.axvline(cp, color="red", linestyle="--")

    im = plot_posterior_probabilities(
        axes[1],
        model.log_posterior(),
        trim_zero=False,
    )[1]
    plt.tight_layout()
    plt.colorbar(im, ax=axes)


EXAMPLES = {
    "random": random_piecewise,
    "triangular": triangular,
}


@click.command()
@click.argument(
    "examples",
    type=click.Choice(list(EXAMPLES.keys())),
    nargs=-1,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def main(examples, output: Path | None) -> None:
    if output and len(examples) != 1:
        raise click.UsageError(
            "--output can only be specified with a single example"
        )

    if not examples:
        examples = EXAMPLES.keys()
    for example in examples:
        click.echo(f"Running '{example}' example")
        EXAMPLES[example](np.random.default_rng(42))

    if output:
        plt.savefig(output, format=None if output.suffix else "png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
