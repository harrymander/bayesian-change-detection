from collections.abc import Sequence
from itertools import batched, chain, pairwise
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from bayesian_change_detection import multivariate_bcdm
from bayesian_change_detection.multivariate import (
    MultivariateBcdmResults,
    NigParams,
    NigPrior,
)
from examples import generate_random_piecewise_data

FIGSIZE = (20, 10)
DATA_DIR = Path(__file__).parent / "data"


def plot_probabilities(
    ax: Axes,
    log_probabilities: np.ndarray,
    *,
    x: np.ndarray | None = None,
    use_x_for_y: bool = True,
    trim_zero: bool = True,
    add_colorbar: bool = True,
    label: str | None = "Probability",
    **kwargs,
) -> tuple[np.ndarray, AxesImage]:
    samples = len(log_probabilities)
    if x is None:
        x = np.arange(samples)
    elif x.shape != (samples,):
        raise ValueError(f"invalid shape for x: must be ({samples},)")

    probs = np.exp(log_probabilities)
    if trim_zero:
        zeros = np.where(np.isclose(probs, 0).all(axis=1))[0]
        if zeros.size:
            zero_row = zeros[0]
            probs = probs[:zero_row]

    idx_y_end = len(probs) - 1
    if use_x_for_y:
        y0 = x[0]
        y1 = x[idx_y_end]
    else:
        y0 = 0
        y1 = idx_y_end

    ax.set_ylabel("Run length")
    kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "cmap": "gray_r",
        "norm": LogNorm(vmin=1e-4, vmax=1),
        "extent": (x[0], x[-1], y0, y1),
    } | kwargs

    im = ax.imshow(probs, **kwargs)  # type: ignore
    if add_colorbar:
        inset = ax.inset_axes((0.025, 0.87, 0.2, 0.05))
        plt.colorbar(im, cax=inset, orientation="horizontal")
        if label:
            ax.text(
                0.5,
                1.1,
                label,
                horizontalalignment="center",
                verticalalignment="bottom",
                transform=inset.transAxes,
            )

    return probs, im


def axvlines(
    ax: Axes,
    xvals: np.ndarray | Sequence[float],
    *args,
    **kwargs,
) -> list[Line2D]:
    """Adds multiple vertical lines with the same color and label such that
    only a single legend entry is generated for the lines."""
    first_line = ax.axvline(xvals[0], *args, **kwargs)
    kwargs.pop("label", None)
    kwargs["color"] = first_line.get_color()
    return [first_line, *(ax.axvline(x, *args, **kwargs) for x in xvals[1:])]


def triangular() -> None:
    """Simple example with triangular wave data."""
    rng = np.random.default_rng(42)

    samples = 500

    # Create input and outputs.
    t = np.linspace(0, 3 * 2 * np.pi, samples)
    noise = 0.1 * rng.standard_normal(samples)
    Y = -np.arcsin(np.sin(t)) * 2 / np.pi + noise
    true_changepoints = np.pi * np.arange(0, 6) + np.pi / 2

    X = np.c_[np.ones_like(t), t]  # add bias (intercept) term
    prior = NigPrior(
        2,
        cov=1e6,
        shape=1e-3,
        scale=1e-6,
    )
    res = multivariate_bcdm(X, Y, prior=prior, hazard=0.02)

    axes = plt.subplots(3, sharex=True, figsize=FIGSIZE)[1]
    axes[0].plot(t, Y, "-o")
    posterior = plot_probabilities(
        axes[1],
        res.log_posterior(),
        x=t,
        label="Posterior probability",
    )[0]
    axes[1].plot(t, t[posterior.argmax(axis=0)], color="green")

    plot_probabilities(
        axes[2],
        res.log_predictive,
        x=t,
        trim_zero=False,
        label="Predictive probability",
    )
    axes[2].plot(
        t,
        t[res.log_predictive.argmax(axis=0)],
        color="orange",
        linewidth=0.8,
        alpha=0.7,
    )

    changepoints = res.changepoints()
    for ax in axes:
        axvlines(
            ax,
            true_changepoints,
            color="red",
            linestyle="--",
            label="Actual changepoints",
        )
        axvlines(
            ax,
            t[changepoints],
            color="green",
            linestyle="--",
            label="Predicted changepoints",
        )

    for ax in axes[1:]:
        ax.set_ylabel("Run length")

    plot_segment_predictions(axes[0], prior, changepoints, X, Y, t)
    axes[0].legend()
    plt.tight_layout()


def random_piecewise() -> None:
    rng = np.random.default_rng(42)

    num_samples = 500
    hazard = 1 / 100  # Constant prior on changepoint probability.
    var0 = 2  # The prior variance for mean parameter.
    data, true_changepoints = generate_random_piecewise_data(
        rng=rng,
        num_samples=num_samples,
        var=1,
        mean_var=var0,
        mean_mean=0,
        hazard=hazard,
    )
    y = np.asarray(data)
    prior = NigPrior(1, cov=var0)
    res = multivariate_bcdm(
        np.ones_like(y),
        y,
        prior=prior,
        hazard=hazard,
    )

    axes = plt.subplots(3, 1, sharex=True, figsize=FIGSIZE)[1]
    axes[0].plot(data, "-o")
    changepoints = res.changepoints()
    for ax in axes:
        axvlines(
            ax,
            true_changepoints,
            color="red",
            linestyle="--",
            label="Actual changepoints",
        )
        axvlines(
            ax,
            changepoints,
            color="green",
            linestyle="--",
            label="Predicted changepoints",
        )

    axes[0].legend()

    plot_segment_predictions(axes[0], prior, changepoints, np.ones_like(y), y)
    posterior = plot_probabilities(
        axes[1],
        res.log_posterior(),
        trim_zero=False,
        label="Posterior probability",
    )[0]
    plot_probabilities(
        axes[2],
        res.log_predictive,
        trim_zero=False,
        label="Predictive probability",
    )
    axes[2].plot(
        res.log_predictive.argmax(axis=0),
        color="orange",
        linewidth=0.8,
        alpha=0.7,
    )

    axes[1].plot(posterior.argmax(axis=0), color="green")
    plt.tight_layout()


def load_well_data() -> tuple[np.ndarray, np.ndarray]:
    y = np.loadtxt(DATA_DIR / "well-data.txt", comments="#")
    assert y.ndim == 1
    return np.ones_like(y), y


def well_data_multivarate_bcdm(
    x: np.ndarray, y: np.ndarray, **kwargs
) -> MultivariateBcdmResults:
    return multivariate_bcdm(x, y, hazard=0.01, mean=1e5, scale=1e4, **kwargs)


def well_data() -> None:
    x, y = load_well_data()
    results = well_data_multivarate_bcdm(
        x,
        y,
        max_num_probs=20,
        min_prob=1e-12,
    )

    axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)[1]
    axes[0].plot(y)
    axes[0].set_ylabel("Nuclear magnetic response")
    axes[-1].set_xlabel("Time")
    plot_probabilities(
        axes[1],
        results.log_posterior(),
        label="Posterior probability",
    )

    changepoints = results.changepoints()
    for ax in axes:
        axvlines(
            ax,
            changepoints,
            color="black",
            linewidth=0.5,
            linestyle="--",
        )

    for segment in batched(chain((0,), changepoints, (len(x) - 1,)), 2):
        if len(segment) < 2:
            break
        for ax in axes:
            ax.axvspan(
                *segment,
                facecolor="yellow",
                edgecolor="none",
                alpha=0.3,
            )


def predict_segment(
    prior: NigPrior,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = NigParams.from_prior(prior).fit_regression(x, y)
    var = model.mvt_variance(x, axis="t")
    var[var < 0] = np.nan
    return model.mvt_mean(x, axis="t"), var


def atleast_2d_col(x: np.ndarray) -> np.ndarray:
    return np.expand_dims(x, 1) if x.ndim < 2 else x


def plot_segment_predictions(
    ax,
    prior: NigPrior,
    changepoints: list[int],
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray | None = None,
) -> None:
    n = len(x)
    if t is None:
        t = np.arange(n)

    for i1, i2 in pairwise(chain((0,), changepoints, (n,))):
        xseg = x[i1:i2]
        yseg = y[i1:i2]
        tseg = t[i1:i2]
        pmean, pvar = predict_segment(prior, atleast_2d_col(xseg), yseg)
        ax.plot(tseg, pmean, color="green")
        sd = np.sqrt(pvar)
        ax.plot(tseg, pmean + sd, color="green", linestyle="--")
        ax.plot(tseg, pmean - sd, color="green", linestyle="--")


EXAMPLES = {
    "random": random_piecewise,
    "triangular": triangular,
    "well": well_data,
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
        EXAMPLES[example]()

    if output:
        plt.savefig(output, format=None if output.suffix else "png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
