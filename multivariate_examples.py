from collections.abc import Callable, Sequence
from functools import partial
from itertools import batched, chain, cycle, pairwise
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from bayesian_change_detection import multivariate_bcdm as _multivariate_bcdm
from bayesian_change_detection.multivariate import (
    MultivariateBcdmResults,
    NigPrior,
    multivariate_bcdm_datagetter,
)
from examples import generate_random_piecewise_data

FIGSIZE = (20, 10)
DATA_DIR = Path(__file__).parent / "data"


def multivariate_bcdm(
    x, y, *, progress_desc: str | None = None, **kwargs
) -> MultivariateBcdmResults:
    """Wrap multivariate_bcdm with progress bar."""
    with tqdm.tqdm(total=len(x), unit="", desc=progress_desc) as pbar:
        return _multivariate_bcdm(
            x,
            y,
            update_hook=lambda *_: pbar.update(),
            **kwargs,
        )


def plot_probabilities(
    ax: Axes,
    log_probabilities: np.ndarray,
    *,
    x: np.ndarray | None = None,
    use_x_for_y: bool = True,
    trim_zero: bool = True,
    add_colorbar: bool = True,
    label: str | None = "Probability",
    time_axis: int = 0,
    **kwargs,
) -> tuple[np.ndarray, AxesImage]:
    if time_axis not in (0, 1):
        raise ValueError("time_axis must be 0 or 1")

    samples: int = log_probabilities.shape[time_axis]
    if time_axis == 0:
        log_probabilities = log_probabilities.T

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


def triangular(reset_time: bool) -> None:
    """Simple example with triangular wave data."""
    rng = np.random.default_rng(42)

    samples = 500

    # Create input and outputs.
    t = np.linspace(0, 3 * 2 * np.pi, samples)
    noise = 0.1 * rng.standard_normal(samples)
    Y = -np.arcsin(np.sin(t)) * 2 / np.pi + noise
    true_changepoints = np.pi * np.arange(0, 6) + np.pi / 2

    p = 2
    prior = NigPrior(
        p,
        cov=1e6,
        shape=1e-3,
        scale=1e-6,
    )
    n = len(t)

    def data_getter(
        i: int,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | float]:
        if not reset_time:
            return np.array([1.0, t[i]]), Y[i]

        m = mask.size
        x = t[i + 1 - m : i + 1]
        X = np.c_[np.ones_like(x), x - x[0]]
        return X, np.repeat(Y[i], m)

    with tqdm.tqdm(total=n) as pbar:
        res = multivariate_bcdm_datagetter(
            n,
            p,
            data_getter,
            prior=prior,
            hazard=0.02,
            update_hook=lambda *_: pbar.update(),
        )

    axes = plt.subplots(3, sharex=True, figsize=FIGSIZE)[1]
    axes[0].plot(t, Y, "-o")
    posterior = plot_probabilities(
        axes[1],
        res.log_posterior(),
        x=t,
        label="Posterior probability",
    )[0]
    axes[1].plot(t, t[posterior.argmax(axis=0)], color="green")

    log_predictive = res.log_predictive.full()
    plot_probabilities(
        axes[2],
        log_predictive,
        x=t,
        trim_zero=False,
        label="Predictive probability",
    )
    axes[2].plot(
        t,
        t[log_predictive.argmax(axis=1)],
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

    def get_regressor(i0: int, i1: int) -> NDArray[np.float64]:
        x = t[i0:i1]
        if reset_time:
            x = x - x[0]
        return np.c_[np.ones_like(x), x]

    plot_segment_predictions(
        axes[0],
        prior,
        changepoints,
        get_regressor,
        Y,
        t,
    )
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

    plot_segment_predictions(
        axes[0],
        prior,
        changepoints,
        lambda i0, i1: np.ones((i1 - i0, 1), np.float64),
        y,
    )
    posterior = plot_probabilities(
        axes[1],
        res.log_posterior(),
        trim_zero=False,
        label="Posterior probability",
    )[0]
    log_predictive = res.log_predictive.full()
    plot_probabilities(
        axes[2],
        log_predictive,
        trim_zero=False,
        label="Predictive probability",
    )
    axes[2].plot(
        log_predictive.argmax(axis=1),
        color="orange",
        linewidth=0.8,
        alpha=0.7,
    )

    axes[1].plot(posterior.argmax(axis=0), color="green")
    plt.tight_layout()


def load_well_data() -> np.ndarray:
    data = np.loadtxt(DATA_DIR / "well-data.txt", comments="#")
    assert data.ndim == 1
    return data


def well_data_multivarate_bcdm(
    data: np.ndarray, **kwargs
) -> MultivariateBcdmResults:
    return multivariate_bcdm(
        np.ones_like(data),
        data,
        hazard=0.01,
        mean=1e5,
        scale=1e4,
        **kwargs,
    )


def well_data() -> None:
    data = load_well_data()
    results = well_data_multivarate_bcdm(data, max_run_length=1000)

    axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)[1]
    axes[0].plot(data)
    axes[0].set_ylabel("Nuclear magnetic response")
    axes[-1].set_xlabel("Time")
    plot_probabilities(
        axes[1],
        results.log_posterior(),
        label="Posterior probability",
        norm=LogNorm(vmin=1e-12, vmax=1),
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

    for segment in batched(chain((0,), changepoints, (len(data) - 1,)), 2):
        if len(segment) < 2:
            break
        for ax in axes:
            ax.axvspan(
                *segment,
                facecolor="yellow",
                edgecolor="none",
                alpha=0.3,
            )


def well_data_params_compare() -> None:
    data = load_well_data()

    results = [
        (
            name := ", ".join(f"{k}={v}" for k, v in kw.items()),
            well_data_multivarate_bcdm(data, **kw, progress_desc=name),
        )
        for kw in (
            dict(max_num_probs=None),
            dict(max_num_probs=20),
            dict(max_run_length=1000),
            dict(max_run_length=1000, max_num_probs=20),
            dict(max_run_length=500),
        )
    ]

    axes: Sequence[Axes]
    axes = plt.subplots(1 + len(results), 1, figsize=FIGSIZE, sharex=True)[1]
    axes[0].plot(data)
    axes[0].set_ylabel("Nuclear magnetic response")
    axes[-1].set_xlabel("Time")

    images = [
        plot_probabilities(
            ax,
            result.log_posterior(),
            label="Posterior probability",
            norm=LogNorm(vmin=1e-20, vmax=1),
            add_colorbar=False,
        )[1]
        for ax, (_, result) in zip(axes[1:], results, strict=True)
    ]
    plt.colorbar(
        images[0],
        ax=axes,
        label="Run length posterior",
        orientation="vertical",
        fraction=0.05,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, color, (name, result) in zip(
        axes[1:], cycle(colors), results, strict=False
    ):
        changepoints = result.changepoints()
        for _ax in (ax, axes[0]):
            axvlines(
                _ax,
                changepoints,
                label=f"Changepoints ({name})",
                color=color,
                linewidth=1,
            )
        ax.legend(loc="upper left")


def predict_segment(
    prior: NigPrior,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    posteriors = prior.fit_regression(x, y)
    var = posteriors.mvt_variance(x)
    var[var < 0] = np.nan
    return posteriors.mvt_mean(x), var


def atleast_2d_col(x: np.ndarray) -> np.ndarray:
    return np.expand_dims(x, 1) if x.ndim < 2 else x


def plot_segment_predictions(
    ax,
    prior: NigPrior,
    changepoints: list[int],
    get_regressor: Callable[[int, int], NDArray[np.float64]],
    y: np.ndarray,
    t: np.ndarray | None = None,
) -> None:
    n = len(y)
    if t is None:
        t = np.arange(n)

    for i1, i2 in pairwise(chain((0,), changepoints, (n,))):
        pmean, pvar = predict_segment(prior, get_regressor(i1, i2), y[i1:i2])
        tseg = t[i1:i2]
        sd = np.sqrt(pvar)

        ax.plot(tseg, pmean, color="green")
        for delta in (sd, -sd):
            ax.plot(tseg, pmean + delta, color="green", linestyle="--")


EXAMPLES: dict[str, Callable[[], Any]] = {
    "random": random_piecewise,
    "triangular": partial(triangular, reset_time=False),
    "triangular-reset-time": partial(triangular, reset_time=True),
    "well": well_data,
    "well-params-compare": well_data_params_compare,
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
@click.option(
    "--show",
    is_flag=True,
    help="Show plot; default if --output not specified.",
)
@click.option(
    "--no-show",
    is_flag=True,
    help="Do not show plot; default if --output specified.",
)
@click.option(
    "--output-dir",
    type=click.Path(
        dir_okay=True,
        file_okay=False,
        writable=True,
        path_type=Path,
    ),
    help="""Write all example images to directory with name [dataset].png.""",
)
def main(
    examples,
    output: Path | None,
    show: bool,
    no_show: bool,
    output_dir: Path | None,
) -> None:
    if no_show:
        if show:
            raise click.UsageError("Cannot specify both --show and --no-show.")
        show = False
    elif not show:
        show = output is None

    if output:
        if len(examples) != 1:
            raise click.UsageError(
                "--output can only be specified with a single example"
            )

    if output_dir:
        if output:
            raise click.UsageError("--output cannot be used with --output-dir")

        if not output_dir.exists():
            msg = f"directory does not exist: {output_dir}"
            raise click.ClickException(msg)

    if not examples:
        examples = EXAMPLES.keys()

    for example in examples:
        click.echo(f"Running '{example}' example")
        EXAMPLES[example]()
        if output_dir:
            path = output_dir / f"{example}.png"
            plt.savefig(path)
            plt.close()
            click.echo(f"Saved image to {path}", err=True)

    if output:
        plt.savefig(output, format=None if output.suffix else "png")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
