from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from tqdm import tqdm

from bayesian_change_detection import MultivariateBcdm
from examples import triangle_wave


def plot_posterior_probabilities(
    ax: Axes,
    probs: list[np.ndarray],
    xlim=None,
) -> tuple[np.ndarray, AxesImage]:
    samples = len(probs)
    log_posterior = np.empty((samples, samples))
    for i, p in enumerate(probs):
        n = len(p)
        log_posterior[:n, i] = p
        log_posterior[n:, i] = -np.inf

    if xlim:
        x0, x1 = xlim
    else:
        x0, x1 = (0, samples - 1)

    # Plot run length posterior
    posterior = np.exp(log_posterior)
    zero_row = np.where(np.isclose(posterior, 0).all(axis=1))[0][0]
    posterior = posterior[:zero_row]
    return posterior, ax.imshow(
        posterior,
        aspect="auto",
        origin="lower",
        cmap="gray_r",
        extent=(x0, x1, 0, len(posterior)),
    )


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

    model = MultivariateBcdm(
        2,
        hazard=0.3,
        # prior_cov=1.0e-3,
        # prior_shape=1.0e-6,
    )

    # Update the segment length hypotheses given the data.
    for x, y in tqdm(zip(X, Y, strict=True), total=len(X)):
        model.update(np.array([1, x]), cast(float, y))

    log_posterior = np.empty((samples, samples))
    for i, probs in enumerate(model.log_posterior):
        n = len(probs)
        log_posterior[:n, i] = probs
        log_posterior[n:, i] = -np.inf

    axes = plt.subplots(3, sharex=True)[1]
    axes[0].plot(X, Y)

    # Plot run length posterior
    posterior, im = plot_posterior_probabilities(
        axes[1], model.log_posterior, X[[0, -1]]
    )
    axes[2].plot(X, posterior.argmax(axis=0))
    for ax in axes[1:]:
        ax.set_ylabel("Run length")

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
    model = MultivariateBcdm(1, hazard=hazard)
    for y in tqdm(data):
        model.update(1, y)

    axes = plt.subplots(2, 1, sharex=True)[1]
    axes[0].plot(data)
    for cp in cps:
        for ax in axes:
            ax.axvline(cp, color="red", linestyle="--")

    plot_posterior_probabilities(axes[1], model.log_posterior)


def main() -> None:
    # triangular(np.random.default_rng(100))
    random_piecewise(np.random.default_rng(42))
    plt.show()


if __name__ == "__main__":
    main()
