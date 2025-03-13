import sys
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from tqdm import tqdm

from bayesian_change_detection import (
    MultivariateBcdm,
)
from examples import (
    triangle_wave,
)


def triangular(
    rng: np.random.Generator, *, show_regression: bool = False, **kw
) -> Figure:
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
        hazard=0.001,
        prior_cov=1.0e-3,
        prior_shape=1.0e-6,
    )

    # Update the segment length hypotheses given the data.
    for i, (x, y) in enumerate(tqdm(zip(X, Y, strict=True), total=len(X))):
        model.update(np.array([1, x]), cast(float, y), debug=i == 300)

    fig, axes = plt.subplots(3, sharex=False)
    axes[0].plot(X, Y)
    axes[1].plot(X, model.argmax_log_likelihoods)

    log_likelihoods = np.full((samples + 1, samples + 1), -np.inf)
    for i, _log_likelihood in enumerate(model.log_likelihoods):
        log_likelihoods[: len(_log_likelihood), i] = _log_likelihood

    if np.any(log_likelihoods > 0):
        print("Positive log likelihoods", file=sys.stderr)
    if np.any(np.isnan(log_likelihoods)):
        print("NaNs in log likelihoods", file=sys.stderr)
    im = axes[2].imshow(log_likelihoods, aspect="auto", origin="lower")
    plt.colorbar(im, ax=axes)

    return fig


def main() -> None:
    triangular(np.random.default_rng(100))
    plt.show()


if __name__ == "__main__":
    main()
