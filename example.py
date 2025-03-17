from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bayesian_change_detection import MultivariateBcdm
from examples import triangle_wave


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
    posterior = np.exp(log_posterior)
    zero_row = np.where(np.isclose(posterior, 0).all(axis=1))[0][0]
    posterior = posterior[:zero_row]
    im = axes[1].imshow(
        posterior,
        aspect="auto",
        origin="lower",
        cmap="gray_r",
        extent=(X[0], X[-1], 0, len(posterior)),
    )
    axes[2].plot(X, posterior.argmax(axis=0))
    for ax in axes[1:]:
        ax.set_ylabel("Run length")

    plt.colorbar(im, ax=axes)


def main() -> None:
    triangular(np.random.default_rng(100))
    plt.show()


if __name__ == "__main__":
    main()
