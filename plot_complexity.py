from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import click

from time_bcdm import TestInfo, TrimOptions, load_test_infos

if TYPE_CHECKING:
    import numpy as np


class FigSizeParam(click.ParamType):
    name = "width,height"

    def convert(self, value, param, ctx):
        size_strs = [s for s in value.split(",") if s.strip()]
        if len(size_strs) != 2:
            self.fail("Must be two comma-separated numbers", param, ctx)

        float_arg = click.FloatRange(min=0, min_open=True)
        return [float_arg.convert(s, param, ctx) for s in size_strs]


def _get_valid_scale_names() -> set[str]:
    from matplotlib.scale import get_scale_names

    scales = set(get_scale_names())
    scales.discard("function")
    scales.discard("functionlog")
    return scales


class PlotScale(click.ParamType):
    name = "linear|log[N]|..."
    _valid_scales = _get_valid_scale_names()

    def convert(self, value, param, ctx) -> dict:
        if value in self._valid_scales:
            return {"value": value}
        elif value.startswith("log"):
            base_str = value.removeprefix("log")
            try:
                return {"value": "log", "base": float(base_str)}
            except ValueError:
                pass

        allowed = ", ".join(
            f"'{val}'" for val in sorted(self._valid_scales | set(["log[N]"]))
        )
        super().fail(
            f"Invalid axis scale '{value}'. Allowed values: {allowed}.",
            param,
            ctx,
        )


@click.command(context_settings=dict(show_default=True))
@click.argument(
    "infos_file",
    type=click.Path(exists=True, dir_okay=False),
    default="complexity.json",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True, dir_okay=False),
    help="Write plot to file rather than opening in GUI.",
)
@click.option(
    "--xscale",
    type=PlotScale(),
    default="linear",
    help="Scale for x axis; 'linear', 'log', 'log2' etc.",
)
@click.option(
    "--yscale",
    type=PlotScale(),
    default="linear",
    help="Scale for y axis; 'linear', 'log', 'log2' etc.",
)
@click.option("--grid/--no-grid", default=True, help="Show grid in plot.")
@click.option(
    "--size",
    "figsize",
    type=FigSizeParam(),
    help="Width and height of figure in inches.",
)
@click.option(
    "--xmin",
    type=click.IntRange(min=0, min_open=True),
    help="Minimum x-axis value to plot.",
)
@click.option(
    "--xmax",
    type=click.IntRange(min=0, min_open=True),
    help="Maximum x-axis value to plot.",
)
@click.option("--fit/--no-fit", default=True, help="Show lines of best fit.")
def main(infos_file: str, **kwargs) -> None:
    """
    Plots times in INFOS_FILE, which defaults to complexity.json if not
    provided.
    """
    plot_times(load_test_infos(infos_file), **kwargs)


def group_times_by_trim_options(
    infos: list[TestInfo],
    min_samples: float | None,
    max_samples: float | None,
) -> dict[TrimOptions | None, tuple[Sequence[int], Sequence[float]]]:
    def _num_samples_in_range(info: TestInfo) -> bool:
        num_samples = info.num_samples
        if num_samples < (min_samples or 0):
            return False
        return max_samples is None or num_samples <= max_samples

    grouped_times: dict[TrimOptions | None, list[tuple[int, float]]] = {}
    for info in filter(_num_samples_in_range, infos):
        grouped_times.setdefault(info.options.trim, []).append(
            (info.num_samples, info.avg_execution_time)
        )

    for times in grouped_times.values():
        times.sort(key=lambda t: t[0])

    return {
        trim: tuple(zip(*times, strict=True))  # type: ignore
        for trim, times in grouped_times.items()
    }


def poly_best_fit(
    x: "Sequence[float] | np.ndarray",
    y: "Sequence[float] | np.ndarray",
    degree: int,
    *,
    bias: bool = False,
    n: int = 100,
) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    xr = np.linspace(x[0], x[-1], n).reshape(-1, 1)
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)

    # Set fit_intercept=False as it will be added by PolynomialFeatures
    model = LinearRegression(fit_intercept=False)
    poly = PolynomialFeatures(degree, include_bias=bias)
    model.fit(poly.fit_transform(x), y)
    return xr, model.predict(poly.fit_transform(xr))


def plot_times(
    infos: list[TestInfo],
    output: str | None,
    xscale: dict,
    yscale: dict,
    grid: bool,
    figsize: tuple[float, float] | None,
    xmin: float | None,
    xmax: float | None,
    fit: bool,
):
    import matplotlib.pyplot as plt

    subplots_kw: dict[str, Any] = {}
    if figsize:
        subplots_kw["figsize"] = figsize
    fig, ax = plt.subplots(**subplots_kw)

    best_fit_kw = dict(linestyle="--", linewidth=1)

    grouped_times = group_times_by_trim_options(infos, xmin, xmax)
    for trim, data in grouped_times.items():
        line = ax.plot(
            *data,
            "o",
            markerfacecolor="none",
            label=f"Trim({trim})" if trim else "No support trimming",
        )
        if fit:
            ax.plot(
                *poly_best_fit(*data, 2),
                color=line[0].get_color(),
                **best_fit_kw,
            )

    ax.set_yscale(**yscale)
    ax.set_xscale(**xscale)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Avg. execution time (s)")
    ax.grid(grid)
    ax.legend()

    if output:
        fig.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
