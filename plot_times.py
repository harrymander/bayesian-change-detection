from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import click

from time_bcdm import Options, TestInfo, load_test_infos

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
    name = "scale"

    def __init__(self, *args, **kwargs) -> None:
        self._valid_scales = _get_valid_scale_names()
        self._metavar_choices = sorted(self._valid_scales | set(["log[N]"]))

    def get_metavar(self, param: click.Parameter) -> str:
        choices = "|".join(self._metavar_choices)

        if param.required and param.param_type_name == "argument":
            return f"{{{choices}}}"

        return f"[{choices}]"

    def convert(self, value, param, ctx) -> dict:
        if value in self._valid_scales:
            return {"value": value}
        elif value.startswith("log"):
            base_str = value.removeprefix("log")
            try:
                base = float(base_str)
            except ValueError:
                self.fail("Invalid log scale: must be a valid number")

            if base <= 0 or base == 1:
                self.fail("Invalid log scale: cannot be <= 0 or == 1")

            return {"value": "log", "base": base}

        allowed = ", ".join(f"'{val}'" for val in self._metavar_choices)
        self.fail(
            f"Invalid axis scale '{value}'. Allowed values: {allowed}.",
            param,
            ctx,
        )


@click.command(context_settings=dict(show_default=True))
@click.argument(
    "infos_files",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    nargs=-1,
    required=True,
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
    help="Scale for x axis",
)
@click.option(
    "--yscale",
    type=PlotScale(),
    default="linear",
    help="Scale for y axis",
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
@click.option(
    "--fit",
    default=2,
    type=click.IntRange(min=0),
    help="Degree of polynomial line of best fit. Pass 0 to disable.",
)
@click.option(
    "--fit-bias",
    "--fit-intercept",
    is_flag=True,
    default=False,
    help="Whether to compute intercept term in line of best fit.",
)
@click.option(
    "--ungroup",
    "-u",
    multiple=True,
    help="Do not group times by this parameter in the plot. Can be repeated.",
    type=click.Choice(list(Options.model_fields)),
)
@click.option(
    "--group-refs/--ungroup-refs",
    default=False,
    help="Whether to group by Git ref in plot (disabled by default).",
)
def main(
    infos_files: list[str],
    ungroup: list[str],
    **kwargs,
) -> None:
    """
    Plots times in INFOS_FILES.
    """
    infos: list[TestInfo] = []
    for file in infos_files:
        infos.extend(load_test_infos(file))
    plot_times(infos, ungroup=set(ungroup), **kwargs)


@dataclass(frozen=True)
class OptionGrouping:
    names: tuple[str, ...]
    values: tuple[Any, ...]

    def __str__(self) -> str:
        name_vals = zip(self.names, self.values, strict=True)
        return ", ".join(f"{k}={v}" for k, v in name_vals)

    @classmethod
    def from_items(
        cls, items: Mapping[str, Any], ungroup: set[str]
    ) -> "OptionGrouping":
        names = []
        values = []
        for k, v in items.items():
            if k not in ungroup:
                names.append(k)
                values.append(v)

        return cls(names=tuple(names), values=tuple(values))


def group_times_by_options(
    infos: list[TestInfo],
    min_samples: float | None,
    max_samples: float | None,
    ungroup: set[str],
    group_refs: bool,
) -> dict[OptionGrouping, tuple[Sequence[int], Sequence[float]]]:
    def _num_samples_in_range(info: TestInfo) -> bool:
        num_samples = info.num_samples
        if num_samples < (min_samples or 0):
            return False
        return max_samples is None or num_samples <= max_samples

    grouped_times: dict[OptionGrouping, list[tuple[int, float]]] = {}
    for info in filter(_num_samples_in_range, infos):
        items = info.options.model_dump()
        if group_refs:
            items["ref"] = info.ref
        group = OptionGrouping.from_items(items, ungroup)
        grouped_times.setdefault(group, []).append(
            (info.num_samples, info.avg_execution_time)
        )

    for times in grouped_times.values():
        times.sort(key=lambda t: t[0])

    return {
        options: tuple(zip(*times, strict=True))  # type: ignore
        for options, times in grouped_times.items()
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
    fit: int,
    fit_bias: bool,
    ungroup: set[str],
    group_refs: bool,
):
    import matplotlib.pyplot as plt

    subplots_kw: dict[str, Any] = {}
    if figsize:
        subplots_kw["figsize"] = figsize
    fig, ax = plt.subplots(**subplots_kw)

    best_fit_kw = dict(linestyle="--", linewidth=1)
    grouped_times = group_times_by_options(
        infos, xmin, xmax, ungroup, group_refs
    )
    for options, data in grouped_times.items():
        line = ax.plot(
            *data,
            "o",
            markerfacecolor="none",
            label=f"{options} [n={len(data[0])}]",
        )
        if fit:
            ax.plot(
                *poly_best_fit(*data, degree=fit, bias=fit_bias),
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
