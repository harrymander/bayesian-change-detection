from collections.abc import Sequence
from typing import Any

import click

from time_bcdm import TestInfo, load_test_infos


class FigSizeParam(click.ParamType):
    name = "width,height"

    def convert(self, value, param, ctx):
        size_strs = [s for s in value.split(",") if s.strip()]
        if len(size_strs) != 2:
            self.fail("Must be two comma-separated numbers", param, ctx)

        float_arg = click.FloatRange(min=0, min_open=True)
        return [float_arg.convert(s, param, ctx) for s in size_strs]


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
@click.option("--logx", is_flag=True, help="Use logarithmic scale for x axis.")
@click.option("--logy", is_flag=True, help="Use logarithmic scale for y axis.")
@click.option("--grid/--no-grid", default=True, help="Show grid in plot.")
@click.option(
    "--size",
    "figsize",
    type=FigSizeParam(),
    help="Width and height of figure in inches.",
)
def main(infos_file: str, **kwargs) -> None:
    """
    Plots times in INFOS_FILE, which defaults to complexity.json if not
    provided.
    """
    plot_times(load_test_infos(infos_file), **kwargs)


def get_times(
    infos: list[TestInfo],
    trim: bool,
) -> tuple[Sequence[int], Sequence[float]]:
    times = sorted(
        (
            (
                info.options["num_samples"],
                info.execution_time / info.number_loops,
            )
            for info in infos
            if info.options["trim"] == trim
        ),
        key=lambda t: t[0],
    )
    num_samples, execution_times = zip(*times, strict=True)
    return num_samples, execution_times


def plot_times(
    infos: list[TestInfo],
    output: str | None,
    logx: bool,
    logy: bool,
    grid: bool,
    figsize: tuple[float, float] | None,
):
    import matplotlib.pyplot as plt

    subplots_kw: dict[str, Any] = {}
    if figsize:
        subplots_kw["figsize"] = figsize
    fig, ax = plt.subplots(**subplots_kw)

    ax.plot(
        *get_times(infos, trim=True),
        "^",
        markerfacecolor="none",
        label="With support trimming",
    )
    ax.plot(
        *get_times(infos, trim=False),
        "o",
        markerfacecolor="none",
        label="Without support trimming",
    )

    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
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
