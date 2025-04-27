from collections.abc import Sequence

import click

from time_bcdm import TestInfo, load_test_infos


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
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
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
