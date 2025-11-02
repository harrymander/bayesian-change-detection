import csv
from typing import IO

import click


@click.command()
@click.argument("json", type=click.File("rb"))
@click.option("--output", "-o", type=click.File("w"), default="-")
def main(json: IO[bytes], output: IO[str]) -> None:
    from pydantic import TypeAdapter

    from time_bcdm import TestInfo

    results = TypeAdapter(list[TestInfo]).validate_json(json.read())

    writer = csv.DictWriter(
        output,
        ["commit", "num_samples", "max_num_probs", "min_prob", "time"],
        lineterminator="\n",
    )
    writer.writeheader()
    for result in results:
        row = {
            "commit": result.commit,
            "num_samples": result.num_samples,
            "max_num_probs": result.options.max_num_probs,
            "min_prob": result.options.min_prob,
        }
        for time in result.execution_times:
            writer.writerow(row | {"time": time})


if __name__ == "__main__":
    main()
