import dataclasses
import datetime
import importlib
import json
import subprocess
import sys
import tempfile
import timeit
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import click
import numpy as np

THIS_DIR = Path(__file__).parent


def slog(*args, **kwargs):
    kwargs.setdefault("err", True)
    click.secho(*args, **kwargs)


def subprocess_run(*args) -> None:
    r = subprocess.run(
        args,
        cwd=THIS_DIR,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    )
    if r.returncode:
        click.echo(r.stdout.decode(), err=True)
        raise RuntimeError(
            f"Command '{args}' returned non-zero exit status {r.returncode}"
        )


@contextmanager
def git_worktree(ref: str) -> Generator[Path, None, None]:
    tmp_prefix = f"{Path(__file__).stem}."
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir:
        try:
            path = Path(tmpdir) / "worktree"
            subprocess_run("git", "worktree", "add", str(path), ref)
            yield path
        finally:
            subprocess_run("git", "worktree", "remove", "--force", path)


def get_commit_message_summary(ref: str) -> str:
    return (
        subprocess.run(
            ("git", "log", "-n", "1", "--format=%s", ref),
            cwd=THIS_DIR,
            check=True,
            stdout=subprocess.PIPE,
        )
        .stdout.strip()
        .decode()
    )


def get_sha_for_ref(ref: str) -> str:
    r = subprocess.run(
        ("git", "rev-parse", ref, "--"),
        cwd=THIS_DIR,
        check=True,
        stdout=subprocess.PIPE,
    )
    return r.stdout.decode().splitlines()[0]


@dataclasses.dataclass
class TestInfo:
    ref: str
    datetime: str
    execution_time: float
    number_loops: int
    options: dict[str, Any]


def current_datetime_str() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S %Z")


def load_test_infos(path: Path) -> list[TestInfo]:
    with path.open() as f:
        data = json.load(f)
    return [TestInfo(**info) for info in data]


def write_test_infos(path: Path, infos: list[TestInfo]) -> None:
    data = [dataclasses.asdict(info) for info in infos]
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


@contextmanager
def load_module(path: Path, name: str):
    path_str = str(path)
    module = None
    sys.path.insert(0, path_str)
    try:
        module = importlib.import_module(name)
        import_path = str(path / name)
        if module.__path__[0] != import_path:
            raise AssertionError(
                f"Got invalid path: {module.__path__}, expected {import_path}"
            )
        yield module
    finally:
        if module:
            sys.modules.pop(name)
        sys.path.pop(sys.path.index(path_str))


@click.command(context_settings=dict(show_default=True))
@click.argument("refs", nargs=-1, required=True)
@click.option("--trim/--no-trim", default=False)
@click.option("--num-samples", "-n", default=2000, type=click.IntRange(min=1))
@click.option("--num-loops", default=100, type=click.IntRange(min=1))
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
def main(refs: list[str], output: Path, num_loops: int, **options):
    infos = load_test_infos(output) if output.exists() else []
    for ref in refs:
        ref_sha = get_sha_for_ref(ref)
        datetime_str = current_datetime_str()
        slog(f"Profiling {ref} ({get_commit_message_summary(ref_sha)})...")
        with (
            git_worktree(ref_sha) as worktree,
            load_module(worktree, "bayesian_change_detection") as bcd,
        ):
            execution_time = profile(
                bcd=bcd,
                num_loops=num_loops,
                **options,
            )

        infos.append(
            TestInfo(
                ref=ref_sha,
                datetime=datetime_str,
                execution_time=execution_time,
                number_loops=num_loops,
                options=options,
            )
        )
        write_test_infos(output, infos)
        slog(f"Time for {num_loops} loops: {execution_time:g} s")


def profile(
    *,
    bcd,
    num_loops: int,
    trim: bool,
    num_samples: int,
) -> float:
    rng = np.random.default_rng(42)
    hazard = 1 / 100
    meanx = 0.0
    data = np.empty(num_samples)
    for i in range(num_samples):
        if rng.random() < hazard:
            meanx = rng.normal(scale=2)
        data[i] = rng.normal(meanx)

    kwargs: dict = dict(hazard=hazard)
    if trim:
        kwargs |= dict(
            max_num_probs=50,
            min_prob=1e-6,
        )

    return timeit.timeit(
        "f(x, y, **kwargs)",
        globals=dict(
            f=bcd.multivariate_bcdm,
            x=data,
            y=np.ones_like(data),
            kwargs=kwargs,
        ),
        number=num_loops,
    )


if __name__ == "__main__":
    main()
