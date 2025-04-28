import datetime
import importlib
import re
import subprocess
import sys
import tempfile
import textwrap
import timeit
from collections.abc import Generator, Iterable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, cast

import click
import numpy as np
from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
)
from pydantic_core import from_json

THIS_DIR = Path(__file__).parent


def slog(*args, **kwargs):
    kwargs.setdefault("err", True)
    click.secho(*args, **kwargs)


def run_git(
    args: Iterable[PathLike | str], **kwargs
) -> subprocess.CompletedProcess:
    kwargs.setdefault("cwd", THIS_DIR)
    return subprocess.run(["git", *args], **kwargs)


@contextmanager
def git_worktree(ref: str) -> Generator[Path, None, None]:
    def _run_git(*args) -> None:
        r = run_git(
            args,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        )
        if r.returncode:
            click.echo(r.stdout.decode(), err=True)
            args = ("git", *args)
            msg = f"Command '{args}' returned exit status {r.returncode}"
            raise RuntimeError(msg)

    tmp_prefix = f"{Path(__file__).stem}."
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir:
        try:
            path = Path(tmpdir) / "worktree"
            _run_git("worktree", "add", str(path), ref)
            yield path
        finally:
            _run_git("worktree", "remove", "--force", path)


def get_commit_message_summary(ref: str) -> str:
    return (
        run_git(
            ("log", "-n", "1", "--format=%s", ref),
            check=True,
            stdout=subprocess.PIPE,
        )
        .stdout.strip()
        .decode()
    )


def get_sha_for_ref(ref: str) -> str:
    r = run_git(
        ("rev-parse", ref, "--"),
        check=True,
        stdout=subprocess.PIPE,
    )
    return r.stdout.decode().splitlines()[0]


def git_worktree_is_dirty() -> bool:
    r = run_git(
        ("status", "--porcelain"),
        check=True,
        stdout=subprocess.PIPE,
    )
    return bool(r.stdout.strip())


class TrimOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_num_probs: int | None = Field(..., ge=1)
    min_prob: float = Field(..., ge=0)


class Options(BaseModel):
    trim: TrimOptions | None = None


def _validate_git_sha(value: str) -> str:
    if re.match(r"^[0-9a-f]{40}(?:-dirty)?$", value):
        return value
    raise ValueError("Invalid Git SHA-1 hash")


class TestInfo(BaseModel):
    ref: Annotated[str, AfterValidator(_validate_git_sha)]
    datetime: AwareDatetime
    num_samples: int = Field(..., ge=1)
    execution_time: float = Field(..., gt=0)
    number_loops: int = Field(..., ge=1)
    options: Options

    @property
    def avg_execution_time(self) -> float:
        return self.execution_time / self.number_loops


def current_datetime() -> AwareDatetime:
    return datetime.datetime.now(datetime.UTC)


def load_test_infos(path: PathLike | str) -> list[TestInfo]:
    with open(path, "rb") as f:
        data = f.read()
    return TypeAdapter(list[TestInfo]).validate_json(data)


def write_test_infos(path: Path, infos: list[TestInfo]) -> None:
    data = TypeAdapter(list[TestInfo]).dump_json(infos, indent=2)
    with path.open("wb") as f:
        f.write(data)
        f.write(b"\n")


def load_module(path: Path, name: str):
    sys.path.insert(0, str(path))
    module = importlib.import_module(name)
    import_path = str(path / name)
    if module.__path__[0] != import_path:
        raise AssertionError(
            f"Got invalid path: {module.__path__}, expected {import_path}"
        )
    return module


def load_options(options_json_or_path: str) -> Options:
    arg_is_path = options_json_or_path.lstrip()[0] != "{"
    if arg_is_path:
        with click.open_file(options_json_or_path) as f:
            options_json_or_path = f.read()

    try:
        json_data = from_json(options_json_or_path)
    except ValueError as e:
        msg = f"Failed to parse JSON: {e}"
        if not arg_is_path:
            msg += (
                "\nIf you meant to pass a path to a file, "
                f"prepend './' to the path, e.g. './{options_json_or_path}'"
            )
        raise click.ClickException(msg) from None

    try:
        return Options.model_validate(json_data)
    except ValidationError as e:
        raise click.ClickException(f"Invalid options: {e}") from None


class JsonOrPath(click.ParamType):
    name = "json|path"

    def convert(self, value: str, param, ctx) -> Any:
        arg_is_path = value.lstrip()[0] != "{"
        if arg_is_path:
            path = click.Path(
                readable=True,
                dir_okay=False,
                exists=True,
                allow_dash=True,
                path_type=str,
            ).convert(value, param, ctx)

            # Safe to cast as path_type=str above guarantees path will be a str
            with click.open_file(cast(str, path)) as f:
                value = f.read()

        try:
            return from_json(value)
        except ValueError as e:
            msg = f"Failed to parse JSON: {e}"
            if not arg_is_path:
                msg += (
                    "\nIf you meant to pass a path to a file, "
                    f"prepend './' to the path, e.g. './{value}'"
                )
            self.fail(msg, param, ctx)


@click.command(context_settings=dict(show_default=True))
@click.argument("ref", required=False)
@click.option(
    "--options",
    "options_json",
    help="""Options JSON or path to JSON file (or - to read from stdin).""",
    type=JsonOrPath(),
)
@click.option(
    "--num-samples",
    "-n",
    type=click.IntRange(min=1),
    required=True,
    help="Size of test data to perform change detection on.",
)
@click.option("--num-loops", default=100, type=click.IntRange(min=1))
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
def main(
    ref: str | None,
    output: Path | None,
    num_loops: int,
    num_samples: int,
    options_json: object | None,
):
    if options_json is None:
        options = Options()
    else:
        try:
            options = Options.model_validate(options_json)
        except ValidationError as e:
            raise click.ClickException(f"Invalid options JSON: {e}") from None

    infos = load_test_infos(output) if output and output.exists() else []
    worktree: AbstractContextManager[Path]
    if ref:
        ref_sha = get_sha_for_ref(ref)
        slog(f"Profiling {ref} ({get_commit_message_summary(ref_sha)})...")
        worktree = git_worktree(ref_sha)
    else:
        worktree = nullcontext(THIS_DIR)
        ref_sha = get_sha_for_ref("HEAD")
        dirty = git_worktree_is_dirty()
        slog("Profiling current worktree")
        if dirty:
            ref_sha = f"{ref_sha}-dirty"
            slog("Warning: worktree is dirty", fg="yellow")

    profile_datetime = current_datetime()
    with worktree as worktree_path:
        bcd = load_module(worktree_path, "bayesian_change_detection")
        execution_time = profile(
            bcd=bcd,
            num_loops=num_loops,
            num_samples=num_samples,
            options=options,
        )

    info = TestInfo(
        ref=ref_sha,
        datetime=profile_datetime,
        execution_time=execution_time,
        number_loops=num_loops,
        num_samples=num_samples,
        options=options,
    )
    infos.append(info)
    if output:
        write_test_infos(output, infos)
    slog(textwrap.indent(info.model_dump_json(indent=2), "  "))


def profile(
    *,
    bcd,
    num_samples: int,
    num_loops: int,
    options: Options,
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
    if options.trim:
        kwargs.update(**options.trim.model_dump())

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
