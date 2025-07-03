import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _set_numpy_err():
    old_settings = np.seterr(all="raise")
    try:
        yield
    finally:
        np.seterr(**old_settings)


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--snapshot-generate",
        action="store_true",
        help="Generate snapshots for selected tests.",
    )


def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers",
        "snapshot: marks a test as using a snapshot fixture",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]):
    for item in items:
        if isinstance(item, pytest.Function):
            if any(name.endswith("_snapshot") for name in item.fixturenames):
                item.add_marker("snapshot")


class SnapshotError(RuntimeError):
    pass


type SnapshotEncoder[T] = Callable[[Path, T], None]
type SnapshotDecoder[T] = Callable[[Path], T]


class _SnapshotFixture:
    suffix: ClassVar[str]
    """
    Must be an empty string or a string of 2 or more chars starting with a dot.
    """

    def __init_subclass__(cls) -> None:
        suffix = getattr(cls, "suffix", None)
        if suffix is None:
            raise ValueError("suffix must be defined")
        if suffix and (len(suffix) == 1 or suffix[0] != "."):
            raise ValueError(f"invalid suffix {suffix!r}")

    def __init__(
        self,
        *,
        nodeid: str,
        test_path: Path,
        generating: bool,
    ):
        # Assume the path does not contain any "::"
        self._nodeid = nodeid
        _, test_name = self._nodeid.split("::", maxsplit=1)
        snapshot_dir = test_path.parent / "snapshots"
        self._snapshot_path = (
            snapshot_dir / f"{test_path.name}::{test_name}{self.suffix}"
        )
        self._generating = generating

    def get_snapshot[T](
        self,
        snapshot_data: T,
        encode: SnapshotEncoder[T],
        decode: SnapshotDecoder[T],
    ) -> T:
        if self._generating:
            self._data = snapshot_data
            self._snapshot_path.parent.mkdir(exist_ok=True, parents=False)
            encode(self._snapshot_path, snapshot_data)
            return snapshot_data

        if not self._snapshot_path.exists():
            msg = (
                f"snapshot file not found for '{self._nodeid}': "
                "did you run pytest with --snapshot-generate?"
            )
            raise SnapshotError(msg)

        return decode(self._snapshot_path)


class NDArraySnapshot(_SnapshotFixture):
    suffix = ".txt"

    def __call__(
        self,
        data: np.ndarray,
        *,
        fmt: str | None = None,
    ) -> np.ndarray:
        if data.ndim > 3:
            # I think we could support arbitrary ndim pretty easily...
            msg = "Only 1, 2, or 3 dimensional arrays supported"
            raise SnapshotError(msg)

        shape = data.shape

        def encode(path: Path, data: np.ndarray) -> None:
            assert shape == data.shape
            b, *end_shape = data.shape
            data = data.reshape(b, int(np.prod(end_shape)))
            kwargs = {"fmt": fmt} if fmt else {}
            np.savetxt(path, data, **kwargs)

        def decode(path: Path) -> np.ndarray:
            return np.loadtxt(path).reshape(*shape)

        return self.get_snapshot(data, encode, decode)


class JsonSnapshot(_SnapshotFixture):
    suffix = ".json"

    def __call__(self, data: Any) -> Any:
        return self.get_snapshot(data, self._save, self._load)

    @staticmethod
    def _save(path: Path, data: Any) -> None:
        with path.open("w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    @staticmethod
    def _load(path: Path) -> Any:
        with path.open() as f:
            return json.load(f)


def _snapshot_fixture(name: str, cls: type[_SnapshotFixture]):
    def fixture(request: pytest.FixtureRequest):
        return cls(
            nodeid=request.node.nodeid,
            test_path=request.path,
            generating=request.config.getoption("--snapshot-generate"),
        )

    return pytest.fixture(name=f"{name}_snapshot")(fixture)


ndarray_snapshot = _snapshot_fixture("ndarray", NDArraySnapshot)
json_snapshot = _snapshot_fixture("json", JsonSnapshot)
