import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

import pytest

if TYPE_CHECKING:
    import numpy as np


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


T = TypeVar("T")


class _SnapshotFixture(ABC, Generic[T]):
    suffix: ClassVar[str]

    @staticmethod
    @abstractmethod
    def _load(path: Path) -> T:
        pass

    @staticmethod
    @abstractmethod
    def _save(path: Path, data: T) -> None:
        pass

    def __init__(
        self,
        *,
        nodeid: str,
        test_path: Path,
        generating: bool,
    ):
        # Assume the path does not contain any "::"
        self.nodeid = nodeid
        _, test_name = self.nodeid.split("::", maxsplit=1)
        snapshot_dir = test_path.parent / "snapshots"
        self.snapshot_path = (
            snapshot_dir / f"{test_path.name}::{test_name}.{self.suffix}"
        )
        self._generating = generating
        self._data: T | None = None

    def __call__(self, data: T) -> T:
        if self._generating:
            self._data = data
            self.snapshot_path.parent.mkdir(exist_ok=True, parents=False)
            self._save(self.snapshot_path, data)
            return data

        if self._data is None:
            path = self.snapshot_path
            if not path.exists():
                msg = (
                    f"snapshot file not found for '{self.nodeid}': "
                    "did you run pytest with --snapshot-generate?"
                )
                raise SnapshotError(msg)
            self._data = self._load(path)

        return self._data


class NDArraySnapshot(_SnapshotFixture["np.ndarray"]):
    suffix = "txt"

    @staticmethod
    def _save(path: Path, data: "np.ndarray") -> None:
        import numpy as np

        data = np.atleast_1d(data)
        if data.ndim > 2:
            raise SnapshotError("Only 1D and 2D arrays are supported.")

        np.savetxt(path, data)

    @staticmethod
    def _load(path: Path) -> "np.ndarray":
        import numpy as np

        return np.loadtxt(path)


class JsonSnapshot(_SnapshotFixture[Any]):
    suffix = "json"

    @staticmethod
    def _save(path: Path, data) -> None:
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
