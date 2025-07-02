import json
from abc import ABC, abstractmethod
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


class _SnapshotFixture[T](ABC):
    suffix: ClassVar[str]

    @staticmethod
    @abstractmethod
    def _load(path: Path, test_data: T, **kwargs) -> T:
        pass

    @staticmethod
    @abstractmethod
    def _save(path: Path, data: T, **kwargs) -> None:
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

    def get_snapshot(
        self,
        test_data: T,
        *,
        load_kwargs: dict | None = None,
        save_kwargs: dict | None = None,
    ) -> T:
        if self._generating:
            self._data = test_data
            self.snapshot_path.parent.mkdir(exist_ok=True, parents=False)
            self._save(self.snapshot_path, test_data, **(save_kwargs or {}))
            return test_data

        if self._data is None:
            path = self.snapshot_path
            if not path.exists():
                msg = (
                    f"snapshot file not found for '{self.nodeid}': "
                    "did you run pytest with --snapshot-generate?"
                )
                raise SnapshotError(msg)
            self._data = self._load(path, test_data, **(load_kwargs or {}))

        return self._data


class NDArraySnapshot(_SnapshotFixture[np.ndarray]):
    suffix = "txt"

    def __call__(
        self,
        data: np.ndarray,
        *,
        fmt: str | None = None,
    ) -> np.ndarray:
        return self.get_snapshot(data, save_kwargs=dict(fmt=fmt))

    @staticmethod
    def _save(
        path: Path,
        data: np.ndarray,
        *,
        fmt: None | str = None,
        **kwargs,
    ) -> None:
        import numpy as np

        # I think we could support arbitrary ndim...
        data = np.atleast_1d(data)
        if data.ndim > 3:
            raise SnapshotError("Only 1, 2, or 3 dimensional arrays supported")

        b, *shape = data.shape
        data = data.reshape(b, int(np.prod(shape)))
        if fmt:
            kwargs["fmt"] = fmt
        np.savetxt(path, data, **kwargs)

    @staticmethod
    def _load(path: Path, test_data: np.ndarray, **kwargs) -> np.ndarray:
        import numpy as np

        assert test_data.ndim <= 3
        return np.loadtxt(path).reshape(*test_data.shape)


class JsonSnapshot(_SnapshotFixture[Any]):
    suffix = "json"

    def __call__(self, data) -> Any:
        return self.get_snapshot(data)

    @staticmethod
    def _save(path: Path, data, **_) -> None:
        with path.open("w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    @staticmethod
    def _load(path: Path, test_data: Any, **_) -> Any:
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
