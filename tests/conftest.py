from pathlib import Path
from typing import TYPE_CHECKING

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


def pytest_collection_modifyitems(session, config, items: list[pytest.Item]):
    for item in items:
        if isinstance(item, pytest.Function):
            if "ndarray_snapshot" in item.fixturenames:
                item.add_marker("snapshot")


class SnapshotError(RuntimeError):
    pass


def _load_numpy_data(path: Path, testname: str) -> "np.ndarray":
    import numpy as np

    if not path.exists():
        msg = (
            f"snapshot file not found for '{testname}': did you run pytest "
            "with --snapshot-generate?"
        )
        raise SnapshotError(msg)

    return np.loadtxt(path)


class NDArraySnapshot:
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
        snapshot_dir = test_path.parent / "__snapshots__"
        self.snapshot_path = (
            snapshot_dir / f"{test_path.name}::{test_name}.txt"
        )
        self._generating = generating
        self._data: np.ndarray | None = None

    def __call__(self, data: "np.ndarray") -> "np.ndarray":
        if self._generating:
            self._save(data)
            return data

        if self._data is None:
            self._data = _load_numpy_data(self.snapshot_path, self.nodeid)

        return self._data

    def _save(self, data: "np.ndarray") -> None:
        import numpy as np

        data = np.atleast_1d(data)
        self._data = data
        self.snapshot_path.parent.mkdir(exist_ok=True, parents=False)
        np.savetxt(self.snapshot_path, data)


@pytest.fixture
def ndarray_snapshot(request: pytest.FixtureRequest) -> NDArraySnapshot:
    return NDArraySnapshot(
        nodeid=request.node.nodeid,
        test_path=request.path,
        generating=request.config.getoption("--snapshot-generate"),
    )
