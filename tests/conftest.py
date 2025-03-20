from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import numpy as np


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--snapshot-generate",
        action="store_true",
        help="Generate snapshots for all tests.",
    )


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
        test_name: str,
        test_path: Path,
        generating: bool,
    ):
        self.test_name = test_name
        self.test_path = test_path
        self.full_test_name = f"{self.test_path.stem}:{self.test_name}"
        self.snapshot_dir = self.test_path.parent / "__snapshots__"
        self.snapshot_path = self.snapshot_dir / f"{self.full_test_name}.txt"

        self._generating = generating
        self._data: np.ndarray | None = None

    def __call__(self, data: "np.ndarray") -> "np.ndarray":
        if self._generating:
            self._save(data)
            return data

        if self._data is None:
            self._data = _load_numpy_data(
                self.snapshot_path, self.full_test_name
            )

        return self._data

    def _save(self, data: "np.ndarray") -> None:
        import numpy as np

        data = np.atleast_1d(data)
        self._data = data
        self.snapshot_dir.mkdir(exist_ok=True, parents=False)
        np.savetxt(self.snapshot_path, data)


@pytest.fixture
def ndarray_snapshot(request: pytest.FixtureRequest) -> NDArraySnapshot:
    return NDArraySnapshot(
        test_name=request.node.name,
        test_path=request.path,
        generating=request.config.getoption("--snapshot-generate"),
    )
