from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import numpy.testing


class _NumpyAssertFunction(Protocol):
    def __call__(
        self,
        actual,
        desired,
        *args: Any,
        err_msg: str | None = ...,
        **kwargs: Any,
    ) -> None: ...


def _make_numpy_assert_function(
    assert_func: Callable,
    default_err_msg: str,
    **default_kwargs,
) -> _NumpyAssertFunction:
    def _assert(
        actual, desired, *args, err_msg: str | None = None, **kwargs
    ) -> None:
        __tracebackhide__ = True
        kwargs = default_kwargs | kwargs
        if not err_msg:
            err_msg = default_err_msg
        try:
            assert_func(actual, desired, *args, err_msg=err_msg, **kwargs)
        except AssertionError as e:
            raise AssertionError(f"{err_msg}{e}") from None

    return _assert


assert_allclose = _make_numpy_assert_function(
    numpy.testing.assert_allclose, "Arrays are not close"
)
assert_array_equal_strict = _make_numpy_assert_function(
    numpy.testing.assert_array_equal, "Arrays are not equal", strict=True
)


def assert_array_less_strict(actual, desired, *args, **kwargs) -> None:
    __tracebackhide__ = True

    # strict only supported on numpy v2
    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        default_kwargs = {"strict": True}
    else:
        default_kwargs = {}
        assert actual.shape == desired.shape, (
            f"Shapes are not equal: {actual.shape} != {desired.shape}"
        )
        assert actual.dtype is desired.dtype, (
            f"Dtypes are not equivalent: {actual.dtype} is not {desired.dtype}"
        )

    _make_numpy_assert_function(
        numpy.testing.assert_array_less,
        "Arrays are not strictly ordered `x < y`",
        **default_kwargs,
    )(actual, desired, *args, **kwargs)
