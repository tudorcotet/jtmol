"""No-op runtime validation helpers.

Upstream tmol uses ``tmol.types.functional.validate_args`` — a typeguard-style
decorator that validates annotated shapes/dtypes on every call. We deliberately
drop runtime validation in the JAX port (it interferes with ``jax.jit`` and adds
overhead), so this module provides a pass-through decorator with the same name.
Downstream modules can keep their ``@validate_args`` annotations untouched.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def validate_args(func: F) -> F:
    """Return ``func`` unchanged. Kept for API compatibility with upstream tmol."""
    return func


__all__ = ["validate_args"]
