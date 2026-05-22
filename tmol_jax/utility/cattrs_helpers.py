"""YAML -> frozen attrs dataclass structuring, cattrs-free.

Upstream tmol parses YAML parameter files into frozen ``attrs``/``cattr``
dataclasses. ``cattrs`` is heavy and we want to minimize dependencies in the
JAX port, so this module provides a small recursive structurer that handles
the subset of type annotations actually used by tmol's databases:

    - attrs-defined classes (nested)
    - ``Tuple[T, ...]`` (variadic)
    - ``Tuple[T1, T2, ...]`` (fixed-length)
    - ``Optional[T]``
    - ``List[T]`` (returned as a tuple for frozen-friendliness)
    - ``Dict[K, V]``
    - ``Union[...]`` (tries each branch in order)
    - primitives (``int``, ``float``, ``str``, ``bool``) — passthrough
    - type aliases like ``BondAngle = NewType("BondAngle", float)`` whose
      supertype is a primitive — coerced with ``float()``/``int()``/``str()``.

Parsers that need richer behavior (e.g. pint-backed angle parsing) should
register a hook via :func:`register_hook`.
"""

from __future__ import annotations

import typing
from typing import Any, Callable, Dict, Type

import attrs
import yaml


# ---------------------------------------------------------------------------
# User-registerable structure hooks for NewType-like aliases.
# ---------------------------------------------------------------------------

_HOOKS: Dict[Any, Callable[[Any], Any]] = {}


def register_hook(tp: Any, fn: Callable[[Any], Any]) -> None:
    """Register a custom structuring function for type ``tp``.

    Useful for ``NewType`` aliases (e.g. ``BondAngle``) that should run
    additional parsing such as pint-based unit conversion.
    """
    _HOOKS[tp] = fn


# ---------------------------------------------------------------------------
# Core structuring.
# ---------------------------------------------------------------------------


def _is_attrs_class(tp: Any) -> bool:
    return isinstance(tp, type) and hasattr(tp, "__attrs_attrs__")


def _structure_attrs(value: Any, cls: Type) -> Any:
    if not isinstance(value, dict):
        raise TypeError(
            f"Expected mapping for attrs class {cls.__name__}, got {type(value).__name__}"
        )
    field_types = {f.name: f.type for f in cls.__attrs_attrs__}
    kw = {}
    for name, raw in value.items():
        if name not in field_types:
            # tolerate stray YAML keys (upstream uses cattrs with forbid_extra=False)
            continue
        kw[name] = structure(raw, field_types[name])
    return cls(**kw)


def structure(value: Any, tp: Any) -> Any:
    """Recursively convert ``value`` (a YAML-parsed native Python object)
    into the shape described by type annotation ``tp``."""

    # explicit user hook takes precedence
    if tp in _HOOKS:
        return _HOOKS[tp](value)

    # untyped / Any
    if tp is Any or tp is None or tp is type(None):
        return value

    # attrs dataclass
    if _is_attrs_class(tp):
        return _structure_attrs(value, tp)

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    # NewType: unwrap to supertype, then coerce if primitive
    if hasattr(tp, "__supertype__"):
        return structure(value, tp.__supertype__)

    # Optional[X] / Union[X, Y, ...]
    if origin is typing.Union:
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        # try each branch; return the first one that succeeds
        last_exc = None
        for branch in non_none:
            try:
                return structure(value, branch)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        return value

    # Tuple[...]
    if origin in (tuple,):
        if len(args) == 2 and args[1] is Ellipsis:
            inner = args[0]
            return tuple(structure(v, inner) for v in value)
        return tuple(structure(v, a) for v, a in zip(value, args))

    # List[X] -> tuple (frozen-friendly)
    if origin in (list,):
        (inner,) = args if args else (Any,)
        return tuple(structure(v, inner) for v in value)

    # Dict[K, V]
    if origin in (dict,):
        if not args:
            return dict(value)
        k_tp, v_tp = args
        return {structure(k, k_tp): structure(v, v_tp) for k, v in value.items()}

    # Plain classes (primitives): coerce if needed but don't clobber subclasses
    if isinstance(tp, type):
        if isinstance(value, tp):
            return value
        # safe coercion for primitives
        if tp in (int, float, str, bool):
            return tp(value)
        # last resort: try calling the constructor
        return tp(value)

    # Unknown annotation — return as-is
    return value


# ---------------------------------------------------------------------------
# YAML entry point.
# ---------------------------------------------------------------------------


def structure_from_yaml(yaml_path: str, cls: Type) -> Any:
    """Load ``yaml_path`` with ``yaml.safe_load`` and structure the result
    into an instance of ``cls``."""
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh)
    return structure(raw, cls)


def structure_from_string(yaml_text: str, cls: Type) -> Any:
    """Structure a YAML-formatted string into an instance of ``cls``."""
    raw = yaml.safe_load(yaml_text)
    return structure(raw, cls)


__all__ = [
    "register_hook",
    "structure",
    "structure_from_yaml",
    "structure_from_string",
]
