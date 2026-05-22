"""Compatibility / defensive-import shims for tmol_jax.

Written against the *declared* contract between subpackages. Because all
30 subpackages are developed in parallel, names may drift. Importing
this module must never fail — it surfaces diagnostics via
``check_install()`` / ``print_install_status()`` and provides broken-stub
fallbacks for the three highest-level entrypoints.

Stdlib only. Keep small.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


# ---------------------------------------------------------------------------
def lazy_import(path: str, attr: str | None = None) -> Any:
    """Import ``path`` on demand; optionally look up ``attr`` on it.

    Raises ``ImportError`` (with upstream chained) on any failure.
    """
    try:
        mod = importlib.import_module(path)
    except Exception as exc:  # noqa: BLE001
        raise ImportError(f"Could not import {path!r}: {exc}") from exc
    if attr is None:
        return mod
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise ImportError(f"Module {path!r} has no attribute {attr!r}") from exc


# ---------------------------------------------------------------------------
# Broken-stub fallbacks for high-level entrypoints.
# ---------------------------------------------------------------------------

_DEP_MSG = (
    "pose, pose_stack_construction, canonical_ordering, and chemical"
)


def _make_broken(name: str, upstream: BaseException) -> Callable[..., Any]:
    if name == "pose_stack_from_pdb":
        prefix = "PDB loading"
    else:
        prefix = name
    msg = (
        f"{prefix} requires all of {_DEP_MSG} to be working. "
        f"Error was: {upstream!r}"
    )

    def _broken(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError(msg)

    _broken.__name__ = f"broken_{name}"
    _broken.__qualname__ = _broken.__name__
    _broken.__doc__ = f"Fallback stub for {name}. Upstream: {upstream!r}"
    return _broken


def get_pose_stack_from_pdb() -> Callable[..., Any]:
    try:
        return lazy_import("tmol_jax.io.pdb_parsing", "pose_stack_from_pdb")
    except ImportError as exc:
        return _make_broken("pose_stack_from_pdb", exc)


def get_beta2016_score_function() -> Callable[..., Any]:
    try:
        return lazy_import("tmol_jax.score", "beta2016_score_function")
    except ImportError as exc:
        return _make_broken("beta2016_score_function", exc)


def get_run_cart_min() -> Callable[..., Any]:
    try:
        return lazy_import("tmol_jax.optimization.minimizers", "run_cart_min")
    except ImportError as exc:
        return _make_broken("run_cart_min", exc)


# ---------------------------------------------------------------------------
# Subpackage probes. Mirrors the contract declared in tmol_jax.__init__.
# ---------------------------------------------------------------------------

_PROBES: tuple[tuple[str, str | None], ...] = (
    ("tmol_jax.chemical", None),
    ("tmol_jax.database", "ParameterDatabase"),
    ("tmol_jax.types", None),
    ("tmol_jax.utility", None),
    ("tmol_jax.pose", None),
    ("tmol_jax.pose.pose_stack", "PoseStack"),
    ("tmol_jax.pose.packed_block_types", "PackedBlockTypes"),
    ("tmol_jax.io.pdb_parsing", "pose_stack_from_pdb"),
    ("tmol_jax.io.write_pose_stack_pdb", "write_pose_stack_pdb"),
    ("tmol_jax.score", "beta2016_score_function"),
    ("tmol_jax.score.score_function", "ScoreFunction"),
    ("tmol_jax.score.score_types", "ScoreType"),
    ("tmol_jax.kinematics.fold_forest", "FoldForest"),
    ("tmol_jax.kinematics.move_map", "MoveMap"),
    ("tmol_jax.optimization.minimizers", "run_cart_min"),
    ("tmol_jax.pack", None),
    ("tmol_jax.relax.fast_relax", "fast_relax"),
)


def check_install() -> dict[str, str]:
    """Probe every expected subpackage. ``{name: "ok" | error_str}``."""
    out: dict[str, str] = {}
    for path, attr in _PROBES:
        key = path if attr is None else f"{path}.{attr}"
        try:
            lazy_import(path, attr)
            out[key] = "ok"
        except Exception as exc:  # noqa: BLE001
            out[key] = f"{type(exc).__name__}: {exc}"
    return out


def print_install_status() -> None:
    """Pretty-print ``check_install()`` as a table."""
    status = check_install()
    width = max((len(k) for k in status), default=20)
    total = len(status)
    ok = sum(1 for v in status.values() if v == "ok")
    print(f"tmol_jax install status  ({ok}/{total} ok)")
    print("-" * (width + 10))
    for name in sorted(status):
        mark = "OK  " if status[name] == "ok" else "FAIL"
        print(f"{mark}  {name.ljust(width)}  {status[name]}")


__all__ = [
    "lazy_import",
    "get_pose_stack_from_pdb",
    "get_beta2016_score_function",
    "get_run_cart_min",
    "check_install",
    "print_install_status",
]
