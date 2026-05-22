"""Backport of ``functools.singledispatchmethod`` for tmol_jax.

The upstream :mod:`tmol.extern.singledispatchmethod` vendored a
Python 2.7 / 3.4-3.7 compatible implementation externed from
https://github.com/ikalnytskyi/singledispatchmethod @2cf8335 (PSF
License).

tmol_jax requires Python >=3.8 (JAX itself does), which means
:class:`functools.singledispatchmethod` is always available.  This
module therefore re-exports the stdlib version and keeps
``from tmol_jax.extern.singledispatchmethod import singledispatchmethod``
working exactly like the upstream import.
"""

from __future__ import annotations

from functools import singledispatch, singledispatchmethod, update_wrapper

__all__ = ["singledispatch", "singledispatchmethod", "update_wrapper"]
