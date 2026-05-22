"""Ensure PyRosetta is initialised with the ``-beta`` score function.

Direct port of :mod:`tmol.support.rosetta.init` — there is nothing
torch- or JAX-specific here; it is purely a PyRosetta helper. Importing
this module triggers a :func:`pyrosetta.distributed.maybe_init` call (or
asserts that a prior ``pyrosetta.init`` was invoked with ``-beta``).

Example::

    import tmol_jax.support.rosetta.init  # noqa: F401
"""

from __future__ import annotations

import pyrosetta  # type: ignore[import-not-found]

if pyrosetta.rosetta.basic.was_init_called():
    assert (
        "-beta" in pyrosetta.rosetta.basic.options.initialize().get_argv()
    ), "pyrosetta initialized with non-'-beta' configuration."
else:
    pyrosetta.distributed.maybe_init(options="-beta")
