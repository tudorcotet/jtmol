"""External-dependency adapters for tmol_jax.

This package mirrors ``tmol.extern`` from the upstream PyTorch codebase.
In the pure-JAX port the heavy C++ vendored dependencies (``Eigen``,
``cppitertools``, ``moderngpu``) are omitted -- they are only used by
the C++/CUDA extensions that are not part of the JAX port -- while the
pure-Python adapters are ported verbatim or lightly adapted so that
``from tmol_jax.extern import ...`` works exactly like upstream.

Ports included:

* :mod:`tmol_jax.extern.toposort` - pure-Python topological sort,
  identical to upstream.
* :mod:`tmol_jax.extern.singledispatchmethod` - backport of
  :func:`functools.singledispatchmethod` for older Pythons.  On Python
  3.8+ this simply re-exports the stdlib implementation.
* :mod:`tmol_jax.extern.pysmiles` - pure-Python SMILES reader/writer
  (depends only on :mod:`networkx`).  Used by
  :mod:`tmol_jax.chemical.patched_chemdb` for residue graph matching.
* :mod:`tmol_jax.extern.openfold` - constants extracted from OpenFold
  (subset of AlphaFold's ``residue_constants``).  Used by the optional
  ``pose_stack_from_openfold`` I/O bridge.
* :mod:`tmol_jax.extern.rosettafold2` - residue name / atom ordering
  tables from RoseTTAFold2.  Used by the optional
  ``pose_stack_from_rosettafold2`` I/O bridge.
"""

from __future__ import annotations

import os.path


def include_paths():
    """Get ``-I`` compatible include dirs for external modules.

    Present for API parity with ``tmol.extern.include_paths``.  In the
    pure-JAX port there are no C++ extensions that need these headers,
    so the returned list is only meaningful for tooling that mirrors
    the upstream layout.
    """

    return [os.path.abspath(os.path.dirname(__file__))]
