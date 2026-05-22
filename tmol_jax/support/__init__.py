"""External support components for ``tmol_jax``.

This package mirrors :mod:`tmol.support` from the upstream PyTorch code
base. It provides utilities that are *not* runtime dependencies of the
library — single-pass parameter import scripts, database-generation
helpers, and optional PyRosetta glue.

Contents:

* :mod:`.rosetta` — optional PyRosetta initialization helper. Only
  useful if ``pyrosetta`` is installed; ``tmol_jax`` itself has no
  PyRosetta dependency.
* :mod:`.scoring` — parameter import / binary rewrite scripts that
  consume a local Rosetta checkout and emit the YAML / binary tables
  shipped under :mod:`tmol_jax.database.scoring`. These are the JAX
  ports of the upstream ``tmol/support/scoring`` scripts and use
  :mod:`numpy` end-to-end (no ``torch`` required for the data import
  itself; the resulting artefacts are plain ``.npz`` / ``.pkl``).

Nothing in this subpackage is imported from the hot code path, and the
import-time cost is minimised accordingly — the submodules are only
touched when the corresponding script is invoked.
"""
