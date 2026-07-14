"""Parameter-import and database-rewrite scripts (JAX port).

These modules reproduce the functionality of
``tmol/support/scoring`` using pure NumPy / YAML and no PyTorch
dependency. Each script consumes a local Rosetta checkout and emits
the lookup YAMLs + table artefacts consumed by
:mod:`jtmol.database.scoring` at runtime.

The public entry points match the upstream module paths so that
upstream documentation and example invocations (``python -m
tmol.support.scoring.rewrite_rama_binary``) translate directly to
``python -m jtmol.support.scoring.rewrite_rama_binary``.

* :mod:`.hbond_param_import` — parse Rosetta CSV hbond parameter
  tables and emit the minimal YAML subset shipped with the library.
* :mod:`.rewrite_rama_binary` — convert Rosetta Ramachandran / P_AA_pp
  text tables to a combined energy grid artefact.
* :mod:`.rewrite_omega_bbdep_binary` — convert backbone-dependent
  omega text tables to a lookup-plus-tables artefact.
* :mod:`.rewrite_dunbrack_binary` — convert Dunbrack rotamer library
  ``.gz`` files to the rotamer/semi-rotamer library artefacts.

All four write ``.npz`` archives via :func:`numpy.savez`/``pickle``
rather than ``torch.save``; the resulting files are readable by the
JAX-port database modules via their ``from_file`` classmethods.
"""
