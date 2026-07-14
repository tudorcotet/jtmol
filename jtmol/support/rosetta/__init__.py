"""Optional PyRosetta initialization helpers.

PyRosetta is *not* a hard dependency of :mod:`jtmol`; these helpers
are only meaningful when a user wants to compare against Rosetta energy
terms in a notebook / test harness. Importing this subpackage does not
import :mod:`pyrosetta` — only :mod:`jtmol.support.rosetta.init` does.
"""
