"""Top-level ``jtmol.database`` package.

This module aggregates the chemical and scoring databases into a single
:class:`ParameterDatabase` — the unified handle most downstream code
(score functions, packers, relax protocols) consumes.

Upstream reference: :mod:`tmol.database.__init__`.

The design mirrors upstream:

* :class:`ChemicalDatabase` — raw YAML-sourced residue / atom / variant
  definitions (see :mod:`jtmol.database.chemical`).
* :class:`PatchedChemicalDatabase` — the "refined" view used by scoring
  / packing code, produced by ``PatchedChemicalDatabase.from_chem_db``
  (see :mod:`jtmol.chemical.patched_chemdb`).
* :class:`ScoringDatabase` — all scoring-term YAMLs bundled together
  (see :mod:`jtmol.database.scoring`).

``ParameterDatabase`` then pairs the patched chemical db with the scoring
db. The two most common entry points are:

* :meth:`ParameterDatabase.get_default` — memoized load of the default
  YAMLs shipped under ``default/`` next to this file. This is what
  :func:`jtmol.score.beta2016_score_function` uses when no database
  is passed in explicitly.
* :meth:`ParameterDatabase.from_file` — load from an arbitrary directory
  that contains ``chemical/`` and ``scoring/`` sub-directories in the
  same layout as ``default/``.
"""

from __future__ import annotations

import os
from typing import ClassVar, Optional

import attr

from jtmol.database.chemical import ChemicalDatabase
from jtmol.database.scoring import ScoringDatabase

# NOTE: PatchedChemicalDatabase is imported from jtmol.chemical, which
# depends on this package. Import it lazily inside methods to avoid a
# circular import at module-load time.


@attr.s(auto_attribs=True, slots=False)
class ParameterDatabase:
    """Aggregate of chemical + scoring parameter databases.

    Fields
    ------
    scoring:
        A :class:`ScoringDatabase` — every scoring-term YAML bundled
        into one record.
    chemical:
        A :class:`PatchedChemicalDatabase` — the refined, variant-aware
        view of the raw :class:`ChemicalDatabase`.

    Use :meth:`get_default` for the built-in defaults, or
    :meth:`from_file` to load from a custom directory.
    """

    scoring: ScoringDatabase
    chemical: "PatchedChemicalDatabase"  # type: ignore[name-defined]

    # Class-level memo for ``get_default``; we use a name-mangled
    # ``__default`` attribute like upstream so subclasses can override
    # it cleanly if they ever subclass ``ParameterDatabase``.
    _default: ClassVar[Optional["ParameterDatabase"]] = None

    @classmethod
    def get_default(cls) -> "ParameterDatabase":
        """Return the default :class:`ParameterDatabase` (memoized).

        Loads YAMLs from ``jtmol/database/default/`` the first time
        it is called and caches the result on the class. Subsequent
        calls return the same instance — this is important because the
        default chemical database is large (hundreds of residue types)
        and should not be re-parsed on every score-function build.
        """
        if cls._default is None:
            cls._default = cls.from_file(
                os.path.join(os.path.dirname(__file__), "default")
            )
        return cls._default

    @classmethod
    def from_file(cls, path: str) -> "ParameterDatabase":
        """Load a ``ParameterDatabase`` from a directory.

        ``path`` must contain two sub-directories:

        * ``chemical/`` — passed to
          :meth:`ChemicalDatabase.from_file`, then wrapped with
          :meth:`PatchedChemicalDatabase.from_chem_db`.
        * ``scoring/`` — passed to :meth:`ScoringDatabase.from_file`.

        The layout mirrors upstream ``tmol/database/default/``.
        """
        chem_path = os.path.join(path, "chemical")
        scoring_path = os.path.join(path, "scoring")

        from jtmol.chemical.patched_chemdb import PatchedChemicalDatabase

        chemdb = ChemicalDatabase.from_file(chem_path)
        patched_chemdb = PatchedChemicalDatabase.from_chem_db(chemdb)
        scoring = ScoringDatabase.from_file(scoring_path)

        return cls(scoring=scoring, chemical=patched_chemdb)


__all__ = [
    "ParameterDatabase",
    "ChemicalDatabase",
    "ScoringDatabase",
]
