"""Score-flavoured chemical-database wrapper.

Upstream reference: :class:`tmol.score.chemical_database.AtomTypeParamResolver`.

Upstream bundles the chemical database together with a pandas ``Index``
of atom-type names and a pytorch tensor group of per-type flags
(``is_hydrogen``, ``is_polarh``, ...). The JAX port drops the pandas /
torch machinery in favour of plain numpy arrays plus a ``dict[str, int]``
for name-to-index lookups -- same semantics, zero heavyweight imports.

:class:`ChemicalDatabaseScore` extends (by composition) a
:class:`PatchedChemicalDatabase` with the score-specific precomputation
that LJ/LK, hbond, elec, Dunbrack, and cartbonded all share:

* a string-to-int atom-type index table,
* per-atom-type boolean flags as numpy arrays,
* an ``atom_type_index_for(name)`` lookup that returns ``-1`` for
  unknown atom-type names.

The wrapper is intentionally lightweight: it owns small numpy arrays,
does not depend on PyTorch, and is safe to share across energy terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

import attrs
import numpy as np

if TYPE_CHECKING:  # pragma: no cover -- import-cycle avoidance
    from tmol_jax.chemical.patched_chemdb import PatchedChemicalDatabase


# --------------------------------------------------------------------- #
# ChemicalDatabaseScore                                                  #
# --------------------------------------------------------------------- #


@attrs.define(slots=True)
class ChemicalDatabaseScore:
    """Score-specific annotations on top of :class:`PatchedChemicalDatabase`.

    Attributes
    ----------
    chem_db
        The wrapped :class:`PatchedChemicalDatabase`.
    atom_type_index
        ``dict[str, int]`` mapping atom-type name -> index into the
        ``chem_db.atom_types`` tuple.
    atom_type_is_hydrogen
        ``numpy.bool_[n_atom_types]`` per-type "is hydrogen" flag
        (``element == "H"``).
    atom_type_is_heavyatom
        Logical negation of :attr:`atom_type_is_hydrogen`.
    atom_type_is_acceptor
        ``numpy.bool_[n_atom_types]`` per-type acceptor flag.
    atom_type_is_donor
        ``numpy.bool_[n_atom_types]`` per-type donor flag.
    atom_type_is_hydroxyl
        ``numpy.bool_[n_atom_types]`` per-type hydroxyl flag.
    atom_type_is_polarh
        ``numpy.bool_[n_atom_types]`` per-type polar-hydrogen flag.
    acceptor_hybridization
        ``numpy.int32[n_atom_types]`` acceptor hybridization code:
        ``0=none, 1=sp2, 2=sp3, 3=ring``.
    element_index
        ``dict[str, int]`` mapping element symbol -> index into
        ``chem_db.element_types``.
    """

    chem_db: Any  # PatchedChemicalDatabase (typed via TYPE_CHECKING)
    atom_type_index: Mapping[str, int]
    atom_type_is_hydrogen: np.ndarray
    atom_type_is_heavyatom: np.ndarray
    atom_type_is_acceptor: np.ndarray
    atom_type_is_donor: np.ndarray
    atom_type_is_hydroxyl: np.ndarray
    atom_type_is_polarh: np.ndarray
    acceptor_hybridization: np.ndarray
    element_index: Mapping[str, int]

    # ------------------------------------------------------------------ #
    # Constructors                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_chem_db(cls, chem_db: Any) -> "ChemicalDatabaseScore":
        """Build a :class:`ChemicalDatabaseScore` from a chem-db-like
        object.

        ``chem_db`` can be either a :class:`PatchedChemicalDatabase` or
        any duck-typed object with ``atom_types`` and ``element_types``
        tuples; the original is stored on ``self.chem_db`` and is not
        mutated.
        """
        atom_types = tuple(chem_db.atom_types)
        n = len(atom_types)

        atom_type_index: Dict[str, int] = {at.name: i for i, at in enumerate(atom_types)}

        is_hydrogen = np.zeros(n, dtype=np.bool_)
        is_acceptor = np.zeros(n, dtype=np.bool_)
        is_donor = np.zeros(n, dtype=np.bool_)
        is_hydroxyl = np.zeros(n, dtype=np.bool_)
        is_polarh = np.zeros(n, dtype=np.bool_)
        acc_hyb = np.zeros(n, dtype=np.int32)

        # Match upstream's acceptor-hybridization int code:
        # {None: 0, "sp2": 1, "sp3": 2, "ring": 3}.
        hyb_code = {None: 0, "sp2": 1, "sp3": 2, "ring": 3}

        for i, at in enumerate(atom_types):
            is_hydrogen[i] = at.element == "H"
            is_acceptor[i] = bool(at.is_acceptor)
            is_donor[i] = bool(at.is_donor)
            is_hydroxyl[i] = bool(at.is_hydroxyl)
            is_polarh[i] = bool(at.is_polarh)
            acc_hyb[i] = hyb_code.get(at.acceptor_hybridization, 0)

        is_heavyatom = np.logical_not(is_hydrogen)

        element_types = tuple(chem_db.element_types)
        element_index: Dict[str, int] = {
            el.name: i for i, el in enumerate(element_types)
        }

        return cls(
            chem_db=chem_db,
            atom_type_index=atom_type_index,
            atom_type_is_hydrogen=is_hydrogen,
            atom_type_is_heavyatom=is_heavyatom,
            atom_type_is_acceptor=is_acceptor,
            atom_type_is_donor=is_donor,
            atom_type_is_hydroxyl=is_hydroxyl,
            atom_type_is_polarh=is_polarh,
            acceptor_hybridization=acc_hyb,
            element_index=element_index,
        )

    # ------------------------------------------------------------------ #
    # Lookups                                                            #
    # ------------------------------------------------------------------ #
    def atom_type_index_for(self, name: str) -> int:
        """Return the chem-db atom-type index for a name, or ``-1`` if
        unknown.

        Matches the semantics of upstream's
        ``pandas.Index.get_indexer([name])[0]`` -- missing entries are
        represented as ``-1``.
        """
        return int(self.atom_type_index.get(name, -1))

    def atom_type_inds_for(self, names: Tuple[str, ...]) -> np.ndarray:
        """Vectorised :meth:`atom_type_index_for` returning a
        ``numpy.int32`` array of the same length as ``names``."""
        return np.asarray(
            [self.atom_type_index_for(n) for n in names], dtype=np.int32
        )

    def element_index_for(self, name: str) -> int:
        """Return the element index for a symbol, or ``-1`` if unknown."""
        return int(self.element_index.get(name, -1))

    # ------------------------------------------------------------------ #
    # Attribute forwarding                                               #
    # ------------------------------------------------------------------ #
    @property
    def atom_types(self) -> Tuple[Any, ...]:
        """Pass-through to :attr:`chem_db.atom_types`."""
        return self.chem_db.atom_types

    @property
    def element_types(self) -> Tuple[Any, ...]:
        """Pass-through to :attr:`chem_db.element_types`."""
        return self.chem_db.element_types

    @property
    def residues(self) -> Tuple[Any, ...]:
        """Pass-through to :attr:`chem_db.residues`."""
        return self.chem_db.residues

    @property
    def variants(self) -> Tuple[Any, ...]:
        """Pass-through to :attr:`chem_db.variants`."""
        return self.chem_db.variants

    @property
    def n_atom_types(self) -> int:
        return len(self.atom_type_index)


__all__ = ["ChemicalDatabaseScore"]
