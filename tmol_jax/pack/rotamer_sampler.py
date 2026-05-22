"""Dunbrack-backed rotamer sampler.

Given a residue type, a backbone ``(phi, psi)`` pair and (optionally) a
:class:`tmol_jax.database.scoring.dunbrack_libraries.DunbrackRotamerLibrary`,
enumerate up to ``n_rots`` rotamers as Cartesian-space coordinate arrays.

Exact vs. simplified:

* Exact — looks up the per-AA Dunbrack table keyed by
  ``restype.name3`` / ``restype.base_name``, identifies the grid cell
  closest to ``(phi, psi)`` (10 degree step, no bilinear blending), and
  returns the top-``k`` rotamers ranked by probability at that cell.
  Chi means come straight from the library (degrees), and are written
  into the residue's chi dihedrals via explicit Rodrigues rotations of
  the downstream atoms. This matches upstream's chi-sample construction
  for the rotameric chis.
* Simplified — ignores chi-expansions (upstream adds ``+/-1 sigma``
  extra samples around each mean), treats all residues as "buried"
  (the upstream burial classifier is not ported), and drops the
  non-rotameric chi sweep for semirotameric residues (we just use the
  library's mean of the last chi rather than sampling over the
  continuous density). For the packer these approximations matter only
  at the margins — the top rotamers at any given phi/psi dominate
  packing decisions.

The sampler returns numpy arrays so the coord array can be fed through
``PoseStack.replace(coords=...)`` via :func:`jax.numpy.asarray`.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Geometry helpers (pure numpy)                                                #
# --------------------------------------------------------------------------- #


def _rodrigues_rotate(
    points: np.ndarray, axis_start: np.ndarray, axis_end: np.ndarray, angle: float
) -> np.ndarray:
    """Rotate ``points`` about the line ``axis_start -> axis_end`` by ``angle``
    radians. Returns the rotated coords.

    Implements the Rodrigues rotation formula:

        v_rot = v * cos(theta)
              + (k x v) * sin(theta)
              + k * (k . v) * (1 - cos(theta))

    with ``v`` the point expressed relative to ``axis_start`` and
    ``k`` the unit vector along ``axis_end - axis_start``.
    """
    axis = axis_end - axis_start
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return points.copy()
    k = axis / n
    v = points - axis_start[None, :]
    c = math.cos(angle)
    s = math.sin(angle)
    kxv = np.cross(k[None, :], v)
    kdotv = (v @ k)[:, None]
    rot = v * c + kxv * s + k[None, :] * kdotv * (1.0 - c)
    return rot + axis_start[None, :]


def _dihedral_angle_np(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Signed dihedral angle (radians) defined by the four points a->b->c->d."""
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
    x = float(n1 @ n2)
    y = float(m1 @ n2)
    return math.atan2(y, x)


# --------------------------------------------------------------------------- #
# Chi-atom discovery                                                           #
# --------------------------------------------------------------------------- #


def _chi_atoms_for_restype(restype: Any) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Return ``[(chi_name, (a, b, c, d)), ...]`` for each chi torsion.

    Only keeps intra-residue chis (all four atoms with ``atom_idx >= 0``).
    Order: ``chi1, chi2, ...``.
    """
    out: List[Tuple[str, Tuple[int, int, int, int]]] = []
    t2u = getattr(restype, "torsion_to_uaids", None)
    if t2u is None:
        return out
    chi_names = sorted([k for k in t2u if k.startswith("chi")])
    for name in chi_names:
        uaids = t2u[name]
        # Require all four atoms intra-residue (for the two inner atoms this
        # is automatic; the first atom of chi1 is typically N which is fine).
        ok = True
        atoms: List[int] = []
        for u in uaids:
            if u[0] < 0:
                ok = False
                break
            atoms.append(int(u[0]))
        if ok and len(atoms) == 4:
            out.append((name, tuple(atoms)))  # type: ignore[arg-type]
    return out


def _downstream_atoms_of_chi(
    restype: Any, chi_atoms: Tuple[int, int, int, int]
) -> np.ndarray:
    """Find atoms "downstream" of a chi torsion axis ``(b, c)``.

    Walks the residue's bond graph from atom ``c`` without crossing back
    through ``b``; every atom visited (excluding ``c`` itself — it sits
    on the axis) is part of the "downstream" set that gets rotated when
    we change the chi torsion.

    Parameters
    ----------
    restype
        A :class:`RefinedResidueType` with ``bond_indices`` and
        ``n_atoms``.
    chi_atoms
        ``(a, b, c, d)`` — the four atoms defining the chi dihedral.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``[n_atoms]`` marking downstream atoms.
    """
    n = restype.n_atoms
    a_idx, b_idx, c_idx, d_idx = chi_atoms

    # Build adjacency list from undirected bonds.
    adj: List[List[int]] = [[] for _ in range(n)]
    bi = np.asarray(restype.bond_indices)
    for i in range(bi.shape[0]):
        u, v = int(bi[i, 0]), int(bi[i, 1])
        if 0 <= u < n and 0 <= v < n:
            adj[u].append(v)

    # BFS from c, refusing to cross back through b.
    visited = np.zeros(n, dtype=bool)
    visited[b_idx] = True  # block crossing
    visited[c_idx] = True  # c sits on the axis; don't move it
    frontier = [c_idx]
    downstream = np.zeros(n, dtype=bool)
    while frontier:
        new_frontier: List[int] = []
        for u in frontier:
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    downstream[v] = True
                    new_frontier.append(v)
        frontier = new_frontier
    return downstream


# --------------------------------------------------------------------------- #
# Dunbrack library lookup                                                      #
# --------------------------------------------------------------------------- #


def _find_library_for_restype(restype: Any, dun_db: Any) -> Optional[Any]:
    """Return the ``(RotamericAADunbrackLibrary | SemiRotamericAADunbrackLibrary)``
    that handles ``restype``, or ``None`` if the residue has no entry
    (e.g. ALA, GLY, or a residue whose table is missing).
    """
    if dun_db is None:
        return None
    name3 = getattr(restype, "name3", None) or getattr(restype, "base_name", None)
    if name3 is None:
        return None
    lookup = {m.residue_name: m.dun_table_name for m in (dun_db.dun_lookup or ())}
    table_name = lookup.get(name3)
    if table_name is None:
        return None
    for lib in (dun_db.rotameric_libraries or ()):
        if lib.table_name == table_name:
            return lib
    for lib in (dun_db.semi_rotameric_libraries or ()):
        if lib.table_name == table_name:
            return lib
    return None


def _phi_psi_grid_index(
    phi_deg: float, psi_deg: float, lib_data: Any
) -> Tuple[int, int]:
    """Snap ``(phi_deg, psi_deg)`` to the nearest grid cell of ``lib_data``.

    The library stores a 36x36 grid spanning -180..180 degrees at 10-degree
    resolution. Angle wrapping is applied before indexing.
    """
    start = np.asarray(lib_data.backbone_dihedral_start, dtype=np.float64)
    step = np.asarray(lib_data.backbone_dihedral_step, dtype=np.float64)
    n_phi = lib_data.rotamer_probabilities.shape[1]
    n_psi = lib_data.rotamer_probabilities.shape[2]

    # Wrap to [-180, 180)
    def _wrap(x):
        return ((x + 180.0) % 360.0) - 180.0

    phi_w = _wrap(phi_deg)
    psi_w = _wrap(psi_deg)

    i = int(round((phi_w - float(start[0])) / float(step[0])))
    j = int(round((psi_w - float(start[1])) / float(step[1])))
    i = i % n_phi
    j = j % n_psi
    return i, j


# --------------------------------------------------------------------------- #
# Coordinate-space rotamer construction                                        #
# --------------------------------------------------------------------------- #


def _apply_chi_values(
    coords: np.ndarray,
    chi_atoms_list: Sequence[Tuple[str, Tuple[int, int, int, int]]],
    chi_values_rad: Sequence[float],
    restype: Any,
) -> np.ndarray:
    """Rotate ``coords`` so that each chi dihedral takes the target value.

    Applies chi1, chi2, ... in order. For each chi we:

      1. Measure the current dihedral.
      2. Compute the delta angle to the target.
      3. Rotate all downstream atoms about the ``(b, c)`` axis.

    This matches the in-place rotation semantics of upstream's chi DOF
    pipeline, modulo the fact that we work directly in Cartesian rather
    than via a kinforest.
    """
    new = coords.copy()
    n_chis = min(len(chi_atoms_list), len(chi_values_rad))
    for i in range(n_chis):
        _, atoms = chi_atoms_list[i]
        target = float(chi_values_rad[i])
        a, b, c, d = atoms
        if a >= new.shape[0] or b >= new.shape[0] or c >= new.shape[0] or d >= new.shape[0]:
            continue
        current = _dihedral_angle_np(new[a], new[b], new[c], new[d])
        delta = target - current
        # Normalize delta into (-pi, pi].
        delta = ((delta + math.pi) % (2.0 * math.pi)) - math.pi
        mask = _downstream_atoms_of_chi(restype, atoms)
        if not mask.any():
            continue
        # Rotation convention: rotating about the axis (b->c) with the
        # right-hand rule by +theta decreases the dihedral angle by
        # ``theta`` (verified empirically). So to *add* ``delta`` to the
        # dihedral we rotate by ``-delta``.
        rotated = _rodrigues_rotate(new[mask], new[b], new[c], -delta)
        new[mask] = rotated
    return new


def sample_rotamers(
    restype: Any,
    phi: float = -60.0,
    psi: float = -45.0,
    n_rots: int = 1,
    base_coords: Optional[np.ndarray] = None,
    dun_db: Optional[Any] = None,
    return_probs: bool = False,
) -> Any:
    """Return a list of candidate rotamer coord arrays for ``restype``.

    Parameters
    ----------
    restype
        The :class:`RefinedResidueType` whose side-chain we are
        enumerating.
    phi, psi
        Backbone dihedrals in degrees. Used to look up the Dunbrack
        rotamer table.
    n_rots
        Upper bound on the number of rotamers returned. Smaller if the
        library has fewer, or 1 when no library is available.
    base_coords
        Optional ``[n_atoms, 3]`` numpy array to start from. If omitted
        we use ``restype.ideal_coords[:n_atoms]``.
    dun_db
        Optional :class:`DunbrackRotamerLibrary`. When ``None`` (or
        when the residue has no library entry) we return a single
        identity-conformation rotamer built from the base coords.

    Returns
    -------
    List[np.ndarray] or (List[np.ndarray], List[float])
        When ``return_probs`` is ``False`` (default): a list of
        ``[n_atoms, 3]`` coord arrays. The first entry is always the
        base conformation; subsequent entries are the top-ranked
        Dunbrack rotamers at ``(phi, psi)``.

        When ``return_probs`` is ``True``: a tuple ``(coords_list,
        neglnprob_list)`` — the coord list plus a matching list of
        ``-log P(rotamer | phi, psi)`` values from the Dunbrack
        library. The first entry (base conformation) is assigned
        ``0.0``.
    """
    def _ret(coords, probs):
        if return_probs:
            return coords, probs
        return coords

    # Pull base coordinates — either caller-supplied or the residue's ideal.
    try:
        n_atoms = int(restype.n_atoms)
    except Exception:
        return _ret([], [])
    if base_coords is None:
        ideal = np.asarray(restype.ideal_coords)
        # ``ideal_coords`` may include icoor rows beyond n_atoms; slice down.
        base = np.asarray(ideal[:n_atoms], dtype=np.float64)
    else:
        base = np.asarray(base_coords[:n_atoms], dtype=np.float64)

    # Always include the base as rotamer #0 so the "do nothing" move is
    # available (important for non-canonical / library-less residues).
    out: List[np.ndarray] = [base.copy()]
    neglnp_out: List[float] = [0.0]
    if n_rots <= 1:
        return _ret(out, neglnp_out)

    lib = _find_library_for_restype(restype, dun_db)
    if lib is None:
        # No library: duplicate the base to reach the requested count.
        return _ret(out, neglnp_out)

    chi_atoms = _chi_atoms_for_restype(restype)
    if not chi_atoms:
        return _ret(out, neglnp_out)

    rd = lib.rotameric_data
    i, j = _phi_psi_grid_index(phi, psi, rd)

    # Probabilities at this grid cell: [nrot]. Use prob_sorted_rot_inds
    # when present (it's a per-cell ranking).
    probs = np.asarray(rd.rotamer_probabilities, dtype=np.float64)[:, i, j]
    n_rot_total = probs.shape[0]
    n_keep = min(int(n_rots) - 1, n_rot_total)
    if n_keep <= 0:
        return _ret(out, neglnp_out)

    sort_inds = np.argsort(-probs)[:n_keep]

    # means: shape either [nrot, nphi, npsi, nchi] or [nrot, nchi, nphi, npsi]
    means_raw = np.asarray(rd.rotamer_means, dtype=np.float64)
    # Detect layout: last axis should be nchi (small), or axis 1 is nchi.
    if means_raw.shape[-1] <= 6 and means_raw.ndim == 4:
        # [nrot, nphi, npsi, nchi]
        means_for_cell = means_raw[:, i, j, :]  # [nrot, nchi]
    elif means_raw.ndim == 4 and means_raw.shape[1] <= 6:
        # [nrot, nchi, nphi, npsi]
        means_for_cell = means_raw[:, :, i, j]  # [nrot, nchi]
    else:
        # Unknown layout — abort gracefully.
        return _ret(out, neglnp_out)

    # Upstream "wrap negative means" correction.
    means_for_cell = np.where(means_for_cell < -120.0, means_for_cell + 360.0, means_for_cell)
    means_rad = means_for_cell * (math.pi / 180.0)

    for ri in sort_inds:
        chi_vals = means_rad[int(ri)]  # [nchi]
        rot_coords = _apply_chi_values(base, chi_atoms, chi_vals, restype)
        out.append(rot_coords)
        # -log P, clamped to avoid log(0).
        p = float(max(probs[int(ri)], 1e-9))
        neglnp_out.append(-math.log(p))
        if len(out) >= int(n_rots):
            break

    return _ret(out, neglnp_out)


__all__ = ["sample_rotamers"]
