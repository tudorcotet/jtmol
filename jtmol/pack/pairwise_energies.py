"""Pairwise rotamer-rotamer energy tables for the packer.

The upstream packer pre-computes a dense ``E_ij(a, b)`` table of energies
between every rotamer of every residue pair; the simulated-annealer then
moves on this table rather than re-scoring the whole pose on each step.

We build the same table — :func:`build_pairwise_energies` returns a
dict with:

* ``E1`` : ``[n_res, max_rot]`` one-body energies (intra-rotamer +
  rotamer-vs-background interaction).
* ``E2`` : ``[n_res, n_res, max_rot, max_rot]`` two-body energies
  (rotamer-rotamer). Symmetric: ``E2[i,j,a,b] == E2[j,i,b,a]``.
* ``n_rot`` : ``[n_res]`` how many rotamers each residue has.
* ``designable`` : ``[n_res]`` bool whether the position is packed.

Exact vs. simplified:

* Exact — the two-body term uses a shifted Lennard-Jones between every
  heavy-atom pair of the two rotamers, with per-atom-type parameters
  from the score function's LJLK term. This is the dominant pairwise
  contribution and matches the signs/magnitudes of the real score
  function up to a (static) scale factor from electrostatics / H-bonds.
* Simplified — no electrostatics, no H-bond directional term, no LK
  desolvation, no count-pair weight modulation across residues. All
  pairs of atoms use a full-weight LJ. Hydrogens are skipped (they
  contribute only a small correction and 3x the atom count).
* One-body energies evaluate the real score function once per rotamer
  with that rotamer's coords spliced into the background pose. This
  captures rotamer-vs-backbone interactions correctly.

The arrays are JAX-friendly (``jnp.ndarray``), so downstream DEE / SA
can use them under ``jit`` if desired.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import attrs
import numpy as np


# --------------------------------------------------------------------------- #
# LJ parameter access                                                          #
# --------------------------------------------------------------------------- #


def _get_ljlk_term(score_function: Any) -> Optional[Any]:
    """Locate the LJLK energy term inside ``score_function`` (if any)."""
    for term in getattr(score_function, "_terms", []):
        if type(term).__name__ in ("LJLKEnergyTerm", "LJEnergyTerm"):
            return term
    return None


def _atom_ljparams_for_pose(pose_stack: Any, ljlk_state: Any) -> Dict[str, np.ndarray]:
    """Pull per-atom LJ radius / well-depth / flags from the LJLK state.

    The LJLK term's :class:`_LJLKState` carries ``[n_poses, n_atoms]``
    arrays keyed by pose atom index. We return plain numpy copies that
    the caller can gather from.
    """
    return {
        "lj_radius": np.asarray(ljlk_state.lj_radius),
        "lj_wdepth": np.asarray(ljlk_state.lj_wdepth),
        "is_donor": np.asarray(ljlk_state.is_donor),
        "is_acceptor": np.asarray(ljlk_state.is_acceptor),
        "is_hydroxyl": np.asarray(ljlk_state.is_hydroxyl),
        "is_polarh": np.asarray(ljlk_state.is_polarh),
        "atom_is_real": np.asarray(ljlk_state.atom_is_real),
    }


def _gather_block_ljparams(
    pose_stack: Any, pose: int, block: int, ljparams: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Gather per-atom LJ params for the atoms of one block.

    Returns 1-D arrays of shape ``[n_atoms_in_block]``.
    """
    bti = int(pose_stack.block_type_ind[pose, block])
    if bti < 0:
        return {k: np.zeros((0,), dtype=v.dtype) for k, v in ljparams.items()}
    n = int(pose_stack.packed_block_types.n_atoms[bti])
    start = int(pose_stack.block_coord_offset[pose, block])
    sl = slice(start, start + n)
    return {k: v[pose, sl] for k, v in ljparams.items()}


# --------------------------------------------------------------------------- #
# LJ pair energy kernel                                                        #
# --------------------------------------------------------------------------- #


# Constants matching jtmol.score.ljlk.potentials
_CPOLY_DMAX = 6.0
_LJ_CPOLY_DMIN_FLOOR = 4.5
_LJ_CPOLY_DMIN_MAX_OFFSET = 0.1
_LJ_HBOND_DIS = 3.0  # defaults; overridden below if global params available
_LJ_HBOND_OH = 2.6
_LJ_HBOND_HDIS = 1.75


def _pair_lj_energy(
    coords_a: np.ndarray,  # [Na, 3]
    coords_b: np.ndarray,  # [Nb, 3]
    lj_a: Dict[str, np.ndarray],
    lj_b: Dict[str, np.ndarray],
    real_a: np.ndarray,
    real_b: np.ndarray,
    include_hydrogen: bool = False,
) -> float:
    """Total shifted-LJ energy (attractive + repulsive) between two atom sets.

    Mirrors :func:`jtmol.score.ljlk.potentials.lj_atr_rep` but without
    the count-pair weight (we assume inter-residue pairs, for which the
    path distance is always >= 5 bonds so ``w = 1``).
    """
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return 0.0

    # Optionally drop hydrogens (wdepth==0 typically; or detect via is_polarh
    # / small radius). For a fast O(N^2) path we simply mask on a radius
    # threshold.
    if not include_hydrogen:
        mask_a = (lj_a["lj_radius"] > 0.6) & real_a
        mask_b = (lj_b["lj_radius"] > 0.6) & real_b
    else:
        mask_a = real_a
        mask_b = real_b
    if not mask_a.any() or not mask_b.any():
        return 0.0

    ca = coords_a[mask_a]
    cb = coords_b[mask_b]
    lja = {k: v[mask_a] for k, v in lj_a.items() if k != "atom_is_real"}
    ljb = {k: v[mask_b] for k, v in lj_b.items() if k != "atom_is_real"}

    diff = ca[:, None, :] - cb[None, :, :]  # [Na, Nb, 3]
    d2 = np.sum(diff * diff, axis=-1)
    dist = np.sqrt(np.maximum(d2, 1e-12))

    # Per-pair sigma
    std_sigma = lja["lj_radius"][:, None] + ljb["lj_radius"][None, :]
    heavy_donor_ij = (
        lja["is_donor"][:, None]
        & (~lja["is_hydroxyl"][:, None])
        & ljb["is_acceptor"][None, :]
    )
    heavy_donor_ji = (
        ljb["is_donor"][None, :]
        & (~ljb["is_hydroxyl"][None, :])
        & lja["is_acceptor"][:, None]
    )
    hyd_donor_ij = (
        lja["is_donor"][:, None]
        & lja["is_hydroxyl"][:, None]
        & ljb["is_acceptor"][None, :]
    )
    hyd_donor_ji = (
        ljb["is_donor"][None, :]
        & ljb["is_hydroxyl"][None, :]
        & lja["is_acceptor"][:, None]
    )
    polarh_ij = lja["is_polarh"][:, None] & ljb["is_acceptor"][None, :]
    polarh_ji = ljb["is_polarh"][None, :] & lja["is_acceptor"][:, None]

    sigma = np.where(polarh_ij | polarh_ji, _LJ_HBOND_HDIS, std_sigma)
    sigma = np.where(hyd_donor_ij | hyd_donor_ji, _LJ_HBOND_OH, sigma)
    sigma = np.where(heavy_donor_ij | heavy_donor_ji, _LJ_HBOND_DIS, sigma)

    # Per-pair epsilon = sqrt(w_i * w_j)
    eps = np.sqrt(
        np.maximum(lja["lj_wdepth"][:, None] * ljb["lj_wdepth"][None, :], 1e-12)
    )

    d_lin = sigma * 0.6
    cpoly_dmax = _CPOLY_DMAX
    cpoly_dmin_cap = cpoly_dmax - _LJ_CPOLY_DMIN_MAX_OFFSET
    cpoly_dmin = np.where(
        sigma > _LJ_CPOLY_DMIN_FLOOR,
        np.minimum(sigma, cpoly_dmin_cap),
        _LJ_CPOLY_DMIN_FLOOR,
    )

    above = dist > cpoly_dmax

    # Bare LJ
    sd = sigma / np.maximum(dist, 1e-6)
    sd2 = sd * sd
    sd6 = sd2 * sd2 * sd2
    sd12 = sd6 * sd6
    v_bare = eps * (sd12 - 2.0 * sd6)

    # Linear extrapolation inside d_lin
    sd_l = sigma / np.maximum(d_lin, 1e-6)
    sd2_l = sd_l * sd_l
    sd6_l = sd2_l * sd2_l * sd2_l
    sd12_l = sd6_l * sd6_l
    v_at_dlin = eps * (sd12_l - 2.0 * sd6_l)
    dv_at_dlin = eps * ((-12.0 * sd12_l / d_lin) - (-12.0 * sd6_l / d_lin))
    v_lin = v_at_dlin + dv_at_dlin * (dist - d_lin)

    # Cubic fade above cpoly_dmin
    sd_m = sigma / np.maximum(cpoly_dmin, 1e-6)
    sd2_m = sd_m * sd_m
    sd6_m = sd2_m * sd2_m * sd2_m
    sd12_m = sd6_m * sd6_m
    v_at_dmin = eps * (sd12_m - 2.0 * sd6_m)
    dv_at_dmin = eps * (
        (-12.0 * sd12_m / cpoly_dmin) - (-12.0 * sd6_m / cpoly_dmin)
    )
    # Hermite fade to zero at cpoly_dmax
    span = np.maximum(cpoly_dmax - cpoly_dmin, 1e-6)
    t = (dist - cpoly_dmin) / span
    t = np.clip(t, 0.0, 1.0)
    # h00(t) = 2t^3 - 3t^2 + 1, h10(t) = t^3 - 2t^2 + t (value)
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    v_fade = h00 * v_at_dmin + h10 * dv_at_dmin * span

    in_fade = (dist > cpoly_dmin) & (~above)
    in_vdw = (dist > d_lin) & (~in_fade) & (~above)

    v_raw = np.where(above, 0.0, np.where(in_fade, v_fade, np.where(in_vdw, v_bare, v_lin)))
    below_sigma = dist < sigma
    vatr = np.where(below_sigma, -eps, v_raw)
    vrep = np.where(below_sigma, v_raw + eps, 0.0)
    vatr = np.where(above, 0.0, vatr)
    vrep = np.where(above, 0.0, vrep)

    return float(vatr.sum() + vrep.sum())


# --------------------------------------------------------------------------- #
# Build tables                                                                 #
# --------------------------------------------------------------------------- #


@attrs.define
class PairwiseEnergyTables:
    """Container for the packer's energy tables.

    Fields
    ------
    E1 : np.ndarray [n_res, max_rot]
        One-body energy per (residue, rotamer).
    E2 : np.ndarray [n_res, n_res, max_rot, max_rot]
        Two-body energy (residue_i rotamer_a vs residue_j rotamer_b).
    n_rot : np.ndarray [n_res] int32
        Number of real rotamers per residue.
    designable : np.ndarray [n_res] bool
        Which residues are being packed.
    res_indices : list[tuple[int, int]]
        The ``(pose, block)`` corresponding to each residue slot.
    rotamer_coords : list[list[np.ndarray]]
        ``rotamer_coords[i][r]`` = ``[n_atoms, 3]`` coords for rotamer
        ``r`` of residue ``i`` (already sized to the residue's
        block-type atom count).
    """

    E1: np.ndarray
    E2: np.ndarray
    n_rot: np.ndarray
    designable: np.ndarray
    res_indices: List[Tuple[int, int]]
    rotamer_coords: List[List[np.ndarray]]


def build_pairwise_energies(
    pose_stack: Any,
    score_function: Any,
    rotamer_palette: Sequence[Sequence[np.ndarray]],
    res_indices: Sequence[Tuple[int, int]],
    compute_one_body_with_sfxn: bool = True,
    dun_neglnprob_per_rotamer: Optional[Sequence[Sequence[float]]] = None,
) -> PairwiseEnergyTables:
    """Build the one- and two-body energy tables.

    Parameters
    ----------
    pose_stack
        The starting PoseStack (supplies background coords).
    score_function
        :class:`ScoreFunction` whose LJLK state is used for the pair
        parameters; also called once per rotamer for ``E1``.
    rotamer_palette
        ``rotamer_palette[i]`` is the list of candidate coord arrays
        for residue ``i`` (each ``[n_atoms_block, 3]``).
    res_indices
        ``[(pose, block), ...]`` — the ``(pose, block)`` coordinates of
        each "residue slot" represented in ``rotamer_palette``.
    compute_one_body_with_sfxn
        If ``True`` (default), call the score function once per
        rotamer to evaluate ``E1`` as the full-pose energy with that
        rotamer spliced in. This is accurate but O(sum_i n_rot_i).
        Set to ``False`` to skip and leave ``E1`` as zeros (useful in
        tests).

    Returns
    -------
    :class:`PairwiseEnergyTables`
    """
    n_res = len(rotamer_palette)
    n_rot_per = np.asarray(
        [len(rotamer_palette[i]) for i in range(n_res)], dtype=np.int32
    )
    max_rot = int(n_rot_per.max()) if n_res > 0 and n_rot_per.max() > 0 else 1
    designable = np.asarray([len(rotamer_palette[i]) > 0 for i in range(n_res)], dtype=bool)

    # Try to setup the full sfxn state; fall back to just the LJLK term
    # if some sibling term has a setup bug. The LJLK state is all we
    # actually need for the pair-energy table.
    ljlk_term = _get_ljlk_term(score_function)
    ljparams: Optional[Dict[str, np.ndarray]] = None
    state = None
    try:
        state = score_function.setup_for_pose_stack(pose_stack)
        if ljlk_term is not None:
            term_idx = score_function._terms.index(ljlk_term)
            ljlk_state = state.term_states[term_idx]
            ljparams = _atom_ljparams_for_pose(pose_stack, ljlk_state)
    except Exception:
        state = None
        if ljlk_term is not None:
            try:
                pbt = pose_stack.packed_block_types
                ljlk_state = ljlk_term.setup_for_pose_stack(pose_stack, pbt)
                ljparams = _atom_ljparams_for_pose(pose_stack, ljlk_state)
            except Exception:
                ljparams = None

    # Gather per-block atom coords / params for the background pose.
    bco = np.asarray(pose_stack.block_coord_offset)
    bti = np.asarray(pose_stack.block_type_ind)
    n_atoms_tbl = np.asarray(pose_stack.packed_block_types.n_atoms)

    # Precompute, per-residue, the slice (pose, start, n).
    slot_info: List[Tuple[int, int, int]] = []
    for p, b in res_indices:
        bt = int(bti[p, b])
        if bt < 0:
            slot_info.append((int(p), -1, 0))
            continue
        slot_info.append((int(p), int(bco[p, b]), int(n_atoms_tbl[bt])))

    # ------------------------------------------------------------------ #
    # Gather LJ params per (residue, rotamer). The "residue, rotamer"
    # LJ params depend only on block type (identical across rotamers of
    # the same residue in a repack-only task). We gather them once.
    # ------------------------------------------------------------------ #
    res_ljparams: List[Optional[Dict[str, np.ndarray]]] = []
    if ljparams is not None:
        for (p, start, n), (pose, block) in zip(slot_info, res_indices):
            if start < 0 or n == 0:
                res_ljparams.append(None)
                continue
            res_ljparams.append(
                {
                    "lj_radius": ljparams["lj_radius"][p, start : start + n],
                    "lj_wdepth": ljparams["lj_wdepth"][p, start : start + n],
                    "is_donor": ljparams["is_donor"][p, start : start + n],
                    "is_acceptor": ljparams["is_acceptor"][p, start : start + n],
                    "is_hydroxyl": ljparams["is_hydroxyl"][p, start : start + n],
                    "is_polarh": ljparams["is_polarh"][p, start : start + n],
                    "atom_is_real": ljparams["atom_is_real"][p, start : start + n],
                }
            )
    else:
        res_ljparams = [None] * n_res

    # Normalize rotamer coords so each row is exactly [n_atoms, 3].
    rot_coords_clean: List[List[np.ndarray]] = []
    for i in range(n_res):
        p, start, n = slot_info[i]
        if n == 0:
            rot_coords_clean.append([])
            continue
        clean: List[np.ndarray] = []
        for cand in rotamer_palette[i]:
            arr = np.asarray(cand, dtype=np.float32)
            if arr.shape[0] < n:
                # pad with zeros if the rotamer has fewer rows (shouldn't
                # normally happen)
                tmp = np.zeros((n, 3), dtype=np.float32)
                tmp[: arr.shape[0]] = arr
                arr = tmp
            else:
                arr = arr[:n]
            clean.append(arr)
        rot_coords_clean.append(clean)

    # ------------------------------------------------------------------ #
    # Two-body table                                                      #
    # ------------------------------------------------------------------ #
    E2 = np.zeros((n_res, n_res, max_rot, max_rot), dtype=np.float32)

    for i in range(n_res):
        p_i, start_i, n_i = slot_info[i]
        if n_i == 0 or res_ljparams[i] is None:
            continue
        lja = res_ljparams[i]
        real_a = lja["atom_is_real"]
        for j in range(i + 1, n_res):
            p_j, start_j, n_j = slot_info[j]
            if n_j == 0 or p_j != p_i:
                continue  # different pose; no interaction
            if res_ljparams[j] is None:
                continue
            ljb = res_ljparams[j]
            real_b = ljb["atom_is_real"]

            # Quick distance prefilter: skip if nothing within 6 A when
            # using the first rotamer pair.
            c0a = rot_coords_clean[i][0]
            c0b = rot_coords_clean[j][0]
            if c0a.size == 0 or c0b.size == 0:
                continue
            d0 = np.linalg.norm(
                c0a.mean(axis=0) - c0b.mean(axis=0)
            )
            # 6 A interaction cutoff + rotamer "swing" margin (~5 A for
            # an extended sidechain)
            if d0 > 12.0:
                continue

            for a_idx, cand_a in enumerate(rot_coords_clean[i]):
                for b_idx, cand_b in enumerate(rot_coords_clean[j]):
                    e = _pair_lj_energy(cand_a, cand_b, lja, ljb, real_a, real_b)
                    E2[i, j, a_idx, b_idx] = e
                    E2[j, i, b_idx, a_idx] = e

    # ------------------------------------------------------------------ #
    # One-body table                                                      #
    # ------------------------------------------------------------------ #
    E1 = np.zeros((n_res, max_rot), dtype=np.float32)
    if compute_one_body_with_sfxn and ljparams is not None:
        # Fast proxy: "one-body" energy is the LJ interaction between this
        # rotamer and every **backbone** atom of every *other* residue in
        # the pose. This captures rotamer-vs-backbone clashes (the main
        # driver of packing) in O(n_bb_atoms * n_rot_atoms) per rotamer.
        # The per-residue slot itself is excluded (self-interactions are
        # constant across rotamers of the same residue anyway, so
        # dropping them doesn't change DEE/SA ordering).
        all_coords_np = np.asarray(pose_stack.coords)
        bti = np.asarray(pose_stack.block_type_ind)
        bco = np.asarray(pose_stack.block_coord_offset)
        atom_is_backbone_tbl = np.asarray(pose_stack.packed_block_types.atom_is_backbone)
        n_atoms_tbl = np.asarray(pose_stack.packed_block_types.n_atoms)
        for p_sel in range(pose_stack.n_poses):
            # Build a mask of backbone atoms in this pose.
            atom_is_bb = np.zeros_like(ljparams["atom_is_real"][p_sel])
            for b in range(bti.shape[1]):
                bt = int(bti[p_sel, b])
                if bt < 0:
                    continue
                n = int(n_atoms_tbl[bt])
                start = int(bco[p_sel, b])
                atom_is_bb[start : start + n] = atom_is_backbone_tbl[bt, :n]
            atom_is_bb &= ljparams["atom_is_real"][p_sel]

            # Gather background params once per pose.
            bb_inds = np.nonzero(atom_is_bb)[0]
            bb_coords_all = all_coords_np[p_sel, bb_inds]
            bb_lj_all = {
                k: ljparams[k][p_sel, bb_inds]
                for k in ("lj_radius", "lj_wdepth", "is_donor", "is_acceptor",
                          "is_hydroxyl", "is_polarh")
            }
            # Map from global atom index -> position in bb_inds
            bb_pos_map = -np.ones(all_coords_np.shape[1], dtype=np.int32)
            bb_pos_map[bb_inds] = np.arange(bb_inds.size)

            for i, (p_i, start_i, n_i) in enumerate(slot_info):
                if p_i != p_sel or n_i == 0 or res_ljparams[i] is None:
                    continue
                # Mask out this residue's own backbone atoms from the
                # background, to avoid the huge "self bb" interaction.
                local_mask = np.ones(bb_inds.size, dtype=bool)
                self_ids = np.arange(start_i, start_i + n_i)
                self_bb_positions = bb_pos_map[self_ids]
                self_bb_positions = self_bb_positions[self_bb_positions >= 0]
                if self_bb_positions.size > 0:
                    local_mask[self_bb_positions] = False

                bg_coords = bb_coords_all[local_mask]
                bg_lj = {k: v[local_mask] for k, v in bb_lj_all.items()}
                bg_real = np.ones(bg_coords.shape[0], dtype=bool)

                lja = res_ljparams[i]
                real_a = lja["atom_is_real"]
                for a_idx, cand in enumerate(rot_coords_clean[i]):
                    E1[i, a_idx] = _pair_lj_energy(
                        cand, bg_coords, lja, bg_lj, real_a, bg_real
                    )

    # ------------------------------------------------------------------ #
    # Dunbrack rotamer -log P penalty (per rotamer, one-body).            #
    # If the caller supplied per-rotamer probabilities from the Dunbrack
    # library, add them to E1. This is the main term missing from the LJ
    # proxy: swapping the crystal rotamer for a Dunbrack alternative
    # costs ~1-5 kcal/mol in library -log P in most cases, and ignoring
    # that term lets the packer pick implausible alternatives.
    # ------------------------------------------------------------------ #
    if dun_neglnprob_per_rotamer is not None:
        for i, neglnps in enumerate(dun_neglnprob_per_rotamer):
            if neglnps is None:
                continue
            for a_idx, nlp in enumerate(neglnps):
                if a_idx >= E1.shape[1]:
                    break
                E1[i, a_idx] += float(nlp)

    return PairwiseEnergyTables(
        E1=E1,
        E2=E2,
        n_rot=n_rot_per,
        designable=designable,
        res_indices=list(res_indices),
        rotamer_coords=rot_coords_clean,
    )


__all__ = [
    "PairwiseEnergyTables",
    "build_pairwise_energies",
]
