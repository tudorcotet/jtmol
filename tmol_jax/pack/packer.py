"""Top-level rotamer packer: enumerate -> E-tables -> DEE -> SA -> impose.

This module orchestrates the pipeline:

    1. ``sample_rotamers`` over every designable residue.
    2. ``build_pairwise_energies`` to compute one- and two-body tables.
    3. ``goldstein_dee`` to prune dominated rotamers.
    4. ``run_sa`` to find a low-energy assignment on the reduced graph.
    5. Impose the best assignment as coords on a new ``PoseStack``.

It mirrors the upstream ``tmol.pack`` pipeline but in pure Python + JAX.
The public entry point is :func:`pack_rotamers`; a :class:`PackerTask`
captures the per-residue rotamer palette and "designable" bitmap.

See the module docstrings in ``rotamer_sampler.py``,
``pairwise_energies.py``, ``dee.py``, ``sim_annealer.py`` for the
"exact vs simplified" notes on each stage.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import attrs
import jax.numpy as jnp
import numpy as np

from tmol_jax.pack.dee import goldstein_dee
from tmol_jax.pack.pairwise_energies import (
    PairwiseEnergyTables,
    build_pairwise_energies,
)
from tmol_jax.pack.rotamer_sampler import sample_rotamers
from tmol_jax.pack.sim_annealer import run_sa


# --------------------------------------------------------------------------- #
# Phi / psi measurement on the starting pose                                   #
# --------------------------------------------------------------------------- #


def _measure_phi_psi(pose_stack: Any, pose: int, block: int) -> Tuple[float, float]:
    """Return ``(phi_deg, psi_deg)`` for ``(pose, block)``.

    Falls back to the default ``(-60, -45)`` helix values if the
    torsion can't be resolved (terminal residue, non-alpha polymer,
    etc.). We compute the dihedrals on numpy arrays extracted from the
    pose's coord slab — this is CPU-side Python but only called once
    per residue so the overhead is fine.
    """
    import math as _math

    bt = pose_stack.block_type(pose, block)
    t2u = getattr(bt, "torsion_to_uaids", None)
    if t2u is None:
        return -60.0, -45.0
    coords = np.asarray(pose_stack.coords[pose])
    bco = np.asarray(pose_stack.block_coord_offset[pose])
    irc = np.asarray(pose_stack.inter_residue_connections[pose])
    bti = np.asarray(pose_stack.block_type_ind[pose])
    pbt = pose_stack.packed_block_types
    conn_atoms = np.asarray(pbt.conn_atoms)

    def _resolve(uaids, host_block):
        out = np.empty((4, 3), dtype=np.float64)
        host_bt = int(bti[host_block])
        host_start = int(bco[host_block])
        for k in range(4):
            atom_id, conn_id, _ = uaids[k]
            if atom_id >= 0:
                out[k] = coords[host_start + int(atom_id)]
            elif conn_id >= 0:
                # Inter-residue: follow connection to neighbour block.
                other_block, other_conn = irc[host_block, conn_id]
                if other_block < 0:
                    return None
                other_bt = int(bti[int(other_block)])
                if other_bt < 0:
                    return None
                partner_atom = int(conn_atoms[other_bt, int(other_conn)])
                if partner_atom < 0:
                    return None
                out[k] = coords[int(bco[int(other_block)]) + partner_atom]
            else:
                return None
        return out

    def _dihedral(pts):
        b1 = pts[1] - pts[0]
        b2 = pts[2] - pts[1]
        b3 = pts[3] - pts[2]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
        return _math.degrees(_math.atan2(float(n2 @ m1), float(n1 @ n2)))

    phi = -60.0
    psi = -45.0
    if "phi" in t2u:
        pts = _resolve(t2u["phi"], block)
        if pts is not None:
            phi = _dihedral(pts)
    if "psi" in t2u:
        pts = _resolve(t2u["psi"], block)
        if pts is not None:
            psi = _dihedral(pts)
    return phi, psi


# --------------------------------------------------------------------------- #
# PackerTask                                                                   #
# --------------------------------------------------------------------------- #


@attrs.define
class PackerTask:
    """Per-residue rotamer palette.

    Fields
    ------
    rotamers
        ``rotamers[pose_ind][block_ind]`` is a list of candidate
        ``[n_atoms, 3]`` coord arrays for block ``block_ind`` of pose
        ``pose_ind``.
    designable
        ``bool[n_poses, max_n_blocks]`` — which positions are being
        repacked / designed.
    neglnprob
        Optional — ``neglnprob[pose_ind][block_ind]`` matches
        ``rotamers[pose_ind][block_ind]`` and carries ``-log P(rotamer
        | phi, psi)`` from the Dunbrack library (``0`` for
        non-library rotamers). Used by :func:`pack_rotamers` to fold a
        rotamer-plausibility prior into the one-body energies.
    """

    rotamers: List[List[List[np.ndarray]]]
    designable: np.ndarray
    neglnprob: List[List[List[float]]] = attrs.Factory(list)

    @classmethod
    def all_repack(
        cls,
        pose_stack: Any,
        n_rots: int = 8,
        dun_db: Optional[Any] = None,
    ) -> "PackerTask":
        """Build a task that repacks every residue with up to ``n_rots``
        candidates per position.

        If ``dun_db`` is ``None`` we look for one on ``pose_stack``'s
        default parameter database; if none is available, each position
        gets a single "ideal-coord" rotamer and the packer is a no-op.
        """
        P, B = pose_stack.block_type_ind.shape
        pbt = pose_stack.packed_block_types
        block_type_ind = np.asarray(pose_stack.block_type_ind)

        if dun_db is None:
            try:
                from tmol_jax.database import ParameterDatabase

                dun_db = ParameterDatabase.get_default().scoring.dunbrack
            except Exception:
                dun_db = None

        rotamers: List[List[List[np.ndarray]]] = []
        neglnprob: List[List[List[float]]] = []
        designable = np.zeros((P, B), dtype=bool)
        for p in range(P):
            pose_rots: List[List[np.ndarray]] = []
            pose_nlp: List[List[float]] = []
            for b in range(B):
                bt = int(block_type_ind[p, b])
                if bt < 0:
                    pose_rots.append([])
                    pose_nlp.append([])
                    continue
                restype = pbt.active_block_types[bt]
                phi, psi = _measure_phi_psi(pose_stack, p, b)
                # Start each rotamer from the *current* backbone coords
                # so applying the chi rotations leaves the backbone intact.
                start = int(pose_stack.block_coord_offset[p, b])
                n = int(pbt.n_atoms[bt])
                base_now = np.asarray(pose_stack.coords[p, start : start + n])
                coords_list, nlp_list = sample_rotamers(
                    restype,
                    phi=phi,
                    psi=psi,
                    n_rots=n_rots,
                    base_coords=base_now,
                    dun_db=dun_db,
                    return_probs=True,
                )
                pose_rots.append(list(coords_list))
                pose_nlp.append(list(nlp_list))
                designable[p, b] = True
            rotamers.append(pose_rots)
            neglnprob.append(pose_nlp)
        return cls(rotamers=rotamers, designable=designable, neglnprob=neglnprob)


# --------------------------------------------------------------------------- #
# Pack                                                                         #
# --------------------------------------------------------------------------- #


def _build_palette(
    packer_task: PackerTask,
) -> Tuple[
    List[List[np.ndarray]],
    List[Tuple[int, int]],
    np.ndarray,
    List[List[float]],
]:
    """Flatten the (P, B)-nested palette into a linear ``[n_res]`` list.

    Returns
    -------
    palette
        ``palette[i]`` = list of rotamer coord arrays for residue slot ``i``.
    res_indices
        ``[(pose, block), ...]`` matching the palette order.
    designable
        ``bool[n_res]`` flattened version of ``packer_task.designable``.
    neglnprob
        ``neglnprob[i][r]`` = per-rotamer -log P (empty lists if the
        task doesn't carry library probabilities).
    """
    palette: List[List[np.ndarray]] = []
    res_indices: List[Tuple[int, int]] = []
    des: List[bool] = []
    nlp_out: List[List[float]] = []
    for p in range(len(packer_task.rotamers)):
        pose_nlp = (
            packer_task.neglnprob[p]
            if packer_task.neglnprob and p < len(packer_task.neglnprob)
            else None
        )
        for b in range(len(packer_task.rotamers[p])):
            rots = packer_task.rotamers[p][b]
            if not rots:
                continue
            palette.append(rots)
            res_indices.append((p, b))
            des.append(bool(packer_task.designable[p, b]))
            if pose_nlp is not None and b < len(pose_nlp):
                nlp_out.append(list(pose_nlp[b]))
            else:
                nlp_out.append([0.0] * len(rots))
    return palette, res_indices, np.asarray(des, dtype=bool), nlp_out


def _impose_assignment(
    pose_stack: Any,
    tables: PairwiseEnergyTables,
    assignment: np.ndarray,
) -> Any:
    """Produce a new ``PoseStack`` with the chosen rotamer coords spliced in."""
    base_coords = np.asarray(pose_stack.coords).copy()
    bco = np.asarray(pose_stack.block_coord_offset)
    n_atoms_tbl = np.asarray(pose_stack.packed_block_types.n_atoms)
    bti = np.asarray(pose_stack.block_type_ind)
    for i, (p, b) in enumerate(tables.res_indices):
        r = int(assignment[i])
        if r < 0:
            continue
        bt = int(bti[p, b])
        if bt < 0:
            continue
        n = int(n_atoms_tbl[bt])
        start = int(bco[p, b])
        coords = tables.rotamer_coords[i][r]
        base_coords[p, start : start + n] = coords[:n]
    return pose_stack.replace(coords=jnp.asarray(base_coords, dtype=jnp.float32))


def pack_rotamers(
    pose_stack: Any,
    score_function: Any,
    packer_task: Optional[PackerTask] = None,
    n_mc_steps: int = 1000,
    temperature_hi: float = 100.0,
    temperature_lo: float = 0.3,
    run_dee: bool = True,
    compute_one_body_with_sfxn: bool = True,
    seed: int = 0,
    n_rots: int = 8,
) -> Any:
    """Pack the side-chains of ``pose_stack`` via the full pipeline.

    Parameters
    ----------
    pose_stack
        Starting PoseStack.
    score_function
        :class:`ScoreFunction`. Its LJLK term provides the pair
        parameters for the pair-energy table, and it's called once
        per rotamer for the one-body term (unless
        ``compute_one_body_with_sfxn=False``).
    packer_task
        Per-residue rotamer palette. If omitted, a
        :meth:`PackerTask.all_repack` is built with ``n_rots``
        rotamers per position drawn from the Dunbrack library.
    n_mc_steps
        Number of SA proposals to attempt.
    temperature_hi, temperature_lo
        Endpoints of the geometric temperature schedule.
    run_dee
        Whether to run Goldstein DEE pruning before SA. DEE is cheap
        on small palettes but O(n_res * max_rot^2 * n_res * max_rot)
        per iteration so it can dominate for very large problems;
        set ``False`` to skip.
    compute_one_body_with_sfxn
        If ``False``, skip the sfxn-based one-body evaluation (``E1``
        stays zero). DEE + SA then pick on pairwise terms alone. Faster
        but less accurate.
    seed
        RNG seed.
    n_rots
        Rotamers to request per position when building the default
        :class:`PackerTask`.

    Returns
    -------
    PoseStack
        A new PoseStack with the best-found rotamer assignment
        spliced into the coords.
    """
    if packer_task is None:
        packer_task = PackerTask.all_repack(pose_stack, n_rots=n_rots)

    palette, res_indices, designable, neglnp = _build_palette(packer_task)
    if not palette:
        return pose_stack

    # Build the energy tables.
    tables = build_pairwise_energies(
        pose_stack,
        score_function,
        palette,
        res_indices,
        compute_one_body_with_sfxn=compute_one_body_with_sfxn,
        dun_neglnprob_per_rotamer=neglnp,
    )

    # DEE pruning.
    if run_dee:
        keep, _n_pruned = goldstein_dee(
            tables.E1, tables.E2, tables.n_rot, designable
        )
    else:
        keep = np.zeros_like(tables.E1, dtype=bool)
        for i in range(tables.E1.shape[0]):
            keep[i, : int(tables.n_rot[i])] = True

    # SA.
    assignment, best_E = run_sa(
        tables.E1,
        tables.E2,
        keep,
        tables.n_rot,
        designable,
        n_steps=n_mc_steps,
        T_hi=temperature_hi,
        T_lo=temperature_lo,
        seed=seed,
    )

    # Impose.
    out = _impose_assignment(pose_stack, tables, assignment)

    # Final sanity: if the SA-produced pose is higher-energy than the
    # starting pose, fall back to the starting pose. SA should monotonically
    # improve (or match) against the start given we always include the
    # starting rotamer as option 0. We guard with try/except because the
    # score function can have intermittent setup failures in some
    # environments; in that case just return the SA pose.
    try:
        e_start = float(score_function(pose_stack).sum())
        e_final = float(score_function(out).sum())
        if e_final > e_start + 1e-3:
            return pose_stack
    except Exception:
        pass
    return out


__all__ = ["PackerTask", "pack_rotamers"]
