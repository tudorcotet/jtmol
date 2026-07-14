"""Adaptive Verlet neighbor list — pure JAX, differentiable, uncapped, exact.

The enabling primitive for the fused nonbonded scoring engine: an exact,
moving-coordinate neighbor list.

Why the fixed-capacity `score/common/neighbor_list.build_cell_list` cannot
certify a moving-coordinate energy/gradient:
  * it caps at the closest 512 neighbors (silently drops pairs in crowded
    systems), and
  * it is built once from setup coordinates with no displacement test, so a
    pair can cross the cutoff mid-trajectory without ever entering/leaving the
    list.

This replacement fixes both:

  * UNCAPPED candidate build at radius = cutoff + skin (exact O(N) cell list).
  * Adaptive validity test: on every call, max atom displacement since the last
    build is checked; when max_disp > skin/2 the candidate set may be missing a
    pair, so we rebuild. skin/2 is the tight, provably-correct threshold.
  * The within-cutoff mask is applied at evaluation time on the fixed candidate
    set, so the energy over the list == the brute-force within-cutoff energy,
    EXACTLY, and its gradient w.r.t. coords is exact.

Differentiability, stated honestly
-----------------------------------
A neighbor list is a set of integer index pairs; integers carry no gradient.
The gradient flows through the atom COORDINATES via r = |r_i - r_j|. The
candidate index set is treated as a constant (piecewise-constant in coords).
This is exactly how production MD neighbor lists work (LAMMPS, OpenMM,
jax-md's `neighbor_list`): a single energy evaluation on a fixed candidate set
is fully `jax.grad`-clean; the *rebuild* is host-orchestrated Python control
flow (dynamic shapes + a data-dependent branch), which pure JAX cannot jit and
does not need to. The true energy is continuous across a rebuild because the
within-cutoff mask selects the same subset regardless of how large the complete
candidate superset is.

Correctness threshold (why skin/2)
----------------------------------
If every atom moved <= D from its build position and D <= skin/2, then every
pair within `cutoff` at the new coords was a candidate at build time:
    r_ref <= r_new + |dr_i| + |dr_j| <= cutoff + 2D <= cutoff + skin.
So D <= skin/2 guarantees the candidate set is a complete superset of the
within-cutoff set. Tight.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------------- #
# Candidate build: uncapped exact cell list (host / numpy, non-differentiable) #
# --------------------------------------------------------------------------- #


def _candidate_pairs(
    coords_np: np.ndarray, radius: float, real_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """All real (i<j) pairs with |r_i - r_j| <= radius, via a 3x3x3 cell list.

    UNCAPPED and EXACT: cell size == radius, so the 27-cell stencil is a
    complete superset of every within-radius pair (if |a-b| <= cell_size then
    the per-axis cell index differs by at most 1). Returns global atom indices.
    """
    coords_np = np.asarray(coords_np, dtype=np.float64)
    real_idx = np.asarray(real_idx, dtype=np.int64)
    M = int(real_idx.size)
    if M < 2:
        e = np.empty(0, dtype=np.int32)
        return e, e.copy()

    xyz = coords_np[real_idx]  # [M, 3]
    r2 = float(radius) * float(radius)

    mn = xyz.min(axis=0)
    cell = np.floor((xyz - mn) / float(radius)).astype(np.int64)  # [M, 3] >= 0
    dims = cell.max(axis=0) + 1  # [3]
    Gy, Gz = int(dims[1]), int(dims[2])
    cid = (cell[:, 0] * Gy + cell[:, 1]) * Gz + cell[:, 2]  # [M]

    order = np.argsort(cid, kind="stable")
    cid_s = cid[order]
    atoms_sorted = real_idx[order]  # global atom index, in cell order
    xyz_sorted = xyz[order]

    ncell = int(dims.prod())
    ar = np.arange(ncell + 1, dtype=cid_s.dtype)  # +1 guards flat ids at the max
    cell_start = np.searchsorted(cid_s, ar, side="left")
    cell_end = np.searchsorted(cid_s, ar, side="right")

    stencil = np.array(
        [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=np.int64,
    )

    out_i: list[np.ndarray] = []
    out_j: list[np.ndarray] = []
    for m in range(M):
        gi = int(real_idx[m])
        xi = xyz[m]
        cx, cy, cz = int(cell[m, 0]), int(cell[m, 1]), int(cell[m, 2])
        ranges = []
        for dx, dy, dz in stencil:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if nx < 0 or nx >= dims[0] or ny < 0 or ny >= Gy or nz < 0 or nz >= Gz:
                continue
            nc = (nx * Gy + ny) * Gz + nz
            lo, hi = int(cell_start[nc]), int(cell_end[nc])
            if hi > lo:
                ranges.append(np.arange(lo, hi))
        if not ranges:
            continue
        cand = np.concatenate(ranges)
        cand_atoms = atoms_sorted[cand]
        d = xyz_sorted[cand] - xi
        dsq = np.einsum("kd,kd->k", d, d)
        keep = (dsq <= r2) & (cand_atoms > gi)  # i<j in GLOBAL index, exact superset
        if keep.any():
            js = cand_atoms[keep]
            out_i.append(np.full(js.shape, gi, dtype=np.int32))
            out_j.append(js.astype(np.int32))

    if not out_i:
        e = np.empty(0, dtype=np.int32)
        return e, e.copy()
    return np.concatenate(out_i), np.concatenate(out_j)


# --------------------------------------------------------------------------- #
# The adaptive list                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VerletList:
    """Immutable candidate list + the build state needed to validate it.

    ``pairs_i`` / ``pairs_j``: int32[E] candidate pairs (i<j), within
    ``cutoff + skin`` at build time. ``ref_coords``: [M,3] build coords of the
    real atoms (for the displacement test). ``real_idx``: int64[M] global
    indices of real atoms.
    """

    pairs_i: jax.Array
    pairs_j: jax.Array
    ref_coords: np.ndarray  # host; the displacement baseline
    real_idx: np.ndarray  # host
    cutoff: float
    skin: float


def _coords_np(coords) -> np.ndarray:
    return np.asarray(coords, dtype=np.float64)


def build_verlet_list(coords, cutoff: float, skin: float = 1.0, atom_is_real=None) -> VerletList:
    """Build the uncapped candidate list at radius ``cutoff + skin``.

    ``coords``: [N,3]. ``atom_is_real``: optional bool[N] (default: all real).
    """
    c = _coords_np(coords)
    if c.ndim != 2 or c.shape[-1] != 3:
        raise ValueError(f"coords must be [N,3], got {c.shape}")
    N = c.shape[0]
    if atom_is_real is None:
        real_idx = np.arange(N, dtype=np.int64)
    else:
        real_idx = np.nonzero(np.asarray(atom_is_real, dtype=bool))[0].astype(np.int64)
    if skin <= 0.0:
        raise ValueError("skin must be > 0 (skin==0 has no displacement margin)")

    pi, pj = _candidate_pairs(c, float(cutoff) + float(skin), real_idx)
    return VerletList(
        pairs_i=jnp.asarray(pi),
        pairs_j=jnp.asarray(pj),
        ref_coords=c[real_idx],
        real_idx=real_idx,
        cutoff=float(cutoff),
        skin=float(skin),
    )


@jax.jit
def _max_disp(coords_real: jax.Array, ref_real: jax.Array) -> jax.Array:
    d = coords_real - ref_real
    return jnp.sqrt(jnp.max((d * d).sum(axis=-1)))


def max_displacement(vl: VerletList, coords) -> float:
    """Max displacement (Angstrom) of any real atom since the list was built."""
    cr = jnp.asarray(_coords_np(coords)[vl.real_idx])
    return float(_max_disp(cr, jnp.asarray(vl.ref_coords)))


def needs_rebuild(vl: VerletList, coords) -> bool:
    """True iff the candidate set may be missing a within-cutoff pair.

    Trigger is ``max_disp > skin/2`` — the tight completeness threshold.
    """
    return max_displacement(vl, coords) > 0.5 * vl.skin


def maybe_rebuild(vl: VerletList, coords) -> VerletList:
    """Return a rebuilt list if the current one is stale, else ``vl`` unchanged."""
    if needs_rebuild(vl, coords):
        real = np.zeros(int(np.asarray(coords).shape[0]), dtype=bool)
        real[vl.real_idx] = True
        return build_verlet_list(coords, vl.cutoff, vl.skin, atom_is_real=real)
    return vl


# --------------------------------------------------------------------------- #
# Differentiable evaluation on the fixed candidate set                         #
# --------------------------------------------------------------------------- #


def pair_distances(coords: jax.Array, vl: VerletList) -> jax.Array:
    """r[E] = |coords[i] - coords[j]| over candidate pairs. Differentiable."""
    d = coords[vl.pairs_i] - coords[vl.pairs_j]
    return jnp.sqrt((d * d).sum(axis=-1))


def within_cutoff_pair_energy(coords: jax.Array, vl: VerletList, pair_fn) -> jax.Array:
    """Sum ``pair_fn(r)`` over within-cutoff pairs. Differentiable scalar.

    The hard ``r <= cutoff`` mask matches brute force exactly; its derivative is
    zero a.e. (measure-zero boundary), so the gradient is the exact
    within-cutoff force. ``jnp.where`` guards the value; for potentials singular
    inside the skin band, wrap ``pair_fn`` in the standard double-where so the
    dead branch cannot poison the gradient. (None of jtmol's terms are singular
    in (cutoff, cutoff+skin], so the plain form is safe here.)
    """
    r = pair_distances(coords, vl)
    within = r <= vl.cutoff
    contrib = jnp.where(within, pair_fn(r), 0.0)
    return contrib.sum()


# --------------------------------------------------------------------------- #
# Host-side exactness helpers (for the self-check; consistent float64)         #
# --------------------------------------------------------------------------- #


def within_cutoff_set(vl: VerletList, coords) -> set:
    """{(i,j)} within cutoff, obtained by filtering the candidate list (float64)."""
    c = _coords_np(coords)
    pi = np.asarray(vl.pairs_i)
    pj = np.asarray(vl.pairs_j)
    d = c[pi] - c[pj]
    dsq = np.einsum("kd,kd->k", d, d)
    keep = dsq <= vl.cutoff * vl.cutoff
    return set(zip(pi[keep].tolist(), pj[keep].tolist()))


def brute_within_cutoff_set(coords, cutoff: float, real_idx=None) -> set:
    """Reference: every real i<j with |r_i - r_j| <= cutoff (float64, O(N^2))."""
    c = _coords_np(coords)
    if real_idx is None:
        real_idx = np.arange(c.shape[0], dtype=np.int64)
    else:
        real_idx = np.asarray(real_idx, dtype=np.int64)
    x = c[real_idx]
    diff = x[:, None, :] - x[None, :, :]
    dsq = np.einsum("ijd,ijd->ij", diff, diff)
    a, b = np.triu_indices(real_idx.size, k=1)
    keep = dsq[a, b] <= cutoff * cutoff
    ga, gb = real_idx[a[keep]], real_idx[b[keep]]
    return set(zip(ga.tolist(), gb.tolist()))


# --------------------------------------------------------------------------- #
# SoA work-plan skeleton (SECONDARY — the layout a fused kernel consumes)      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NonbondedWorkPlan:
    """Compact structure-of-arrays a fused nonbonded kernel would read.

    Single topology, one pose. ``coords`` [N,3] is the only differentiable field; ``atom_type`` [N]
    and ``params`` (name -> per-atom array) are static handles the kernel
    dereferences into its parameter tables. ``verlet`` supplies the exact,
    adaptive candidate edge list. A multi-pose S3 plan concatenates these
    ragged per-topology plans sorted by (pose, source_atom); not built until the
    fused kernel needs it.
    """

    coords: jax.Array  # [N,3] differentiable
    atom_type: jax.Array  # [N] int32 static
    params: dict  # name -> [N] or [N,...] static handles
    verlet: VerletList


def build_work_plan(coords, atom_type, cutoff, skin=1.0, atom_is_real=None, **params) -> NonbondedWorkPlan:
    return NonbondedWorkPlan(
        coords=jnp.asarray(coords),
        atom_type=jnp.asarray(atom_type, dtype=jnp.int32),
        params=dict(params),
        verlet=build_verlet_list(coords, cutoff, skin, atom_is_real=atom_is_real),
    )


# --------------------------------------------------------------------------- #
# Assert-based self-check                                                       #
# --------------------------------------------------------------------------- #


def demo() -> None:
    """Fails loudly if the candidate set or the within-cutoff pair set is wrong.

    Loads 1ubq, cutoff 6.0 A, skin 1.0 A (rebuild threshold 0.5 A), and checks:
      1. static:        within-cutoff set == brute force, EXACTLY.
      2. candidate superset: candidate set (radius cutoff+skin) is a superset.
      3. small perturb (<= 0.3 A, no rebuild): skin catches cutoff-crossers ->
         within-cutoff set STILL == brute force without rebuilding.
      4. large perturb (~2 A): rebuild triggers -> within-cutoff set == brute.
      5. jax.grad of sum(1/r over within-cutoff pairs) is finite (no NaN).
    """
    import os

    import jtmol

    pdb = os.path.join(os.path.dirname(jtmol.__file__), "tests", "data", "1ubq.pdb")
    ps = jtmol.pose_stack_from_pdb(pdb)
    coords0 = np.asarray(ps.coords[0], dtype=np.float64)  # [N,3], 1ubq all-real
    N = coords0.shape[0]
    cutoff, skin = 6.0, 1.0

    vl = build_verlet_list(coords0, cutoff, skin)
    n_cand = int(vl.pairs_i.size)
    print(f"[verlet] 1ubq N={N} atoms, cutoff={cutoff} skin={skin} "
          f"-> {n_cand} candidate pairs (uncapped)")

    # 1. static exactness
    got = within_cutoff_set(vl, coords0)
    ref = brute_within_cutoff_set(coords0, cutoff)
    assert got == ref, (
        f"STATIC pair set wrong: missing {len(ref - got)}, extra {len(got - ref)}"
    )
    print(f"[1] static within-cutoff set == brute force: {len(ref)} pairs, EXACT")

    # 2. candidate set is a complete superset at radius cutoff+skin
    cand_all = set(zip(np.asarray(vl.pairs_i).tolist(), np.asarray(vl.pairs_j).tolist()))
    ref_super = brute_within_cutoff_set(coords0, cutoff + skin)
    assert ref_super.issubset(cand_all), (
        f"candidate cell list MISSED {len(ref_super - cand_all)} pairs within cutoff+skin"
    )
    assert ref.issubset(cand_all)
    print(f"[2] candidate cell list is exact superset at cutoff+skin: "
          f"{len(ref_super)}/{n_cand} within radius, no misses")

    # 3. small perturbation, magnitude <= 0.3 A: within skin/2=0.5 -> NO rebuild,
    #    yet the skin must still catch every cutoff-crosser.
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    coords_small = coords0 + dirs * (0.3 * rng.random((N, 1)))  # each |dr| <= 0.3
    assert not needs_rebuild(vl, coords_small), "small move must NOT trigger rebuild"
    got_s = within_cutoff_set(vl, coords_small)  # SAME candidate set, no rebuild
    ref_s = brute_within_cutoff_set(coords_small, cutoff)
    assert got_s == ref_s, (
        f"SMALL-PERTURB set wrong w/o rebuild: missing {len(ref_s - got_s)}, "
        f"extra {len(got_s - ref_s)} -- skin failed to catch a cutoff-crosser"
    )
    print(f"[3] +0.3A perturb, no rebuild (max_disp={max_displacement(vl, coords_small):.3f}"
          f" <= {skin/2}): set STILL == brute ({len(ref_s)} pairs), cutoff-crossers caught")

    # 4. large perturbation ~2 A: must rebuild, then exact.
    coords_big = coords0 + rng.normal(size=(N, 3)) * 2.0
    assert needs_rebuild(vl, coords_big), "large move MUST trigger rebuild"
    vl2 = maybe_rebuild(vl, coords_big)
    assert vl2 is not vl, "maybe_rebuild should have returned a fresh list"
    got_b = within_cutoff_set(vl2, coords_big)
    ref_b = brute_within_cutoff_set(coords_big, cutoff)
    assert got_b == ref_b, (
        f"POST-REBUILD set wrong: missing {len(ref_b - got_b)}, extra {len(got_b - ref_b)}"
    )
    print(f"[4] +2.0A perturb (max_disp={max_displacement(vl, coords_big):.3f} > {skin/2}): "
          f"rebuilt -> set == brute ({len(ref_b)} pairs), EXACT")

    # 5. gradient flows, no NaN.
    cj = jnp.asarray(coords0)
    grad_fn = jax.jit(jax.grad(lambda c: within_cutoff_pair_energy(c, vl, lambda r: 1.0 / r)))
    g = grad_fn(cj)
    assert np.all(np.isfinite(np.asarray(g))), "gradient has NaN/Inf"
    assert g.shape == cj.shape
    print(f"[5] jax.grad of sum(1/r) over within-cutoff pairs: finite, shape {g.shape}, "
          f"|grad|_max={float(jnp.abs(g).max()):.3e}")

    print("\nALL CHECKS PASSED — uncapped adaptive Verlet list is exact "
          "(static + perturbed + post-rebuild) and differentiable.")


if __name__ == "__main__":
    demo()
