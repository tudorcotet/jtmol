"""Integrate tmol_jax as additional loss terms in the mosaic / Boltz-2
binder hallucinator from
https://gist.github.com/nboyd/7b30c4033f834fc532a106d902fbdffa
using the exact mosaic ``LossTerm`` API at
https://github.com/escalante-bio/mosaic.

## What we provide

Three loss terms that compose with mosaic losses via ``w * loss_a + loss_b``:

1. ``TmolInterfaceLoss`` — **binding-affinity proxy**. Scores only the
   cross-chain (binder↔target) atom pair contributions to Rosetta's
   beta_nov2016_cart energy. This mirrors Rosetta's ``InterfaceAnalyzer
   dG_separated`` metric, which correlates with experimental ΔΔG
   across many design campaigns.

   Per-term weights are exposed so you can tune "what matters for
   binding" (shape complementarity via ``fa_ljatr``, electrostatics
   via ``fa_elec``, H-bonds via ``hbond``, etc.).

2. ``TmolBackboneLoss`` — design-time backbone physics. Cart-bonded +
   rama + omega. Works with the hallucinator's X-residue binder
   during design.

3. ``TmolClashFilter`` — cheap ``fa_ljrep``-only term; strong gradient
   to push out of clashing geometries.

## API (matches mosaic LossTerm exactly)

```python
class LossTerm(eqx.Module):
    def __call__(self, sequence: [N, 20], output: StructureModelOutput,
                 key) -> tuple[float, dict[str, Array]]: ...
```

``StructureModelOutput`` has:
  * ``backbone_coordinates: [N, 4, 3]``   — N/CA/C/O
  * ``structure_coordinates``             — all-atom coords
  * ``full_sequence: [N, 20]``            — PSSM / one-hot
  * plus pae, plddt, distogram, etc.

Gradient flow: Boltz-2 coords → tmol_jax (pure JAX) → scalar loss.
``jax.grad`` flows back through both layers, so the mosaic
``simplex_APGM`` optimizer updates the PSSM using the physics gradient
as an additional signal alongside the ML structure-prediction gradients.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


# --------------------------------------------------------------------- #
# Mirror of mosaic's LossTerm protocol (for stand-alone testing).        #
# --------------------------------------------------------------------- #
# When mosaic is installed, replace these with:
#     from mosaic.common import LossTerm
#     from mosaic.losses.structure_prediction import StructureModelOutput


class _StandaloneLossTerm(eqx.Module):
    """Stand-in for mosaic.common.LossTerm when mosaic is not importable.

    Implements the same composition operators so ``0.1 * x + 0.05 * y``
    produces a callable with the same semantics. When mosaic is
    installed, subclass ``mosaic.common.LossTerm`` instead — the
    resulting losses compose across both sources.
    """

    def __call__(self, *args, key, **kwargs):
        raise NotImplementedError

    def __rmul__(self, scalar: float):
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return _StandaloneLinearCombination(l=[self], weights=jnp.array([scalar]))

    def __add__(self, other):
        return 1.0 * self + 1.0 * other

    def __sub__(self, other):
        return self + (-1.0) * other

    def __neg__(self):
        return (-1.0) * self


class _StandaloneLinearCombination(eqx.Module):
    l: list
    weights: jax.Array

    def __call__(self, *args, key, **kwargs):
        r = 0.0
        aux = []
        for w, loss in zip(self.weights, self.l):
            v, a = loss(*args, key=key, **kwargs)
            key = jax.random.fold_in(key, 1)
            r = r + w * v
            aux.append(a)
        return r, aux

    def __rmul__(self, scalar):
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return _StandaloneLinearCombination(l=self.l, weights=self.weights * scalar)

    def __add__(self, other):
        if isinstance(other, _StandaloneLossTerm):
            other = 1.0 * other
        if not isinstance(other, _StandaloneLinearCombination):
            return NotImplemented
        return _StandaloneLinearCombination(
            l=self.l + other.l,
            weights=jnp.concatenate([self.weights, other.weights]),
        )


# Try to use the real mosaic base class when available.
try:
    from mosaic.common import LossTerm as _MosaicLossTerm

    _BaseLoss = _MosaicLossTerm
except Exception:  # pragma: no cover
    _BaseLoss = _StandaloneLossTerm


# --------------------------------------------------------------------- #
# Coord plumbing: atom37 → tmol_jax flat layout                          #
# --------------------------------------------------------------------- #
#
# Boltz-2 output ``structure_coordinates`` is atom-level coords with
# shape model-dependent. For the common all-atom case the layout is
# ``[N_res, 37, 3]`` matching AlphaFold's atom37 convention. tmol_jax
# uses its own packed per-residue ordering keyed by atom name.
#
# The mapping is sequence-dependent but coord-independent, so we build
# it once per template and re-use it at every eval.


# Standard AlphaFold atom37 ordering (copied from
# openfold.np.residue_constants.atom_types).
_ATOM37_NAMES = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1",
    "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE",
    "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1",
    "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)


def _build_atom37_to_flat_mapping(
    pose_stack: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each atom in the tmol_jax flat layout, find the (res, slot)
    pair in the atom37 convention. Returns:

    - ``flat_from_atom37[k]`` = ``res_k * 37 + slot_k``
    - ``is_mapped[k]`` = True if the atom has an atom37 counterpart (N,
      CA, etc. are; variant-specific H1/H2/H3 may not be).
    """
    pbt = pose_stack.packed_block_types
    bti = np.asarray(pose_stack.block_type_ind[0])
    offsets = np.asarray(pose_stack.block_coord_offset[0])
    n_atoms_per_type = np.asarray(pbt.n_atoms)

    atom37_slot = {name: i for i, name in enumerate(_ATOM37_NAMES)}
    total = int(sum(int(n_atoms_per_type[b]) for b in bti if b >= 0))

    flat_from_atom37 = np.zeros(total, dtype=np.int32)
    is_mapped = np.zeros(total, dtype=bool)

    for res_idx, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(n_atoms_per_type[bt])
        start = int(offsets[res_idx])
        atom_names = pbt.atom_names[bt]  # tuple of strings, len == n
        for local in range(n):
            name = atom_names[local]
            if name in atom37_slot:
                flat_from_atom37[start + local] = res_idx * 37 + atom37_slot[name]
                is_mapped[start + local] = True
            else:
                # Variant-specific H atom (H1/H2/H3 on N-term, etc.) —
                # not in the standard atom37 table. Fall back to
                # residue 0 slot 0 (any valid index); the is_mapped
                # flag stops us from reading it.
                flat_from_atom37[start + local] = 0
    return flat_from_atom37, is_mapped


def _build_residue_to_chain_mapping(
    pose_stack: Any,
) -> np.ndarray:
    """Assign each atom in the flat layout to a chain (int id per atom).

    Prefers ``pose_stack.pdb_info.chain_id`` when available (reliable
    string labels set during construction); falls back to walking
    ``inter_residue_connections`` when no PDB info is present.
    """
    pbt = pose_stack.packed_block_types
    bti = np.asarray(pose_stack.block_type_ind[0])
    offsets = np.asarray(pose_stack.block_coord_offset[0])
    n_atoms_per_type = np.asarray(pbt.n_atoms)

    n_res = len(bti)
    chain_id_per_res = np.full(n_res, -1, dtype=np.int32)

    pdb_info = getattr(pose_stack, "pdb_info", None)
    if pdb_info is not None and getattr(pdb_info, "chain_id", None) is not None:
        chain_labels = pdb_info.chain_id[0]  # per-residue string labels
        # Map unique strings to int ids in encounter order
        seen: dict = {}
        for r, label in enumerate(chain_labels):
            if bti[r] < 0 or label in (None, ""):
                continue
            if label not in seen:
                seen[label] = len(seen)
            chain_id_per_res[r] = seen[label]
    else:
        # Fallback: union-find over peptide-bond connections.
        parent = list(range(n_res))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        irc = np.asarray(pose_stack.inter_residue_connections[0])  # [B, C, 2]
        for r in range(n_res):
            if bti[r] < 0:
                continue
            for c in range(irc.shape[1]):
                other = int(irc[r, c, 0])
                if other >= 0 and bti[other] >= 0:
                    union(r, other)
        # Assign chain ids in encounter order.
        root_to_id: dict = {}
        for r in range(n_res):
            if bti[r] < 0:
                continue
            root = find(r)
            if root not in root_to_id:
                root_to_id[root] = len(root_to_id)
            chain_id_per_res[r] = root_to_id[root]

    total_atoms = int(sum(int(n_atoms_per_type[b]) for b in bti if b >= 0))
    atom_chain_id = np.zeros(total_atoms, dtype=np.int32)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(n_atoms_per_type[bt])
        start = int(offsets[r])
        atom_chain_id[start : start + n] = chain_id_per_res[r]
    return atom_chain_id


def _pose_stack_with_atom37_coords(
    pose_stack: Any,
    atom37_coords: Float[Array, "N 37 3"],
    flat_from_atom37: Array,
    is_mapped: Array,
) -> Any:
    """Return a PoseStack with coords replaced from an atom37 array.

    Atoms absent in the atom37 input (variant H1/H2/H3, etc.) keep
    their template coords.
    """
    template_coords = pose_stack.coords  # [1, max_n_atoms, 3]
    # Gather:
    flat = atom37_coords.reshape(-1, 3)[flat_from_atom37]  # [total_atoms, 3]
    max_n = template_coords.shape[1]
    total = flat.shape[0]

    new_coords = template_coords.at[0, :total].set(
        jnp.where(is_mapped[:, None], flat, template_coords[0, :total])
    )
    return pose_stack.replace(coords=new_coords)


# --------------------------------------------------------------------- #
# Build pose-stack template once per sequence                            #
# --------------------------------------------------------------------- #


def build_pose_stack_template(
    sequence: str,
    target_sequence: Optional[str] = None,
) -> Any:
    """Build a PoseStack whose block types match ``sequence`` (and optional
    ``target_sequence`` concatenated after for binder-target complexes).

    This is host-side and runs once per design. The returned PoseStack
    carries:
      - per-atom atom37-to-flat mapping on ``._atom37_map``
      - per-atom chain id on ``._atom_chain_id``
    so the design loop can swap in new atom37 coords via ``jnp.take``
    (jit/grad-safe).

    Sequence is a string of 1-letter codes. Mapped to MET, ALA, ... with
    auto-assigned ``:nterm``/``:cterm`` variants. Any residue typed as
    ``X`` (placeholder glycine-like) is mapped to GLY.
    """
    import tmol_jax
    from tmol_jax.pose.packed_block_types import PackedBlockTypes
    from tmol_jax.pose.pose_stack_builder import PoseStackBuilder
    from tmol_jax.chemical.patched_chemdb import default_patched_chemdb

    one2three = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }

    seq_a = list(sequence.replace("X", "G"))
    seq_b = list(target_sequence.replace("X", "G")) if target_sequence else []

    # Use the full 92-type palette (includes :nterm / :cterm variants).
    pbt = PackedBlockTypes.default()

    # For each residue pick the right block type (base / :nterm / :cterm).
    def _resolve_bt_idx(aa3: str, is_nterm: bool, is_cterm: bool) -> int:
        if is_nterm and is_cterm:
            target = f"{aa3}:cterm:nterm"
        elif is_nterm:
            target = f"{aa3}:nterm"
        elif is_cterm:
            target = f"{aa3}:cterm"
        else:
            target = aa3
        for i, bt in enumerate(pbt.active_block_types):
            if bt.name == target:
                return i
        # Fall back to base type if variant missing.
        for i, bt in enumerate(pbt.active_block_types):
            if bt.name == aa3:
                return i
        raise ValueError(f"no block type for {target} or {aa3}")

    # Build per-pose block type indices.
    bt_inds = []
    chain_ids = []
    res_nums = []
    for ci, chain_seq in enumerate([seq_a, seq_b]):
        if not chain_seq:
            continue
        n = len(chain_seq)
        for r, aa in enumerate(chain_seq):
            aa3 = one2three.get(aa, "GLY")
            is_nterm = r == 0
            is_cterm = r == n - 1
            bt_inds.append(_resolve_bt_idx(aa3, is_nterm, is_cterm))
            chain_ids.append("AB"[ci])
            res_nums.append(r + 1)
    bt_inds_arr = np.asarray(bt_inds, dtype=np.int32)

    # Build ideal coords for each residue — lay out chain A along +z,
    # chain B shifted +50 Å in x to avoid clash.
    coords_list = []
    atom_offset = 0
    for idx_in_list, (aa3, ci) in enumerate(
        [(one2three.get(a, "GLY"), 0) for a in seq_a]
        + [(one2three.get(a, "GLY"), 1) for a in seq_b]
    ):
        bt_idx = int(bt_inds_arr[idx_in_list])
        bt = pbt.active_block_types[bt_idx]
        n_atoms_here = int(np.asarray(pbt.n_atoms)[bt_idx])
        ideal = np.asarray(bt.ideal_coords, dtype=np.float32)[:n_atoms_here]
        # Translate: chain B +50 Å in x; stack residues along z.
        x_offset = 50.0 if ci == 1 else 0.0
        z_offset = idx_in_list * 3.8
        if ci == 1:
            z_offset = (idx_in_list - len(seq_a)) * 3.8
        shifted = ideal + np.asarray([x_offset, 0.0, z_offset])
        coords_list.append(shifted)
    coords = np.concatenate(coords_list, axis=0)

    ps = PoseStackBuilder.from_block_types_and_coords(
        pbt=pbt,
        block_type_inds_per_pose=[bt_inds_arr],
        coords_per_pose=[coords],
        chain_id_per_pose=[chain_ids],
        res_num_per_pose=[res_nums],
    )

    # Build remapping tables.
    flat_from_atom37, is_mapped = _build_atom37_to_flat_mapping(ps)
    atom_chain_id = _build_residue_to_chain_mapping(ps)

    # Stash on the PoseStack. Using object.__setattr__ because PoseStack
    # may be frozen.
    object.__setattr__(ps, "_atom37_map", jnp.asarray(flat_from_atom37))
    object.__setattr__(ps, "_atom37_is_mapped", jnp.asarray(is_mapped))
    object.__setattr__(ps, "_atom_chain_id", jnp.asarray(atom_chain_id))
    return ps


# --------------------------------------------------------------------- #
# Core scoring helper                                                    #
# --------------------------------------------------------------------- #


def _score_breakdown_at_atom37(
    pose_stack: Any,
    sfxn: Any,
    state: Any,
    atom37_coords: Float[Array, "N 37 3"],
) -> Mapping:
    """Score the PoseStack at new atom37 coords; return
    ``{ScoreType: [n_poses]}`` dict."""
    coords = pose_stack.coords
    flat = atom37_coords.reshape(-1, 3)[pose_stack._atom37_map]
    max_n = coords.shape[1]
    total = flat.shape[0]
    new_coords = coords.at[0, :total].set(
        jnp.where(pose_stack._atom37_is_mapped[:, None], flat, coords[0, :total])
    )
    return sfxn.score_by_term(pose_stack, state=state, coords_override=new_coords)


# --------------------------------------------------------------------- #
# Loss terms                                                             #
# --------------------------------------------------------------------- #


class TmolInterfaceLoss(_BaseLoss):
    """Binding-affinity proxy — sum of cross-chain pair energies only.

    Computes beta_nov2016_cart on the full complex, AND on the complex
    with the binder translated 500 Å away. The difference
    ``E_bound - E_separated`` cancels intra-chain contributions and
    leaves only the interface. This matches Rosetta's
    ``InterfaceAnalyzer dG_separated`` metric, which is the standard
    in-silico correlate with experimental binding affinity.

    Per-term weights tunable via the ``term_weights`` constructor
    argument (dict from score-type name to float). Defaults are the
    beta2016 production weights; set an individual term to 0 to exclude
    it.

    Parameters
    ----------
    pose_stack_template
        Pre-built PoseStack for the full binder+target complex (via
        ``build_pose_stack_template``).
    term_weights
        Optional dict overriding default per-channel weights. Example:
        ``{"fa_elec": 2.0, "hbond": 5.0}`` boosts electrostatics and
        hbonds relative to LJ/LK.
    scale
        Global scaling factor applied after the weighted sum. Rosetta
        energies are tens-to-hundreds of kcal/mol; default ``1e-3``
        keeps the loss in the same order as mosaic's other terms.
    """

    # Everything other than ``scale`` is Python-stateful, not pytree
    # data; mark as static so eqx doesn't try to pytree-walk them.
    pose_stack: Any = eqx.field(static=True, repr=False)
    sfxn: Any = eqx.field(static=True, repr=False)
    state: Any = eqx.field(static=True, repr=False)
    scale: float
    translate_vec: Array
    binder_chain_id: int

    def __init__(
        self,
        pose_stack_template: Any,
        term_weights: Optional[Mapping[str, float]] = None,
        scale: float = 1e-3,
        binder_chain_id: int = 0,
    ):
        import tmol_jax
        from tmol_jax.score.score_types import ScoreType
        from tmol_jax.score.score_function import ScoreFunction

        # Default: beta2016 weights.
        from tmol_jax.score import beta2016_score_function

        sfxn = beta2016_score_function()
        if term_weights is not None:
            for name, w in term_weights.items():
                st = getattr(ScoreType, name, None)
                if st is None:
                    raise ValueError(f"Unknown score type: {name}")
                sfxn.set_weight(st, float(w))

        self.pose_stack = pose_stack_template
        self.sfxn = sfxn
        self.state = sfxn.setup_for_pose_stack(pose_stack_template)
        self.scale = scale
        self.binder_chain_id = binder_chain_id
        # 500 Å translation along +x to separate chains
        self.translate_vec = jnp.asarray([500.0, 0.0, 0.0], dtype=jnp.float32)

    def _separated_coords(self, flat_coords: Array) -> Array:
        """Translate binder atoms by ``translate_vec``."""
        atom_chain = self.pose_stack._atom_chain_id
        is_binder = atom_chain == self.binder_chain_id
        shift = jnp.where(is_binder[:, None], self.translate_vec, 0.0)
        return flat_coords + shift

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: Any,
        key,
    ):
        # Pull atom37 coords from the model output.
        atom37_coords = output.structure_coordinates  # [N, 37, 3]

        # Bound-complex scoring.
        bound_per_term = _score_breakdown_at_atom37(
            self.pose_stack, self.sfxn, self.state, atom37_coords
        )
        bound_total = sum(
            jnp.asarray(v).reshape(-1)[0] for v in bound_per_term.values()
        )

        # Separated scoring — translate binder 500 Å away.
        coords_bound = self.pose_stack.coords
        flat = atom37_coords.reshape(-1, 3)[self.pose_stack._atom37_map]
        total = flat.shape[0]
        flat_complete = coords_bound[0, :total].at[:].set(
            jnp.where(
                self.pose_stack._atom37_is_mapped[:, None], flat, coords_bound[0, :total]
            )
        )
        flat_separated = self._separated_coords(flat_complete)
        coords_sep = coords_bound.at[0, :total].set(flat_separated)
        sep_per_term = self.sfxn.score_by_term(
            self.pose_stack, state=self.state, coords_override=coords_sep
        )
        sep_total = sum(jnp.asarray(v).reshape(-1)[0] for v in sep_per_term.values())

        dg_interface = bound_total - sep_total

        return self.scale * dg_interface, {
            "dg_interface_kcal_mol": dg_interface,
            "bound_total": bound_total,
            "separated_total": sep_total,
        }


class TmolBackboneLoss(_BaseLoss):
    """Backbone-only physics — cart_bonded + rama + omega. Design time.

    Scores depend only on N/CA/C/O positions, so this works with the
    hallucinator's X-residue binder during the PSSM-optimization phase.
    """

    # Everything other than ``scale`` is Python-stateful, not pytree
    # data; mark as static so eqx doesn't try to pytree-walk them.
    pose_stack: Any = eqx.field(static=True, repr=False)
    sfxn: Any = eqx.field(static=True, repr=False)
    state: Any = eqx.field(static=True, repr=False)
    scale: float

    def __init__(self, pose_stack_template: Any, scale: float = 1e-2):
        import tmol_jax
        from tmol_jax.score.score_types import ScoreType
        from tmol_jax.score.score_function import ScoreFunction

        sfxn = ScoreFunction(tmol_jax.ParameterDatabase.get_default())
        # beta2016 weights on backbone-only terms
        for st_name, w in (
            ("cart_lengths", 0.5),
            ("cart_angles", 0.5),
            ("cart_torsions", 0.5),
            ("cart_impropers", 0.5),
            ("cart_hxltorsions", 0.5),
            ("rama", 0.50),
            ("omega", 0.48),
        ):
            sfxn.set_weight(getattr(ScoreType, st_name), w)

        self.pose_stack = pose_stack_template
        self.sfxn = sfxn
        self.state = sfxn.setup_for_pose_stack(pose_stack_template)
        self.scale = scale

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: Any,
        key,
    ):
        # Use backbone_coordinates if available (shape [N, 4, 3]), else
        # atom37. backbone_coordinates is N/CA/C/O in that order.
        bb = getattr(output, "backbone_coordinates", None)
        if bb is not None:
            # Expand to atom37 (use ideal coords for everything else).
            n = bb.shape[0]
            atom37 = jnp.zeros((n, 37, 3), dtype=bb.dtype)
            # atom37 slot 0=N, 1=CA, 2=C, 4=O (per openfold atom_types)
            atom37 = atom37.at[:, 0].set(bb[:, 0])
            atom37 = atom37.at[:, 1].set(bb[:, 1])
            atom37 = atom37.at[:, 2].set(bb[:, 2])
            atom37 = atom37.at[:, 4].set(bb[:, 3])
        else:
            atom37 = output.structure_coordinates

        per_term = _score_breakdown_at_atom37(
            self.pose_stack, self.sfxn, self.state, atom37
        )
        total = sum(jnp.asarray(v).reshape(-1)[0] for v in per_term.values())
        return self.scale * total, {"tmol_backbone_energy": total}


class TmolClashFilter(_BaseLoss):
    """fa_ljrep-only term — cheap clash filter with strong gradients."""

    # Everything other than ``scale`` is Python-stateful, not pytree
    # data; mark as static so eqx doesn't try to pytree-walk them.
    pose_stack: Any = eqx.field(static=True, repr=False)
    sfxn: Any = eqx.field(static=True, repr=False)
    state: Any = eqx.field(static=True, repr=False)
    scale: float

    def __init__(self, pose_stack_template: Any, scale: float = 1e-2):
        import tmol_jax
        from tmol_jax.score.score_types import ScoreType
        from tmol_jax.score.score_function import ScoreFunction

        sfxn = ScoreFunction(tmol_jax.ParameterDatabase.get_default())
        sfxn.set_weight(ScoreType.fa_ljrep, 1.0)

        self.pose_stack = pose_stack_template
        self.sfxn = sfxn
        self.state = sfxn.setup_for_pose_stack(pose_stack_template)
        self.scale = scale

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: Any,
        key,
    ):
        atom37 = output.structure_coordinates
        per_term = _score_breakdown_at_atom37(
            self.pose_stack, self.sfxn, self.state, atom37
        )
        total = sum(jnp.asarray(v).reshape(-1)[0] for v in per_term.values())
        return self.scale * total, {"tmol_ljrep": total}


# --------------------------------------------------------------------- #
# Quick self-test                                                        #
# --------------------------------------------------------------------- #


_USAGE_HINT = """
# in hallucinate.py after building folder/features:
from examples.integration_mosaic_hallucinator import (
    build_pose_stack_template,
    TmolInterfaceLoss, TmolBackboneLoss, TmolClashFilter,
)

ps_tpl = build_pose_stack_template(
    sequence="X" * BINDER_LENGTH,
    target_sequence=TARGET_SEQUENCE,
)

# Binding affinity proxy — tune per-term weights to prioritize
# shape complementarity / electrostatics / H-bonds.
tmol_interface = TmolInterfaceLoss(
    ps_tpl,
    term_weights={"fa_elec": 2.0, "hbond": 5.0},  # prioritize electrostatics + hbonds
    scale=1e-3,
)
tmol_clash = TmolClashFilter(ps_tpl)

sp_loss = (
    sp.BinderTargetContact()
    + sp.WithinBinderContact()
    + 10.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.001), bias=bias)
    + 0.05 * sp.TargetBinderPAE()
    + 0.05 * sp.BinderTargetPAE()
    + 0.025 * sp.IPTMLoss()
    + 0.4 * sp.WithinBinderPAE()
    + 0.025 * sp.pTMEnergy()
    + 0.1 * sp.PLDDTLoss()
    + 1.0 * tmol_interface    # NEW: binding-affinity proxy
    + 0.01 * tmol_clash       # NEW: clash filter
)
"""


def _sanity_test_with_real_1ubq():
    """Run the losses on a real PDB to show realistic numbers.

    Uses 1ubq (single-chain; we synthetically split it into "binder"
    (1-40) + "target" (41-76) to exercise the interface loss).
    """
    import tmol_jax

    ps = tmol_jax.pose_stack_from_pdb("tmol_jax/tests/data/1ubq.pdb")

    # 1ubq is a single peptide chain; for this sanity test we
    # *artificially* split it into two "pretend chains" by cutting the
    # inter_residue_connections between residues 40 and 41. In real
    # binder+target complexes the two chains are independent at the PDB
    # level, so no cutting is needed — this is a test artefact only.
    split_res = 40
    irc = np.asarray(ps.inter_residue_connections)  # [P, B, C, 2]
    irc_cut = irc.copy()
    # Cut the peptide bond going C-term from residue split_res-1 to
    # split_res (connection slot 1 = "down"), and symmetrically the
    # N-term connection of residue split_res (slot 0 = "up").
    irc_cut[0, split_res - 1, 1] = -1
    irc_cut[0, split_res, 0] = -1
    ibs = np.asarray(ps.inter_block_bondsep)  # [P, B, B, C, C]
    ibs_cut = ibs.copy()
    ibs_cut[0, split_res - 1, split_res, :, :] = 6  # "far apart"
    ibs_cut[0, split_res, split_res - 1, :, :] = 6
    ps = ps.replace(
        inter_residue_connections=jnp.asarray(irc_cut),
        inter_block_bondsep=jnp.asarray(ibs_cut),
    )

    # Build atom37 from the real PoseStack coords.
    flat_from_atom37, is_mapped = _build_atom37_to_flat_mapping(ps)
    object.__setattr__(ps, "_atom37_map", jnp.asarray(flat_from_atom37))
    object.__setattr__(ps, "_atom37_is_mapped", jnp.asarray(is_mapped))

    # Force chain split: first 40 residues = "binder" (chain 0), rest = "target" (chain 1).
    pbt = ps.packed_block_types
    bti = np.asarray(ps.block_type_ind[0])
    offsets = np.asarray(ps.block_coord_offset[0])
    n_per_bt = np.asarray(pbt.n_atoms)
    split_res = 40
    atom_chain = np.zeros(ps.coords.shape[1], dtype=np.int32)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(n_per_bt[bt])
        atom_chain[int(offsets[r]) : int(offsets[r]) + n] = 1 if r >= split_res else 0
    object.__setattr__(ps, "_atom_chain_id", jnp.asarray(atom_chain))

    # Build the atom37 representation of 1ubq's actual coords.
    n_res = int(ps.n_res_per_pose[0])
    atom37 = np.zeros((n_res, 37, 3), dtype=np.float32)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(n_per_bt[bt])
        start = int(offsets[r])
        names = pbt.atom_names[bt]
        for local in range(n):
            if names[local] in _ATOM37_NAMES:
                slot = _ATOM37_NAMES.index(names[local])
                atom37[r, slot] = np.asarray(ps.coords[0, start + local])
    atom37 = jnp.asarray(atom37)

    class _StructOut(eqx.Module):
        structure_coordinates: Array
        backbone_coordinates: Array
        full_sequence: Array

    output = _StructOut(
        structure_coordinates=atom37,
        backbone_coordinates=atom37[:, [0, 1, 2, 4]],
        full_sequence=jnp.zeros((n_res, 20), dtype=jnp.float32).at[:, 0].set(1.0),
    )

    print("\n=== Sanity test on real 1ubq (split into binder+target) ===")
    print(f"  binder = residues 1-{split_res}, target = residues {split_res+1}-{n_res}")

    # Interface loss with default beta2016 weights.
    iface = TmolInterfaceLoss(ps, scale=1.0, binder_chain_id=0)
    v, aux = iface(output.full_sequence, output, key=jax.random.PRNGKey(0))
    print(f"  dG_interface  = {float(v):+8.2f} kcal/mol")
    print(f"    E_bound     = {float(aux['bound_total']):+10.2f}")
    print(f"    E_separated = {float(aux['separated_total']):+10.2f}")

    # Gradient of dG w.r.t. atom37 coords.
    def loss_of_atom37(c):
        out = _StructOut(
            structure_coordinates=c,
            backbone_coordinates=c[:, [0, 1, 2, 4]],
            full_sequence=output.full_sequence,
        )
        val, _ = iface(output.full_sequence, out, key=jax.random.PRNGKey(0))
        return val

    g = jax.grad(loss_of_atom37)(atom37)
    print(f"  grad shape: {g.shape}, finite: {bool(jnp.all(jnp.isfinite(g)))}, max|.|={float(jnp.max(jnp.abs(g))):.3f}")

    # Per-term weight sweep — show the knob.
    print("\n  Per-term weight tuning on the interface loss:")
    for label, weights in [
        ("default beta2016",         None),
        ("boost electrostatics",     {"fa_elec": 5.0}),
        ("boost H-bonds",            {"hbond": 5.0}),
        ("favour shape (hard wall)", {"fa_ljrep": 2.0, "fa_ljatr": 2.0}),
        ("ignore LJ, only polars",   {"fa_ljatr": 0.0, "fa_ljrep": 0.0, "fa_lk": 0.0}),
    ]:
        iface_v = TmolInterfaceLoss(ps, term_weights=weights, scale=1.0, binder_chain_id=0)
        vv, _ = iface_v(output.full_sequence, output, key=jax.random.PRNGKey(0))
        print(f"    {label:30s}  dG = {float(vv):+8.2f} kcal/mol")


if __name__ == "__main__":
    # --- End-to-end self-test with a synthetic StructureModelOutput ---
    #
    # Verifies:
    #   1. build_pose_stack_template works for a binder+target sequence
    #   2. TmolInterfaceLoss matches the LossTerm API
    #   3. jax.grad flows through to the atom37 coordinates (and thus
    #      back to the PSSM via Boltz-2 in the real pipeline)

    import tmol_jax

    BINDER = "MAEQKLI"       # 7-residue toy binder
    TARGET = "GYRDTSN"       # 7-residue toy target

    print(f"Building pose-stack template for {BINDER} + {TARGET}...")
    ps = build_pose_stack_template(sequence=BINDER, target_sequence=TARGET)
    print(f"  n_res = {int(ps.n_res_per_pose[0])}")
    print(f"  chains found: {int(ps._atom_chain_id.max() + 1)}")

    # Build a synthetic StructureModelOutput shape.
    n_res = int(ps.n_res_per_pose[0])

    class _StructOut(eqx.Module):
        structure_coordinates: Array
        backbone_coordinates: Array
        full_sequence: Array

    # Populate atom37 with the template coords (backbone laid out along Y).
    pbt = ps.packed_block_types
    bti = np.asarray(ps.block_type_ind[0])
    offsets = np.asarray(ps.block_coord_offset[0])
    atom37 = np.zeros((n_res, 37, 3), dtype=np.float32)
    for r, bt in enumerate(bti):
        if bt < 0:
            continue
        n = int(pbt.n_atoms[bt])
        start = int(offsets[r])
        names = pbt.atom_names[bt]
        for local in range(n):
            if names[local] in _ATOM37_NAMES:
                slot = _ATOM37_NAMES.index(names[local])
                atom37[r, slot] = np.asarray(ps.coords[0, start + local])

    bb = jnp.asarray(atom37[:, [0, 1, 2, 4]])  # N/CA/C/O
    output = _StructOut(
        structure_coordinates=jnp.asarray(atom37),
        backbone_coordinates=bb,
        full_sequence=jnp.zeros((n_res, 20), dtype=jnp.float32).at[:, 0].set(1.0),
    )

    # --- Test TmolInterfaceLoss ---
    print("\nTmolInterfaceLoss:")
    iface = TmolInterfaceLoss(
        ps,
        term_weights={"fa_elec": 2.0, "hbond": 5.0},
        scale=1.0,  # unweighted so we see raw kcal/mol
        binder_chain_id=0,
    )
    loss_val, aux = iface(
        output.full_sequence, output, key=jax.random.PRNGKey(0)
    )
    print(f"  dG_interface ~= {float(loss_val):+.3f} kcal/mol")
    print(f"  bound total   = {float(aux['bound_total']):+.3f}")
    print(f"  separated tot = {float(aux['separated_total']):+.3f}")

    # Gradient through atom37 coords (what the real pipeline uses).
    def loss_of_coords(atom37_in):
        out = _StructOut(
            structure_coordinates=atom37_in,
            backbone_coordinates=atom37_in[:, [0, 1, 2, 4]],
            full_sequence=output.full_sequence,
        )
        v, _ = iface(output.full_sequence, out, key=jax.random.PRNGKey(0))
        return v

    g = jax.grad(loss_of_coords)(output.structure_coordinates)
    print(f"  grad shape    = {g.shape}")
    print(f"  grad finite   = {bool(jnp.all(jnp.isfinite(g)))}")
    print(f"  grad max|.|   = {float(jnp.nanmax(jnp.abs(g))):.3f}")

    # --- Test composition ---
    print("\nCombined loss (interface + backbone + clash):")
    bb_loss = TmolBackboneLoss(ps, scale=1.0)
    clash_loss = TmolClashFilter(ps, scale=1.0)

    combined = 1.0 * iface + 0.01 * bb_loss + 0.01 * clash_loss
    loss_val, aux_list = combined(
        output.full_sequence, output, key=jax.random.PRNGKey(0)
    )
    print(f"  combined loss = {float(loss_val):+.3f}")
    print(f"  n aux entries = {len(aux_list)}")

    # --- Test per-term weight tuning effect on binding-affinity proxy ---
    print("\nTuning per-term weights on the interface loss:")
    for weights in [
        {"fa_elec": 1.0, "hbond": 1.0},                 # baseline
        {"fa_elec": 5.0, "hbond": 1.0},                 # boost elec
        {"fa_elec": 1.0, "hbond": 10.0},                # boost hbond
        {"fa_ljatr": 2.0, "fa_ljrep": 0.5},             # emphasize shape, tolerate clashes
    ]:
        iface_v = TmolInterfaceLoss(ps, term_weights=weights, scale=1.0)
        v, _ = iface_v(output.full_sequence, output, key=jax.random.PRNGKey(0))
        w_str = ", ".join(f"{k}={v:g}" for k, v in weights.items())
        print(f"  [{w_str}]  dG = {float(v):+7.3f}")

    print("\nAll loss terms are drop-in compatible with mosaic.LossTerm.")
    print("See _USAGE_HINT at the bottom of this file for the hallucinate.py patch.")

    # Run the real-1ubq sanity test last so we get a believable number.
    _sanity_test_with_real_1ubq()
