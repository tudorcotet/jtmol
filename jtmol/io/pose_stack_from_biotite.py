"""Build a :class:`PoseStack` from a :mod:`biotite` structure.

Upstream reference: ``tmol.io.pose_stack_from_biotite``. The torch
version carries a fairly elaborate canonical-form-based pipeline that
includes rebuilding missing side chains via the Dunbrack rotamer
library; in the JAX port we only implement the minimal "iterate biotite
residues, fill canonical coords, pack into a PoseStack" path. Missing
side-chain reconstruction is intentionally out of scope -- callers that
need it should fall back to the torch port.

The primary entry point is :func:`pose_stack_from_biotite`. Helper
getters mirror the upstream tmol naming convention:

* :func:`canonical_ordering_for_biotite` -- the default canonical
  ordering (biotite residues are string-keyed so there is no bespoke
  mapping to build).
* :func:`packed_block_types_for_biotite` -- the default
  :class:`PackedBlockTypes`.
* :func:`canonical_form_from_biotite` -- biotite ``AtomArray`` ->
  canonical-form dict.

Pure Python + numpy. No torch, no JAX required (we do call into the
shared ``jtmol.pose.PoseStackBuilder`` which produces JAX arrays).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import attr
import numpy as np


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# BiotitePoseBuildContext (port of upstream PR #361 scaffold)            #
# --------------------------------------------------------------------- #


@attr.s(auto_attribs=True, frozen=True, slots=True)
class BiotitePoseBuildContext:
    """Frozen container threaded through the biotite -> PoseStack pipeline.

    Mirrors the upstream ``tmol.io.pose_stack_from_biotite.BiotitePoseBuildContext``
    dataclass introduced in PR #361. Holds the (possibly ligand-extended)
    chemical database + canonical-ordering + packed-block-types triple
    plus the structured canonical-form dict that
    :func:`pose_stack_from_canonical_form` consumes.

    Fields
    ------
    chem_db:
        :class:`jtmol.database.chemical.ChemicalDatabase` — extended
        with ligand atom_types + residues if ``ligand_params_files``
        was supplied to :func:`build_context_from_biotite`.
    canonical_ordering:
        :class:`jtmol.io.canonical_ordering.CanonicalOrdering`. The
        scaffold uses the default 20-AA ordering even when ligands are
        supplied — extending the ordering to include the ligand restype
        is deferred to a follow-up (see ``jtmol/ligand/__init__.py``).
    packed_block_types:
        :class:`jtmol.pose.packed_block_types.PackedBlockTypes`.
        Same caveat as ``canonical_ordering``.
    canonical_form:
        The canonical-form dict returned by
        :func:`canonical_form_from_biotite` (numpy arrays keyed by
        ``chain_id`` / ``res_types`` / ``coords`` / etc.).
    ligand_params_files:
        Tuple of :class:`jtmol.ligand.params_file.LigandParamsFile`
        objects parsed from ``ligand_params_files``, or ``()``. Held so
        downstream callers can pick up per-atom partial charges and
        bonded params without re-parsing.
    """

    chem_db: Any
    canonical_ordering: Any
    packed_block_types: Any
    canonical_form: Dict[str, Any]
    ligand_params_files: Tuple[Any, ...] = ()


# --------------------------------------------------------------------- #
# Biotite import guard                                                   #
# --------------------------------------------------------------------- #


def _try_import_biotite():
    """Return ``biotite.structure`` or ``None`` if biotite isn't installed."""
    try:  # pragma: no cover - import guard
        import biotite.structure as _biotite_structure  # noqa: F401

        return _biotite_structure
    except Exception:  # pragma: no cover
        return None


def _require_biotite():
    mod = _try_import_biotite()
    if mod is None:
        raise ImportError(
            "pose_stack_from_biotite requires the 'biotite' package; "
            "install it with `pip install biotite` or `pip install 'tmol-jax[biotite]'`."
        )
    return mod


# --------------------------------------------------------------------- #
# Canonical-ordering / PackedBlockTypes helpers                          #
# --------------------------------------------------------------------- #


_CACHE: Dict[str, Any] = {}


def canonical_ordering_for_biotite():
    """Return the default :class:`CanonicalOrdering` for biotite inputs.

    Biotite stores residue / atom names as strings, so we do not need a
    bespoke integer-to-integer mapping the way OpenFold and RoseTTAFold2
    do; the default canonical ordering already exposes the
    string-indexed ``restypes_atom_index_mapping`` that downstream code
    looks up atom indices through.
    """
    if "co" in _CACHE:
        return _CACHE["co"]
    from jtmol.io.canonical_ordering import default_canonical_ordering

    co = default_canonical_ordering()
    _CACHE["co"] = co
    return co


def packed_block_types_for_biotite():
    """Return the default :class:`PackedBlockTypes` for biotite inputs."""
    if "pbt" in _CACHE:
        return _CACHE["pbt"]
    from jtmol.io.canonical_ordering import default_packed_block_types

    pbt = default_packed_block_types()
    _CACHE["pbt"] = pbt
    return pbt


# --------------------------------------------------------------------- #
# Per-residue iteration                                                  #
# --------------------------------------------------------------------- #


def _iter_biotite_residues(atoms: Any) -> List[Tuple[int, int]]:
    """Return ``(start_idx, end_idx)`` atom-array slices, one per residue.

    Follows the same residue-boundary detection biotite's own
    ``get_residue_starts`` performs, but without requiring it to be
    importable at module-load time (the top-level biotite import happens
    inside this helper via :func:`_try_import_biotite`).
    """
    mod = _require_biotite()
    starts = mod.get_residue_starts(atoms)
    n_atoms = int(atoms.array_length())
    ends = np.append(starts[1:], n_atoms)
    return list(zip(starts.tolist(), ends.tolist()))


def _lookup_canonical_atom_index(co: Any, res_name: str, atom_name: str) -> int:
    """Return the canonical atom index for ``(res_name, atom_name)`` or -1."""
    mapping = getattr(co, "restypes_atom_index_mapping", None)
    if mapping is None:
        return -1
    res_mapping = mapping.get(res_name, None)
    if res_mapping is None:
        return -1
    idx = res_mapping.get(atom_name, -1)
    try:
        return int(idx)
    except (TypeError, ValueError):
        return -1


# --------------------------------------------------------------------- #
# CanonicalForm construction                                             #
# --------------------------------------------------------------------- #


def canonical_form_from_biotite(
    biotite_structure: Any,
    *,
    co: Optional[Any] = None,
    missing_density_distance_threshold: float = 2.4,
) -> Dict[str, Any]:
    """Convert a biotite ``AtomArray`` / ``AtomArrayStack`` to a canonical-form dict.

    The returned dict mirrors the upstream :class:`tmol.io.canonical_form.CanonicalForm`
    field layout, but uses plain numpy arrays since the JAX port of
    :class:`CanonicalForm` is itself numpy-backed:

    * ``chain_id``        -- ``int32[n_poses, n_res]``
    * ``res_types``       -- ``int32[n_poses, n_res]``   (index into
      ``co.restype_io_equiv_classes``)
    * ``coords``          -- ``float32[n_poses, n_res, max_n_canonical_atoms, 3]``
    * ``chain_labels``    -- ``object[n_poses, n_res]``
    * ``res_labels``      -- ``int32[n_poses, n_res]``
    * ``residue_insertion_codes`` -- ``object[n_poses, n_res]``
    * ``atom_occupancy``  -- ``float32[n_poses, n_res, max_n_canonical_atoms]`` or None
    * ``atom_b_factor``   -- ``float32[n_poses, n_res, max_n_canonical_atoms]`` or None
    * ``res_not_connected`` -- ``bool[n_poses, n_res, 2]``
    """
    mod = _require_biotite()
    if co is None:
        co = canonical_ordering_for_biotite()

    # Normalise AtomArrayStack -> single stack with shape [n_poses, ...]
    if isinstance(biotite_structure, mod.AtomArrayStack):
        n_poses = int(biotite_structure.coord.shape[0])
        template = biotite_structure[0]
    else:
        n_poses = 1
        template = biotite_structure

    known_residues = set(getattr(co, "restype_io_equiv_classes", ()) or ())
    restype_to_index = {
        name: i for i, name in enumerate(getattr(co, "restype_io_equiv_classes", ()) or ())
    }
    max_n_atoms = int(getattr(co, "max_n_canonical_atoms", 0) or 0)

    res_slices = _iter_biotite_residues(template)

    # Filter out residues whose 3lc isn't known to the canonical ordering.
    valid_residues: List[Tuple[int, int, str]] = []
    for start, end in res_slices:
        res_name = str(template.res_name[start])
        if res_name == "HOH":
            continue
        if res_name not in known_residues:
            logger.warning("Unrecognised residue name %r; skipping", res_name)
            continue
        valid_residues.append((start, end, res_name))

    n_res = len(valid_residues)

    # Allocate outputs.
    coords = np.full(
        (n_poses, n_res, max(max_n_atoms, 1), 3), np.nan, dtype=np.float32
    )
    res_types = np.zeros((n_poses, n_res), dtype=np.int32)
    chain_id_int = np.zeros((n_poses, n_res), dtype=np.int32)
    chain_labels = np.empty((n_poses, n_res), dtype=object)
    res_labels = np.zeros((n_poses, n_res), dtype=np.int32)
    ins_codes = np.empty((n_poses, n_res), dtype=object)
    chain_labels[:] = ""
    ins_codes[:] = ""

    # Get biotite arrays in a shape-uniform way.
    if isinstance(biotite_structure, mod.AtomArrayStack):
        all_coords = np.asarray(biotite_structure.coord, dtype=np.float32)
        # Annotations come from the template (they're identical across poses).
    else:
        all_coords = np.asarray(template.coord, dtype=np.float32)[np.newaxis, :]
    atom_names_arr = np.asarray(template.atom_name)
    chain_id_arr = np.asarray(template.chain_id)
    res_id_arr = np.asarray(template.res_id)
    ins_code_arr = np.asarray(getattr(template, "ins_code", np.array([""] * len(template))))
    b_factor_arr = np.asarray(
        getattr(template, "b_factor", np.zeros(len(template), dtype=np.float32)),
        dtype=np.float32,
    )
    occ_arr = np.asarray(
        getattr(template, "occupancy", np.ones(len(template), dtype=np.float32)),
        dtype=np.float32,
    )
    has_bfactor = hasattr(template, "b_factor")
    has_occupancy = hasattr(template, "occupancy")

    b_factors = (
        np.zeros((n_poses, n_res, max(max_n_atoms, 1)), dtype=np.float32)
        if has_bfactor
        else None
    )
    occupancy = (
        np.ones((n_poses, n_res, max(max_n_atoms, 1)), dtype=np.float32)
        if has_occupancy
        else None
    )

    chain_label_to_int: Dict[str, int] = {}
    for i_res, (start, end, res_name) in enumerate(valid_residues):
        res_types[:, i_res] = restype_to_index.get(res_name, 0)
        chain_str = str(chain_id_arr[start])
        chain_labels[:, i_res] = chain_str
        if chain_str not in chain_label_to_int:
            chain_label_to_int[chain_str] = len(chain_label_to_int)
        chain_id_int[:, i_res] = chain_label_to_int[chain_str]
        try:
            res_labels[:, i_res] = int(res_id_arr[start])
        except (TypeError, ValueError):
            res_labels[:, i_res] = 0
        ins_codes[:, i_res] = str(ins_code_arr[start]) if start < len(ins_code_arr) else ""

        # Map each atom of the residue into the canonical slot for res_name.
        for k in range(start, end):
            at_name = str(atom_names_arr[k])
            canon_idx = _lookup_canonical_atom_index(co, res_name, at_name)
            if canon_idx < 0 or canon_idx >= max_n_atoms:
                continue
            for p in range(n_poses):
                coords[p, i_res, canon_idx] = all_coords[p, k]
            if b_factors is not None:
                b_factors[:, i_res, canon_idx] = float(b_factor_arr[k])
            if occupancy is not None:
                occupancy[:, i_res, canon_idx] = float(occ_arr[k])

    # res_not_connected based on same-chain consecutive residues.
    res_not_connected = np.zeros((n_poses, n_res, 2), dtype=bool)
    if n_res > 0:
        res_not_connected[:, 0, 0] = True
        res_not_connected[:, -1, 1] = True
        for i in range(n_res - 1):
            same_chain = chain_id_int[0, i] == chain_id_int[0, i + 1]
            if not same_chain:
                res_not_connected[:, i, 1] = True
                res_not_connected[:, i + 1, 0] = True

    # Optional geometry-based missing-density break.
    if missing_density_distance_threshold > 0 and n_res > 1:
        _break_connections_for_missing_density(
            res_not_connected,
            chain_id_int,
            coords,
            missing_density_distance_threshold,
        )

    return {
        "chain_id": chain_id_int,
        "res_types": res_types,
        "coords": coords,
        "chain_labels": chain_labels,
        "res_labels": res_labels,
        "residue_insertion_codes": ins_codes,
        "atom_occupancy": occupancy,
        "atom_b_factor": b_factors,
        "disulfides": None,
        "res_not_connected": res_not_connected,
    }


def _break_connections_for_missing_density(
    res_not_connected: np.ndarray,
    chain_id_int: np.ndarray,
    coords: np.ndarray,
    threshold: float,
) -> None:
    """In-place: break adjacent-residue connections that are too far apart.

    For each consecutive (i, i+1) pair on the same chain still marked as
    connected, take the minimum inter-atom distance across poses. If it
    exceeds ``threshold`` Å, break the connection.
    """
    n_res = int(res_not_connected.shape[1])
    for i in range(n_res - 1):
        if res_not_connected[0, i, 1] or res_not_connected[0, i + 1, 0]:
            continue
        if chain_id_int[0, i] != chain_id_int[0, i + 1]:
            continue
        min_dist = np.inf
        for p in range(int(coords.shape[0])):
            c_i = coords[p, i]
            c_j = coords[p, i + 1]
            valid_i = ~np.isnan(c_i[:, 0])
            valid_j = ~np.isnan(c_j[:, 0])
            if not valid_i.any() or not valid_j.any():
                continue
            ci = c_i[valid_i]
            cj = c_j[valid_j]
            diffs = ci[:, None, :] - cj[None, :, :]
            pose_min = float(np.sqrt((diffs ** 2).sum(axis=-1)).min())
            if pose_min < min_dist:
                min_dist = pose_min
            if min_dist <= threshold:
                break
        if min_dist > threshold:
            res_not_connected[:, i, 1] = True
            res_not_connected[:, i + 1, 0] = True


# --------------------------------------------------------------------- #
# End-to-end builder                                                     #
# --------------------------------------------------------------------- #


def _filter_supported_atoms_and_connectivity(
    atoms: Any,
    co: Any,
    extra_known_residues: Sequence[str] = (),
) -> Any:
    """Drop residues whose 3-letter code is not in ``co`` or in
    ``extra_known_residues``.

    Scaffold port of the upstream helper of the same name. The upstream
    version also prunes residues missing mainchain atoms and rewrites
    inter-residue connectivity. The jtmol scaffold only does the 3lc
    filter (which is what :func:`canonical_form_from_biotite` already
    does implicitly via :func:`_lookup_canonical_atom_index`). It is
    exposed as a standalone helper so the ligand path can ask "is this
    PDB after-filtering still non-empty?" without re-running the full
    canonical-form construction.

    Mainchain-completeness pruning is **deferred**: it currently lives
    inside ``canonical_form_from_biotite`` as an "atom drop" rather than
    "residue drop". See ``# ponytail`` note below.
    """
    mod = _require_biotite()

    known: set = set(getattr(co, "restype_io_equiv_classes", ()) or ())
    known.update(extra_known_residues)

    template = atoms[0] if isinstance(atoms, mod.AtomArrayStack) else atoms
    res_names_arr = np.asarray(template.res_name)
    keep = np.array(
        [str(r) in known and str(r) != "HOH" for r in res_names_arr],
        dtype=bool,
    )
    # ponytail: residue-level mainchain-completeness pruning deferred;
    # upstream's version inspects each residue's mainchain set and drops
    # the whole row if any mainchain atom is missing. Add when a
    # parity-regression PDB demonstrates the need.
    return atoms[..., keep]


def _derived_types_for_param_db(
    chem_db: Any,
    ligand_params_files: Sequence[Any],
    *,
    strict_atom_types: bool = False,
) -> Tuple[Any, Tuple[Any, ...], Tuple[str, ...]]:
    """Return ``(extended_chem_db, parsed_params_files, ligand_3lcs)``.

    Pure scaffold: delegates to
    :func:`jtmol.ligand.params_io.extend_db_from_yaml` and surfaces
    the per-residue ``name3`` values so callers can extend their
    canonical-ordering palettes.

    Marked with a ``ponytail`` comment because the heavy version
    upstream uses RDKit + SMARTS atom typing; this scaffold only
    accepts pre-baked YAML and leaves the RDKit path explicitly
    unsupported.
    """
    # ponytail: RDKit-driven atom typing path skipped; see ligand/__init__.py
    # for the deferred-work note.
    from jtmol.ligand.params_io import extend_db_from_yaml

    extended, parsed = extend_db_from_yaml(
        chem_db, ligand_params_files, strict_atom_types=strict_atom_types
    )
    ligand_3lcs: List[str] = []
    seen: set = set()
    for lp in parsed:
        for res in lp.residues:
            if res.name3 not in seen:
                seen.add(res.name3)
                ligand_3lcs.append(res.name3)
    return extended, parsed, tuple(ligand_3lcs)


def _extended_co_pbt_for_ligands(
    parsed_files: Sequence[Any],
) -> Tuple[Any, Any]:
    """Return ``(canonical_ordering, packed_block_types)`` extended with
    the ligand restypes parsed from ``parsed_files``.

    The protein half reproduces the default 20-AA (+ termini variant)
    palette exactly; the ligand restypes are appended on top. Because the
    generic-potential ligand atom types now live in ``chemical.yaml``
    (upstream PR #361), the default patched chem-db already resolves them,
    so building the ligand ``RefinedResidueType`` (icoors -> ideal coords)
    and packing it needs no extra atom-type bookkeeping.
    """
    from jtmol.chemical.constants import canonical_aa_3letter
    from jtmol.chemical.patched_chemdb import default_patched_chemdb
    from jtmol.chemical.restypes import _refined_from_raw
    from jtmol.io.canonical_ordering import CanonicalOrdering
    from jtmol.pose.packed_block_types import PackedBlockTypes

    ligand_raws = [res for lp in parsed_files for res in lp.residues]

    # Canonical ordering: default protein residues + ligand residues.
    patched = default_patched_chemdb()
    canonical_set = set(canonical_aa_3letter)
    protein_res = [
        r for r in patched.residues
        if getattr(r, "name3", None) in canonical_set
    ]

    class _MiniDb:
        __slots__ = ("residues",)

    mini = _MiniDb()
    mini.residues = tuple(protein_res) + tuple(ligand_raws)
    co = CanonicalOrdering.from_chemdb(mini)

    # PackedBlockTypes: default 20-AA + variant palette, then ligands.
    base_pbt = PackedBlockTypes.default()
    ligand_refined = [_refined_from_raw(r) for r in ligand_raws]

    # Carry per-atom partial charges (parsed from the ``.tmol`` elec
    # block) on each ligand RefinedResidueType so the fa_elec term picks
    # them up via ``_charges_for_block_type``. Keyed {res_name: {atom: q}}.
    ligand_charges: Dict[str, Dict[str, float]] = {}
    for lp in parsed_files:
        for (res_name, atom_name), q in getattr(lp, "atom_charges", {}).items():
            ligand_charges.setdefault(res_name, {})[atom_name] = float(q)
    for rr in ligand_refined:
        q = ligand_charges.get(rr.name) or ligand_charges.get(rr.name3)
        if q:
            rr.elec_partial_charges = q

    pbt = PackedBlockTypes.from_restypes(
        base_pbt.chem_db,
        list(base_pbt.active_block_types) + ligand_refined,
    )
    return co, pbt


def build_context_from_biotite(
    biotite_structure: Any,
    *,
    canonical_ordering: Optional[Any] = None,
    packed_block_types: Optional[Any] = None,
    chem_db: Optional[Any] = None,
    missing_density_distance_threshold: float = 2.4,
    prepare_ligands: bool = False,
    ligand_ph: float = 7.0,
    strict_atom_types: bool = False,
    ligand_params_files: Optional[Sequence[Any]] = None,
) -> "BiotitePoseBuildContext":
    """Build a :class:`BiotitePoseBuildContext` from a biotite structure.

    The single seam where the ligand path plugs in. Mirrors the upstream
    PR #361 signature:

    * ``ligand_params_files`` — list of ``.xtal-lig.mmff94.tmol`` YAML
      files. When supplied, the returned context's ``chem_db`` is
      extended with the parsed atom_types + residues. **RDKit-free.**
    * ``prepare_ligands`` — when ``True``, upstream runs the
      RDKit/SMARTS/dimorphite pipeline. jtmol does not vendor RDKit, so
      this raises ``NotImplementedError`` and tells the caller to pre-
      bake the YAML instead.
    * ``ligand_ph`` / ``strict_atom_types`` — accepted for signature
      parity. ``ligand_ph`` is unused without ``prepare_ligands=True``;
      ``strict_atom_types`` is forwarded to
      :func:`extend_db_from_yaml`.
    """
    _require_biotite()

    if prepare_ligands:
        # ponytail: RDKit-only; SMARTS atom typing + 3D bond perception
        # have no JAX-friendly substitute, so we hard-fail with a clear
        # message rather than silently degrade.
        raise NotImplementedError(
            "prepare_ligands=True requires RDKit (SMARTS atom typing and "
            "3D bond perception); jtmol does not vendor RDKit. Pre-bake "
            "the ligand restype with upstream tmol and pass it via "
            "ligand_params_files=[...]."
        )

    parsed_files: Tuple[Any, ...] = ()
    extra_3lcs: Tuple[str, ...] = ()
    if ligand_params_files:
        chem_db, parsed_files, extra_3lcs = _derived_types_for_param_db(
            chem_db,
            ligand_params_files,
            strict_atom_types=strict_atom_types,
        )

    if parsed_files:
        # Extend co/pbt so the ligand 3lc enters the canonical ordering
        # and the ligand restype lands in the packed block types
        # (upstream PR #361). Caller-supplied co/pbt take precedence.
        ext_co, ext_pbt = _extended_co_pbt_for_ligands(parsed_files)
        co = canonical_ordering or ext_co
        pbt = packed_block_types or ext_pbt
    else:
        co = canonical_ordering or canonical_ordering_for_biotite()
        pbt = packed_block_types or packed_block_types_for_biotite()

    cf = canonical_form_from_biotite(
        biotite_structure,
        co=co,
        missing_density_distance_threshold=missing_density_distance_threshold,
    )

    if chem_db is None:
        from jtmol.database.chemical import ChemicalDatabase as _CD

        chem_db = _CD.get_default()

    return BiotitePoseBuildContext(
        chem_db=chem_db,
        canonical_ordering=co,
        packed_block_types=pbt,
        canonical_form=cf,
        ligand_params_files=parsed_files,
    )


def pose_stack_from_biotite(
    biotite_structure: Any,
    *,
    canonical_ordering: Optional[Any] = None,
    packed_block_types: Optional[Any] = None,
    missing_density_distance_threshold: float = 2.4,
    ligand_params_files: Optional[Sequence[Any]] = None,
    **kwargs: Any,
):
    """Build a :class:`PoseStack` from a biotite ``AtomArray`` / ``AtomArrayStack``.

    Parameters
    ----------
    biotite_structure:
        Input ``AtomArray`` (single model) or ``AtomArrayStack`` (NMR).
    canonical_ordering:
        Optional :class:`CanonicalOrdering`. Defaults to the module's
        :func:`canonical_ordering_for_biotite`.
    packed_block_types:
        Optional :class:`PackedBlockTypes`. Defaults to
        :func:`packed_block_types_for_biotite`.
    missing_density_distance_threshold:
        Å threshold above which an adjacent-residue connection is treated
        as broken. Set to 0 to disable. Default 2.4.
    **kwargs:
        Forwarded to ``jtmol.io.pose_stack_construction.pose_stack_from_canonical_form``
        when that module is available.

    Returns
    -------
    PoseStack
        A ``PoseStack`` built from the canonical form.
    """
    _require_biotite()

    # Ligand path: route through build_context_from_biotite so the ligand
    # restype enters the (extended) canonical ordering + packed block
    # types, then pack via the name3-keyed fallback (the construction
    # bridge does protein-specific terminus variant assignment that a
    # non-polymer ligand block must skip).
    if ligand_params_files:
        ctx = build_context_from_biotite(
            biotite_structure,
            canonical_ordering=canonical_ordering,
            packed_block_types=packed_block_types,
            missing_density_distance_threshold=missing_density_distance_threshold,
            ligand_params_files=ligand_params_files,
        )
        return _build_via_bridge_or_fallback(
            ctx.canonical_form, ctx.packed_block_types, ctx.canonical_ordering, **kwargs
        )

    co = canonical_ordering or canonical_ordering_for_biotite()
    pbt = packed_block_types or packed_block_types_for_biotite()

    cf = canonical_form_from_biotite(
        biotite_structure,
        co=co,
        missing_density_distance_threshold=missing_density_distance_threshold,
    )
    return _build_via_bridge_or_fallback(cf, pbt, co, **kwargs)


def _build_via_bridge_or_fallback(cf, pbt, co, **kwargs):
    """Route a canonical form through the pose_stack_construction bridge, which
    builds missing leaf atoms (hydrogens, backbone O/OXT) from ideal icoors so
    the pose scores correctly. Falls back to the direct PoseStackBuilder only
    if the bridge is unavailable or errors.

    The bridge resolves integer ``res_types`` via the ``canonical_ordering``
    entry of the canonical-form dict, so we inject ``co`` (biotite forms carry
    res_types, not res_names). Both the protein-only and ligand-complex paths
    use this: the older code called the bridge with the wrong argument order
    (swallowed by a bare except) and silently dropped to the fallback, which
    leaves absent hydrogens at the origin and inflates ``fa_rep`` by orders of
    magnitude on any heavy-atom-only input (crystal structures, ESMFold folds).
    """
    try:
        from jtmol.io.pose_stack_construction import (  # type: ignore
            pose_stack_from_canonical_form,
        )
    except Exception:
        return _pose_stack_from_canonical_form_fallback(pbt, cf)

    if isinstance(cf, dict) and "canonical_ordering" not in cf and co is not None:
        cf = {**cf, "canonical_ordering": co}
    return pose_stack_from_canonical_form(cf, pbt, **kwargs)


def _pose_stack_from_canonical_form_fallback(
    pbt: Any, cf: Dict[str, Any], *, co: Optional[Any] = None
):
    """Build a PoseStack from a canonical-form dict without the
    ``pose_stack_construction`` bridge.

    This is a best-effort shortcut: it looks up each residue's block-type
    index by 3lc (``pbt.active_block_types[i].name``) and packs only the
    real atoms, producing a ``PoseStack`` with sensible ``PDBInfo``.
    Non-canonical residues are skipped.
    """
    from jtmol.pose.pose_stack_builder import PoseStackBuilder

    res_types = np.asarray(cf["res_types"])  # [P, R]
    coords = np.asarray(cf["coords"])  # [P, R, A, 3]
    chain_labels = np.asarray(cf["chain_labels"])
    res_labels = np.asarray(cf["res_labels"])

    n_poses = int(res_types.shape[0])
    n_res = int(res_types.shape[1])

    # restype-index-in-co -> pbt block-type-index.
    active = getattr(pbt, "active_block_types", ())
    pbt_name_to_ind = {getattr(bt, "name", None): i for i, bt in enumerate(active)}
    # restype_io_equiv_classes gives the restype ordering used to build res_types.
    # We rely on the shared canonical ordering + pbt having the same labels,
    # which is the contract promised by ``default_canonical_ordering`` +
    # ``default_packed_block_types``.

    # Build per-pose lists for PoseStackBuilder.
    block_type_inds: List[np.ndarray] = []
    coords_per_pose: List[np.ndarray] = []
    chain_per_pose: List[List[Any]] = []
    res_num_per_pose: List[List[int]] = []
    res_name_per_pose: List[List[str]] = []

    # Build a map from the co-index of a restype to the pbt index by label.
    # We need co to go from "restype index used in cf" -> 3lc -> pbt index.
    if co is None:
        from jtmol.io.canonical_ordering import default_canonical_ordering

        co = default_canonical_ordering()
    co_restypes = getattr(co, "restype_io_equiv_classes", ())
    co_ind_to_3lc = {i: name for i, name in enumerate(co_restypes)}

    for p in range(n_poses):
        bti = np.full(n_res, -1, dtype=np.int32)
        per_res_coords: List[np.ndarray] = []
        chain_p: List[Any] = []
        res_num_p: List[int] = []
        res_name_p: List[str] = []

        for i in range(n_res):
            res3 = co_ind_to_3lc.get(int(res_types[p, i]), None)
            if res3 is None:
                continue
            pbt_ind = pbt_name_to_ind.get(res3, -1)
            if pbt_ind < 0:
                continue
            bti[i] = pbt_ind
            # Determine atom count for this block type.
            n_atoms_bt = int(np.asarray(pbt.n_atoms)[pbt_ind])
            atom_block = coords[p, i, :n_atoms_bt]
            # Fill nan with 0 so packed coords are at least numeric;
            # the missing-atom annotations live elsewhere.
            atom_block = np.where(np.isnan(atom_block), 0.0, atom_block)
            per_res_coords.append(atom_block.astype(np.float32))
            chain_p.append(str(chain_labels[p, i]) if chain_labels.size else "")
            res_num_p.append(int(res_labels[p, i]) if res_labels.size else (i + 1))
            res_name_p.append(res3)

        # Drop the padding (-1) slots.
        real_mask = bti >= 0
        block_type_inds.append(bti[real_mask])
        if per_res_coords:
            coords_per_pose.append(np.concatenate(per_res_coords, axis=0))
        else:
            coords_per_pose.append(np.zeros((0, 3), dtype=np.float32))
        chain_per_pose.append(chain_p)
        res_num_per_pose.append(res_num_p)
        res_name_per_pose.append(res_name_p)

    return PoseStackBuilder.from_block_types_and_coords(
        pbt,
        block_type_inds,
        coords_per_pose,
        chain_per_pose,
        res_num_per_pose,
        res_name_per_pose,
    )


__all__ = [
    "pose_stack_from_biotite",
    "canonical_form_from_biotite",
    "canonical_ordering_for_biotite",
    "packed_block_types_for_biotite",
    "BiotitePoseBuildContext",
    "build_context_from_biotite",
]
