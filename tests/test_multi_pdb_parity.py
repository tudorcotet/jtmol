"""Multi-PDB, 20-term parity check for :mod:`jtmol`.

This test takes the beta_nov2016_cart score function that
:mod:`jtmol.tests.test_upstream_parity` already validates against
upstream ``tmol`` on ``1ubq``, and re-exercises it on a small but
intentionally diverse panel of extra PDBs:

* ``1ubq`` — 76-res baseline monomer, no Cys.
* ``1qys`` — 92-res de novo design (Top7 family), no Cys.
* ``1BL8`` — KcsA transmembrane protein (chain A only in upstream's
  test PDB, 97 residues). Behaves like a monomer here but uses a
  different fold/composition than 1ubq / 1qys.
* ``1r21`` — small NMR ensemble with 3 Cys residues. At the coordinates
  in the first NMR model the Cys SGs sit ~8–10 Å apart, so the
  disulfide term should evaluate to zero. Kept in the panel as a
  "Cys-present but no bridge" negative control.
* ``3plc`` — 54-res structure with *multiple* real SG-SG disulfide
  bonds (6 pairs under 4 Å), the case that exercises the non-zero
  path of :class:`DisulfideEnergyTerm`.
* ``6DMZ_A`` — 47-res cysteine-rich disulfide bundle, a second
  independent non-zero disulfide case.

The harness:

1. Loads each PDB with :func:`jtmol.pose_stack_from_pdb`.
2. Builds a beta_nov2016 score function with **every** channel at
   weight 1.0 (so ``score_by_term`` returns the unweighted per-term
   energies that the upstream YAML goldens encode).
3. Computes ``sfxn.score_by_term(ps)`` and per-term gradients via
   ``jax.grad(lambda c: sfxn(ps, coords_override=c).sum())(ps.coords)``,
   plus per-term gradient sanity (finite, correct shape).
4. For 1ubq, compares to the frozen upstream goldens at
   ``tmol/tests/data/term_baselines/<Term>/test_whole_pose_scoring_10.yaml``.
5. Flags the disulfide term as non-zero-expected on 3plc / 6DMZ_A and
   documents it as an open issue if jtmol reports zero.

Running as a standalone script (the mode mentioned in the task) writes a
self-contained markdown report to ``docs/MULTI_PDB_PARITY.md`` and
prints a one-line summary to stdout. Running under pytest exercises the
same parametrized grid via assertions.

Run with::

    PYTHONPATH=. python tests/test_multi_pdb_parity.py
    # or, under pytest:
    PYTHONPATH=. python -m pytest tests/test_multi_pdb_parity.py -v
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

# Silence jtmol's plentiful UserWarning/DeprecationWarning spam so the
# report is legible. Only filter for this module's lifetime.
warnings.filterwarnings("ignore")

import numpy as np

# JAX is a hard dep for jtmol itself; if it's not importable everything
# downstream will fail anyway, so we import at module scope.
import jax
import jax.numpy as jnp


# --------------------------------------------------------------------- #
# Paths                                                                 #
# --------------------------------------------------------------------- #

#: Repo root, computed from this file. All PDB and golden paths below
#: hang off this, so the script/pytest run works regardless of cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: Test PDBs selected per the task brief. Paths point at the upstream
#: ``tmol/tests/data/pdb`` directory because those are the files the
#: upstream YAML goldens were generated against — avoiding accidental
#: golden/PDB version skew.
PDB_PATHS: dict[str, str] = {
    "1ubq": os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/1ubq.pdb"),
    "1qys": os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/1qys.pdb"),
    "1BL8": os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/1BL8.pdb"),
    "1r21": os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/1r21.pdb"),
    "3plc": os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/3plc.pdb"),
    "6DMZ_A": os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/6DMZ_A.pdb"),
}


# --------------------------------------------------------------------- #
# Score-type list + upstream golden map                                 #
# --------------------------------------------------------------------- #

#: The 20 score-type names we check. Matches the 20 non-``constraint``
#: channels that :func:`jtmol.beta2016_score_function` weights. Order
#: mirrors the canonical display order in :doc:`/docs/PARITY.md`.
TERM_NAMES: list[str] = [
    "fa_ljatr",
    "fa_ljrep",
    "fa_lk",
    "fa_elec",
    "hbond",
    "rama",
    "omega",
    "dunbrack_rot",
    "dunbrack_rotdev",
    "dunbrack_semirot",
    "cart_lengths",
    "cart_angles",
    "cart_torsions",
    "cart_impropers",
    "cart_hxltorsions",
    "lk_ball_iso",
    "lk_ball",
    "lk_bridge",
    "lk_bridge_uncpl",
    "ref",
]

#: ``disulfide`` lives outside the 20-term list because no upstream
#: golden exists for a Cys-less PDB and we check it separately on
#: cysteine-containing structures (1r21 / 3plc / 6DMZ_A).
DISULFIDE = "disulfide"

#: Maps each score-type name to the upstream ``term_baselines`` directory
#: and the 0-based index within that term's ``score_types()`` list. The
#: index is what makes ``term0`` / ``term1`` / ... in the YAML files
#: meaningful.
#:
#: Source of truth: the ``_score_types`` list in each
#: ``tmol/tmol/score/terms/*_creator.py``. If upstream reorders one of
#: those lists, regenerate this mapping.
UPSTREAM_TERM_DIR: dict[str, tuple[str, int]] = {
    "fa_ljatr": ("LJLKEnergyTerm", 0),
    "fa_ljrep": ("LJLKEnergyTerm", 1),
    "fa_lk": ("LJLKEnergyTerm", 2),
    "fa_elec": ("ElecEnergyTerm", 0),
    "hbond": ("HBondEnergyTerm", 0),
    "rama": ("BackboneTorsionEnergyTerm", 0),
    "omega": ("BackboneTorsionEnergyTerm", 1),
    "dunbrack_rot": ("DunbrackEnergyTerm", 0),
    "dunbrack_rotdev": ("DunbrackEnergyTerm", 1),
    "dunbrack_semirot": ("DunbrackEnergyTerm", 2),
    "cart_lengths": ("CartBondedEnergyTerm", 0),
    "cart_angles": ("CartBondedEnergyTerm", 1),
    "cart_torsions": ("CartBondedEnergyTerm", 2),
    "cart_impropers": ("CartBondedEnergyTerm", 3),
    "cart_hxltorsions": ("CartBondedEnergyTerm", 4),
    "lk_ball_iso": ("LKBallEnergyTerm", 0),
    "lk_ball": ("LKBallEnergyTerm", 1),
    "lk_bridge": ("LKBallEnergyTerm", 2),
    "lk_bridge_uncpl": ("LKBallEnergyTerm", 3),
    "ref": ("RefEnergyTerm", 0),
    "disulfide": ("DisulfideEnergyTerm", 0),
}


#: Forward tolerance. Looser than ``jtmol.tests.test_gradcheck``'s
#: because we're comparing frozen floats with some slight dtype/algorithm
#: drift (see :doc:`/docs/PARITY.md`): upstream and jtmol now agree
#: to well under 1% on every term for 1ubq, so a 0.01 absolute + 1%
#: relative budget captures all of them without masking real regressions.
FWD_ATOL = 1e-2
FWD_RTOL = 1e-2


# --------------------------------------------------------------------- #
# Small helpers                                                         #
# --------------------------------------------------------------------- #


def _close(a: float, b: float, atol: float = FWD_ATOL, rtol: float = FWD_RTOL) -> bool:
    """``|a - b| <= atol + rtol * |b|`` — mirrors :mod:`numpy.isclose`."""
    return abs(a - b) <= atol + rtol * abs(b)


def _load_yaml_golden(term_dir: str, term_idx: int) -> Optional[float]:
    """Return ``pose0`` scalar from the upstream frozen baseline, or None.

    Returns ``None`` if the YAML file is missing or malformed — upstream
    doesn't ship goldens for every term we might be interested in, and
    the fallback "no reference" branch of the caller is always valid.
    """
    import yaml  # lazy: only used when a golden is actually requested

    path = os.path.join(
        _REPO_ROOT,
        "tmol/tmol/tests/data/term_baselines",
        term_dir,
        "test_whole_pose_scoring_10.yaml",
    )
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        return None
    term_key = f"term{term_idx}"
    try:
        return float(data[term_key]["pose0"])
    except (KeyError, TypeError, ValueError):
        return None


def _build_full_sfxn():
    """Return a beta2016 score function with every channel weight set to 1.0.

    The stock factory applies the published beta_nov2016_cart weights
    (e.g. ``fa_ljrep = 0.55``, ``omega = 0.48``). The caller wants
    *unweighted* per-term energies to compare against the YAML goldens,
    which are themselves unweighted, so we override every weight to 1.0
    on top of the factory's layout. This keeps the underlying terms and
    their siblings registered without re-implementing term discovery.
    """
    import jtmol
    from jtmol.score.score_types import ScoreType

    sfxn = jtmol.beta2016_score_function()
    for st in list(ScoreType):
        if st is ScoreType.n_score_types:
            continue
        # Force every provided score type to unit weight. ``set_weight``
        # is idempotent and will lazily register any missing term. We
        # swallow KeyError because not every ScoreType value has a
        # concrete term (e.g. upstream's DNA/RNA entries aren't wired up
        # in jtmol yet and would fail the registry lookup).
        try:
            sfxn.set_weight(st, 1.0)
        except KeyError:
            pass
    return sfxn


@dataclass
class _TermResult:
    """Per-PDB, per-term evaluation result bag.

    One of these is populated for every (PDB, term) cell of the final
    report table. The rendering code below converts these into markdown
    rows; no state is kept inside the dataclass beyond what's needed for
    display + assertion.
    """

    term: str
    value: Optional[float]
    golden: Optional[float]
    grad_ok: bool
    grad_max_abs: float
    error: Optional[str] = None


@dataclass
class _PdbResult:
    """Aggregated result for one PDB across all 20 (+disulfide) terms."""

    label: str
    pdb_path: str
    load_error: Optional[str] = None
    n_atoms: int = 0
    n_res: int = 0
    cys_count: int = 0
    any_ss_close: bool = False
    grad_finite: bool = True
    grad_nan_terms: list[str] = field(default_factory=list)
    terms: dict[str, _TermResult] = field(default_factory=dict)


# --------------------------------------------------------------------- #
# PDB introspection (no energy eval)                                    #
# --------------------------------------------------------------------- #


def _pdb_stats(ps) -> tuple[int, int, int, bool]:
    """Return ``(n_atoms, n_res, n_cys, any_SG_SG_lt_3p5)``.

    Walks the pose stack's block-type index to count cystines, and if
    any are present measures their SG-SG pair distances. This is the
    same geometry the disulfide term uses internally
    (:mod:`jtmol.score.disulfide.disulfide_energy_term` uses a 3.5 Å
    cutoff), so ``any_ss_close`` is a faithful predictor of whether the
    term should report a non-zero energy.
    """
    pbt = ps.packed_block_types
    bt_ind = np.asarray(ps.block_type_ind)
    bco = np.asarray(ps.block_coord_offset)
    coords = np.asarray(ps.coords)

    n_atoms = int(ps.coords.shape[1])
    n_res = int((bt_ind[0] >= 0).sum())

    sg_positions: list[np.ndarray] = []
    for i in range(bt_ind.shape[1]):
        bi = bt_ind[0, i]
        if bi < 0:
            continue
        bt = pbt.active_block_types[int(bi)]
        if getattr(bt, "name3", None) != "CYS":
            continue
        a2i = getattr(bt, "atom_to_idx", None)
        if a2i is None or "SG" not in a2i:
            continue
        flat = int(bco[0, i] + a2i["SG"])
        sg_positions.append(np.asarray(coords[0, flat]))

    any_close = False
    for i in range(len(sg_positions)):
        for j in range(i + 1, len(sg_positions)):
            d = float(np.linalg.norm(sg_positions[i] - sg_positions[j]))
            if d < 3.5:
                any_close = True
                break
        if any_close:
            break

    return n_atoms, n_res, len(sg_positions), any_close


# --------------------------------------------------------------------- #
# Core evaluation                                                       #
# --------------------------------------------------------------------- #


def _evaluate_pdb(label: str, path: str) -> _PdbResult:
    """Evaluate all 20 terms (+disulfide) on ``path`` and return a report.

    Ordering:

    1. Load the PoseStack and capture its shape/composition so we can
       annotate the "why is this zero?" cells in the report (e.g. "no
       Cys so disulfide=0 is expected").
    2. Build the full-weight sfxn, run ``setup_for_pose_stack`` once so
       we don't pay the setup cost on every grad call.
    3. Compute ``score_by_term`` to fill the energies column.
    4. For each term, compute the per-term gradient via a fresh
       ``jax.grad`` on a closure that isolates *that* term via the
       score function's weight dict. This is more expensive than a
       single global grad but makes per-term NaN detection trivial.
       A final "sum of all terms" grad is also taken for the coarse
       check the task asks for.
    5. On 1ubq, compare to the upstream YAML goldens and record the
       delta in the result rows.
    """
    result = _PdbResult(label=label, pdb_path=path)

    if not os.path.isfile(path):
        result.load_error = f"PDB not found: {path}"
        return result

    try:
        import jtmol
        from jtmol.score.score_types import ScoreType
    except Exception as exc:
        result.load_error = f"jtmol import failed: {exc!r}"
        return result

    try:
        ps = jtmol.pose_stack_from_pdb(path)
    except Exception as exc:
        result.load_error = f"pose_stack_from_pdb failed: {exc!r}"
        return result

    try:
        result.n_atoms, result.n_res, result.cys_count, result.any_ss_close = (
            _pdb_stats(ps)
        )
    except Exception as exc:
        # Stats failure shouldn't nuke the whole row. Record and continue.
        result.load_error = f"pdb stats failed: {exc!r}"

    try:
        sfxn = _build_full_sfxn()
        state = sfxn.setup_for_pose_stack(ps)
    except Exception as exc:
        result.load_error = f"score function setup failed: {exc!r}"
        return result

    # ---- forward pass: score_by_term ----
    try:
        per_term = sfxn.score_by_term(ps, state=state)
    except Exception as exc:
        result.load_error = f"score_by_term failed: {exc!r}"
        return result

    # ---- global gradient sum ----
    def _total(coords):
        return sfxn(ps, state=state, coords_override=coords).sum()

    try:
        g_total = jax.grad(_total)(ps.coords)
        result.grad_finite = bool(jnp.all(jnp.isfinite(g_total)))
    except Exception as exc:
        # Global grad failure is a big deal — mark every term unknown
        # and move on so the per-term loop can still tell us *which*
        # term tripped the runtime.
        result.load_error = f"jax.grad of sfxn sum raised: {exc!r}"
        result.grad_finite = False

    # ---- per-term energies + per-term grad sanity ----
    # To isolate a single term for gradient purposes we clone the
    # existing sfxn and zero its weights. That's cheap (plain dict)
    # and avoids re-running setup_for_pose_stack per term.
    for name in TERM_NAMES + [DISULFIDE]:
        st = getattr(ScoreType, name)
        # Forward value (always well-defined; zero when the term returns
        # nothing for this structure).
        try:
            arr = per_term.get(st, None)
            e_val = float(jnp.asarray(arr).reshape(-1)[0]) if arr is not None else 0.0
        except Exception as exc:
            result.terms[name] = _TermResult(
                term=name,
                value=None,
                golden=None,
                grad_ok=False,
                grad_max_abs=float("nan"),
                error=f"forward eval failed: {exc!r}",
            )
            continue

        # Per-term gradient: rebuild a throwaway sfxn with *only* this
        # weight set. Recomputing setup per term would be wasteful since
        # the underlying term state is identical; instead we reuse the
        # existing ``state`` and just toggle weights by mutating a copy
        # of ``sfxn._weights`` inside a closure.
        #
        # Note: this is the one place we poke at private state. We do it
        # because ScoreFunction doesn't expose a "run me with a modified
        # weight dict, leaving self alone" API. Mutating inside a
        # try/finally restores the dict before the closure returns.
        saved_weights = dict(sfxn._weights)
        try:
            for other in list(saved_weights.keys()):
                sfxn._weights[other] = 0.0
            sfxn._weights[st] = 1.0

            def _single(coords, _st=st):
                per = sfxn.score_by_term(ps, state=state, coords_override=coords)
                if _st in per:
                    return per[_st].sum()
                return jnp.asarray(0.0, dtype=coords.dtype)

            try:
                g_single = jax.grad(_single)(ps.coords)
                gfin = bool(jnp.all(jnp.isfinite(g_single)))
                gmax = float(jnp.max(jnp.abs(g_single)))
                if not gfin:
                    result.grad_nan_terms.append(name)
            except Exception as exc:
                gfin = False
                gmax = float("nan")
                # Stash on result so the report shows a specific
                # "couldn't grad this term" note instead of swallowing
                # the failure silently.
                result.terms[name] = _TermResult(
                    term=name,
                    value=e_val,
                    golden=None,
                    grad_ok=False,
                    grad_max_abs=float("nan"),
                    error=f"grad failed: {exc!r}",
                )
                continue
        finally:
            sfxn._weights.clear()
            sfxn._weights.update(saved_weights)

        # Golden lookup: only meaningful on 1ubq for the 20-term panel.
        # For ``disulfide`` the 1ubq YAML golden was regenerated upstream
        # on a Cys-containing PDB and is not comparable to jtmol's
        # value on 1ubq (which must be 0). See the report for details.
        golden: Optional[float] = None
        if label == "1ubq" and name in UPSTREAM_TERM_DIR and name != DISULFIDE:
            term_dir, term_idx = UPSTREAM_TERM_DIR[name]
            golden = _load_yaml_golden(term_dir, term_idx)

        result.terms[name] = _TermResult(
            term=name,
            value=e_val,
            golden=golden,
            grad_ok=gfin,
            grad_max_abs=gmax,
        )

    return result


# --------------------------------------------------------------------- #
# Report rendering                                                      #
# --------------------------------------------------------------------- #


def _render_markdown(results: list[_PdbResult]) -> str:
    """Render the full multi-PDB report as a markdown document.

    Layout mirrors the task's Report contract:

    * Top-level summary of panel coverage.
    * Per-PDB shape stats (atoms / residues / Cys count / close SG-SG).
    * A wide per-term table with one row per PDB.
    * For 1ubq, a side-by-side "jax / upstream / delta" block.
    * A disulfide section calling out Cys-containing PDBs and flagging
      any open issue where the term should be non-zero but isn't.
    """
    lines: list[str] = []
    lines.append("# Multi-PDB 20-term parity check\n\n")
    lines.append(
        "_Generated by ``tests/test_multi_pdb_parity.py``._\n\n"
        "This document exercises the beta_nov2016_cart score function "
        "on several PDBs beyond 1ubq to surface edge cases that the "
        "single-monomer 1ubq baseline does not touch — disulfides, "
        "different chain compositions, and de novo folds.\n\n"
    )

    # ---- Panel overview ----
    lines.append("## Panel\n\n")
    lines.append(
        "| PDB | n_atoms | n_res | Cys | SG-SG<3.5A | Role |\n"
        "|---|---:|---:|---:|:---:|---|\n"
    )
    roles = {
        "1ubq": "simple monomer (baseline, matches PARITY.md)",
        "1qys": "larger monomer (Top7 family, designed)",
        "1BL8": "KcsA chain A (tests different fold / composition)",
        "1r21": "NMR ensemble with 3 Cys but no SG-SG < 3.5 A",
        "3plc": "cysteine-rich, 6 SG-SG disulfide contacts",
        "6DMZ_A": "cysteine-rich, 5 SG-SG disulfide contacts",
    }
    for r in results:
        role = roles.get(r.label, "")
        close_mark = "yes" if r.any_ss_close else "no"
        lines.append(
            f"| {r.label} | {r.n_atoms} | {r.n_res} | {r.cys_count} "
            f"| {close_mark} | {role} |\n"
        )
    lines.append("\n")

    # ---- Per-term table ----
    lines.append("## Per-term energies\n\n")
    header = "| PDB | status |"
    sep = "|---|:---:|"
    for name in TERM_NAMES:
        header += f" {name} |"
        sep += " ---: |"
    lines.append(header + "\n" + sep + "\n")
    for r in results:
        status = "load_error" if r.load_error else (
            "OK" if r.grad_finite and not r.grad_nan_terms else "NaN grad"
        )
        if r.load_error:
            line = f"| {r.label} | {status}: {r.load_error} |"
            line += " |" * len(TERM_NAMES)
            lines.append(line + "\n")
            continue
        line = f"| {r.label} | {status} |"
        for name in TERM_NAMES:
            tr = r.terms.get(name)
            if tr is None or tr.value is None:
                cell = "?"
            else:
                cell = f"{tr.value:+.3f}"
                if not tr.grad_ok:
                    cell += " (NaN grad)"
            line += f" {cell} |"
        lines.append(line + "\n")
    lines.append("\n")

    # ---- Upstream goldens (1ubq side-by-side) ----
    lines.append("## 1ubq vs upstream goldens\n\n")
    lines.append(
        "Goldens loaded from "
        "``tmol/tmol/tests/data/term_baselines/<Term>/test_whole_pose_scoring_10.yaml``"
        " (pose0 entry — all 10 poses are identical copies).\n\n"
    )
    lines.append(
        "| Term | jtmol | upstream | abs err | % err |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    ubq = next((r for r in results if r.label == "1ubq"), None)
    if ubq is not None and ubq.load_error is None:
        for name in TERM_NAMES:
            tr = ubq.terms.get(name)
            if tr is None or tr.golden is None:
                continue
            abs_err = abs(tr.value - tr.golden)
            pct_err = (
                100.0 * abs_err / abs(tr.golden) if abs(tr.golden) > 1e-9 else 0.0
            )
            lines.append(
                f"| {name} | {tr.value:+.4f} | {tr.golden:+.4f} "
                f"| {abs_err:.4f} | {pct_err:.3f}% |\n"
            )
    else:
        lines.append("_(1ubq load failed; no comparison available.)_\n")
    lines.append("\n")

    # ---- Disulfide block ----
    lines.append("## Disulfide term\n\n")
    lines.append(
        "jtmol's :class:`DisulfideEnergyTerm` detects Cys residues "
        "via block-type ``name3 == 'CYS'`` and then filters candidate "
        "pairs by SG-SG distance under a 3.5 A cutoff (see "
        "``jtmol/score/disulfide/disulfide_energy_term.py``).\n\n"
    )
    lines.append(
        "| PDB | Cys | any SG-SG<3.5A | disulfide E | expected | note |\n"
        "|---|---:|:---:|---:|:---:|---|\n"
    )
    for r in results:
        tr = r.terms.get(DISULFIDE) if not r.load_error else None
        val = "?" if tr is None or tr.value is None else f"{tr.value:+.4f}"
        if r.cys_count == 0:
            expected = "0"
            note = "no Cys — zero is correct"
        elif not r.any_ss_close:
            expected = "0"
            note = "Cys present but all SG-SG > 3.5 A"
        else:
            expected = "<0"
            note = "SG-SG pairs at disulfide distance"
            if tr is not None and tr.value is not None and tr.value == 0.0:
                note += (
                    " — **OPEN ISSUE**: disulfide term returned 0 despite "
                    "disulfide-compatible Cys pairs"
                )
        close_mark = "yes" if r.any_ss_close else "no"
        lines.append(
            f"| {r.label} | {r.cys_count} | {close_mark} | {val} "
            f"| {expected} | {note} |\n"
        )
    lines.append("\n")

    # ---- Gradient sanity ----
    lines.append("## Gradient sanity\n\n")
    lines.append(
        "For every PDB we take ``jax.grad(lambda c: sfxn(ps, coords_override=c).sum())(ps.coords)``"
        " over the full 20-term sum, then separately over each individual "
        "term (to pinpoint which term produced a NaN, if any). "
        "A term is marked ``NaN grad`` in the energy table above if its "
        "isolated gradient has any non-finite entries.\n\n"
    )
    lines.append(
        "| PDB | all-terms grad finite | per-term NaNs |\n"
        "|---|:---:|---|\n"
    )
    for r in results:
        if r.load_error:
            lines.append(f"| {r.label} | n/a | {r.load_error} |\n")
            continue
        finite_mark = "yes" if r.grad_finite else "**NO**"
        nan_list = ", ".join(r.grad_nan_terms) if r.grad_nan_terms else "(none)"
        lines.append(f"| {r.label} | {finite_mark} | {nan_list} |\n")
    lines.append("\n")

    return "".join(lines)


# --------------------------------------------------------------------- #
# Standalone driver                                                     #
# --------------------------------------------------------------------- #


def run_all() -> tuple[list[_PdbResult], str]:
    """Execute the full sweep and return ``(results, markdown_report)``.

    Separated from :func:`main` so the pytest wrapper below can reuse
    the same evaluation pipeline without re-implementing the loop.
    """
    results: list[_PdbResult] = []
    for label, path in PDB_PATHS.items():
        print(f"[multi-pdb parity] evaluating {label} ({path})...", flush=True)
        r = _evaluate_pdb(label, path)
        if r.load_error:
            print(f"  -> load/setup error: {r.load_error}")
        else:
            bad = [n for n, tr in r.terms.items() if not tr.grad_ok]
            print(
                f"  -> n_res={r.n_res}, n_atoms={r.n_atoms}, "
                f"Cys={r.cys_count}, any_ss_close={r.any_ss_close}, "
                f"all_finite={r.grad_finite}, NaN_terms={bad or '[]'}"
            )
        results.append(r)
    report = _render_markdown(results)
    return results, report


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point: run the sweep and write ``docs/MULTI_PDB_PARITY.md``.

    Returns 0 on success, 1 if any PDB failed to load or produced a
    NaN gradient. Forward-value mismatches against upstream are reported
    in the markdown but don't fail the CLI — the task's assertion surface
    is "can we score all of these and did any grad NaN?".
    """
    results, report = run_all()
    out_path = os.path.join(_REPO_ROOT, "docs", "MULTI_PDB_PARITY.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {out_path}")

    # Exit code: non-zero if anything outright broke (load failure, NaN grad).
    status = 0
    for r in results:
        if r.load_error:
            print(f"[fail] {r.label}: {r.load_error}")
            status = 1
        if not r.grad_finite or r.grad_nan_terms:
            print(
                f"[fail] {r.label}: grad_finite={r.grad_finite}, "
                f"nan_terms={r.grad_nan_terms}"
            )
            status = 1
    return status


# --------------------------------------------------------------------- #
# pytest wrappers                                                       #
# --------------------------------------------------------------------- #


def _run_sweep_once(cache: dict) -> list[_PdbResult]:
    """Memoised accessor so pytest parametrize doesn't re-run the whole sweep.

    The ``cache`` argument is the pytest session cache (or a plain dict
    when called from :func:`main`). Keyed once under ``"results"`` so
    every (pdb, term) cell of the parametrize grid reuses the single
    evaluation.
    """
    results = cache.get("results")
    if results is None:
        results, _report = run_all()
        cache["results"] = results
    return results


try:
    import pytest  # noqa: E402  - optional, only imported when available
except ImportError:  # pragma: no cover - pytest is a test-time dep
    pytest = None


if pytest is not None:

    @pytest.fixture(scope="session")
    def _sweep_results():
        """Session-wide cache of the full multi-PDB sweep.

        Scoping this at session level means the 20 * 6 parametrize grid
        runs the sweep once total, not 120 times. We also take this
        opportunity to write the markdown report so ``pytest`` invocations
        keep ``docs/MULTI_PDB_PARITY.md`` fresh.
        """
        results, report = run_all()
        try:
            out_path = os.path.join(_REPO_ROOT, "docs", "MULTI_PDB_PARITY.md")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                f.write(report)
        except OSError:
            # Don't fail the test suite just because we couldn't write
            # the report -- asserts below still run.
            pass
        return results

    @pytest.mark.parametrize("pdb_label", list(PDB_PATHS.keys()))
    def test_pdb_loads_and_scores(pdb_label: str, _sweep_results):
        """Smoke test: every PDB must load and score all 20+1 terms.

        Factored separately from the per-term parity tests so that if a
        PDB fails to load, pytest reports that as a single failure rather
        than 20 downstream failures.
        """
        r = next(x for x in _sweep_results if x.label == pdb_label)
        assert r.load_error is None, (
            f"{pdb_label}: {r.load_error}"
        )
        assert r.grad_finite, (
            f"{pdb_label}: total-energy grad has non-finite entries; "
            f"terms with NaN = {r.grad_nan_terms}"
        )

    @pytest.mark.parametrize("pdb_label", list(PDB_PATHS.keys()))
    @pytest.mark.parametrize("term_name", TERM_NAMES)
    def test_per_term_grad_finite(
        pdb_label: str, term_name: str, _sweep_results
    ):
        """Per-term NaN/Inf detector.

        We isolate the gradient for each term via the score function's
        weight dict so any one term that produces a non-finite gradient
        surfaces cleanly. Failure message includes the term's current
        energy for context.
        """
        r = next(x for x in _sweep_results if x.label == pdb_label)
        if r.load_error:
            pytest.skip(f"{pdb_label} did not load: {r.load_error}")
        tr = r.terms.get(term_name)
        assert tr is not None, f"no result captured for {pdb_label} / {term_name}"
        assert tr.grad_ok, (
            f"{pdb_label} / {term_name}: grad not finite "
            f"(E = {tr.value}, max|g| = {tr.grad_max_abs}, "
            f"error = {tr.error})"
        )

    @pytest.mark.parametrize("term_name", TERM_NAMES)
    def test_1ubq_matches_upstream_goldens(term_name: str, _sweep_results):
        """On 1ubq, each of the 20 terms must match upstream within tol.

        This is the "1 ubq matches upstream" half of the task's side-by-side
        report requirement, expressed as asserts. Other PDBs have no
        per-PDB upstream golden (upstream only ships ubq goldens via
        ``test_whole_pose_scoring_10.yaml``), so those comparisons live
        in the markdown report, not in asserts.
        """
        r = next(x for x in _sweep_results if x.label == "1ubq")
        if r.load_error:
            pytest.skip(f"1ubq failed to load: {r.load_error}")
        tr = r.terms[term_name]
        if tr.golden is None:
            pytest.skip(f"no YAML golden for {term_name}")
        assert _close(tr.value, tr.golden, atol=FWD_ATOL, rtol=FWD_RTOL), (
            f"{term_name} on 1ubq: jax={tr.value:+.6f}, "
            f"upstream={tr.golden:+.6f}, "
            f"delta={tr.value - tr.golden:+.4e} "
            f"(atol={FWD_ATOL}, rtol={FWD_RTOL})"
        )


# --------------------------------------------------------------------- #
# __main__                                                              #
# --------------------------------------------------------------------- #


if __name__ == "__main__":
    sys.exit(main())
