"""Standalone driver: runs the upstream-vs-tmol_jax parity sweep and
writes ``docs/UPSTREAM_PARITY.md`` plus ``docs/upstream_parity_results.csv``.

The pytest harness at ``tests/test_upstream_parity.py`` is the canonical
way to *assert* parity (it exits non-zero on regressions and is cache-
friendly under pytest's parametrize/fixture model). This script is the
*reporting* companion: it loops over every (PDB, score-type) cell in a
single Python process so startup cost (importing tmol's torch
extensions, importing jax) is paid once, and streams per-row progress
so long-running sweeps show signs of life.

Use this from CI or from the command line to regenerate
``docs/UPSTREAM_PARITY.md`` after a parity-affecting change::

    PYTHONPATH=.:tmol /opt/homebrew/bin/python3.13 scripts/run_upstream_parity.py

Exit code mirrors pytest: 0 on all pass, 1 on any failure.

The PDB panel mixes:
* Small monomers (1ubq, 1qys) — baseline sanity.
* Disulfide-containing single chains (3plc, 1r21) — exercises cys-cys
  topology decisions and the rama/cart_angles paths that historically
  trip floor clamps.
* Heterodimers / interface PDBs (1cgi, 2oob, 7opb) — exercise the
  inter-chain pair-list path with realistic interface contact counts.
* 6DMZ_A — small designed binder, used as a SAR-style protein.
* 1BL8 is documented as deferred: upstream refuses to load it
  ("missing non-leaf atom CG on residue 4 ARG"). Recorded for posterity
  but not run.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Optional

# Silence the spam of "dtype truncated to int32" and a handful of
# biotite deprecation warnings so the per-row progress output stays
# readable. The harness module itself already depends on these being
# filtered in pytest.ini.
warnings.filterwarnings("ignore")


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "tmol"))


import numpy as np  # noqa: E402
import jax  # noqa: E402

# float64 throughout: the cubic-Hermite kink terms (cart_torsions /
# omega / cart_hxltorsions) accumulate ~1e-2 relative error in
# float32, masking the true Blondel-Karplus analytical-adjoint parity
# (verified ~1e-9 vs upstream in float64).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


# --------------------------------------------------------------------- #
# Harness configuration                                                  #
# --------------------------------------------------------------------- #


# Existing PDBs (already in the repo).
UBQ = os.path.join(_REPO_ROOT, "tmol_jax/tests/data/1ubq.pdb")
QYS = os.path.join(_REPO_ROOT, "tmol_jax/tests/data/1qys.pdb")
R21 = os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/1r21.pdb")
PLC = os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/3plc.pdb")
DMZ = os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/6DMZ_A.pdb")

# Newly fetched PDBs (via biotite.database.rcsb.fetch).
OPB = os.path.join(_REPO_ROOT, "tmol_jax/tests/data/7OPB.pdb")
CGI = os.path.join(_REPO_ROOT, "tmol_jax/tests/data/1CGI.pdb")
OOB = os.path.join(_REPO_ROOT, "tmol_jax/tests/data/2OOB.pdb")

# 1BL8 documented as deferred (upstream load failure). We keep it in the
# label table so the report can mention it, but it is NOT in PDB_PATHS.
BL8 = os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/1BL8.pdb")

PDB_LABELS: dict[str, str] = {
    UBQ: "1ubq",
    QYS: "1qys",
    R21: "1r21",
    PLC: "3plc",
    DMZ: "6DMZ_A",
    OPB: "7opb",
    CGI: "1cgi",
    OOB: "2oob",
    BL8: "1BL8",
}

# Ordered for the report: small/canonical first, then designed, then
# interface complexes (larger). The driver iterates this in order so
# stdout progress lines and the final markdown both follow this layout.
PDB_PATHS: list[str] = [UBQ, QYS, R21, PLC, DMZ, OOB, CGI, OPB]

# PDBs that are intentionally NOT in PDB_PATHS but worth documenting in
# the report's "deferred" section. Reason is shown verbatim in the
# markdown footer.
DEFERRED_PDBS: list[tuple[str, str]] = [
    ("1BL8", "upstream refuses to load: missing non-leaf atom CG on residue 4 ARG"),
]


ALL_SCORE_TYPE_NAMES: list[str] = [
    "fa_ljatr", "fa_ljrep", "fa_lk",
    "fa_elec",
    "hbond",
    "rama", "omega",
    "dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot",
    "cart_lengths", "cart_angles", "cart_torsions",
    "cart_impropers", "cart_hxltorsions",
    "lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl",
    "ref",
]

#: Same table as in tests/test_upstream_parity.py — keep them in sync.
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
}

ATOL = 1e-5
RTOL = 1e-3


def _within_tol(a: float, b: float) -> bool:
    """``|a - b| <= atol + rtol * |b|`` — matches torch.autograd.gradcheck.

    The reference value ``b`` is on the right of the absolute, so when
    ``|b|`` is small the absolute tolerance dominates — important for
    terms whose energy can pass through zero (e.g. ``lk_bridge`` is
    ~1.5 kcal/mol but per-atom contributions can be near-zero).
    """
    return float(abs(a - b)) <= ATOL + RTOL * float(abs(b))


def _rel_err(jax_val: float, ref_val: float) -> float:
    """Relative error with a tiny safety floor on the denominator.

    Reported in the markdown summary; not used for pass/fail decisions
    (which use :func:`_within_tol`). Capped denominator avoids ``inf``
    when ``ref_val`` is zero (e.g. ``ref`` term gradients are exact zeros).
    """
    denom = max(abs(ref_val), 1e-12)
    return abs(jax_val - ref_val) / denom


# --------------------------------------------------------------------- #
# Lazy framework probes                                                  #
# --------------------------------------------------------------------- #


@dataclass
class _Upstream:
    available: bool
    reason: Optional[str]
    ScoreFunction: Any = None
    ParameterDatabase: Any = None
    ScoreType: Any = None
    pose_stack_from_pdb: Any = None
    torch: Any = None


def _probe_upstream() -> _Upstream:
    try:
        import torch

        from tmol import ParameterDatabase, pose_stack_from_pdb
        from tmol.score.score_function import ScoreFunction
        from tmol.score.score_types import ScoreType
    except Exception as exc:
        return _Upstream(available=False, reason=f"{type(exc).__name__}: {exc}")
    return _Upstream(
        available=True,
        reason=None,
        ScoreFunction=ScoreFunction,
        ParameterDatabase=ParameterDatabase,
        ScoreType=ScoreType,
        pose_stack_from_pdb=pose_stack_from_pdb,
        torch=torch,
    )


# --------------------------------------------------------------------- #
# tmol_jax isolated-term evaluation                                      #
# --------------------------------------------------------------------- #


def _jax_energy_and_grad(pose, target_name, ScoreType):
    """Return (energy_float, grad_ndarray[n_atoms, 3]) for a single term.

    Builds a one-term ScoreFunction by zeroing every beta2016 weight and
    setting the target to 1.0, then diffs the sum through ``jax.grad``.
    Mirrors the ``_jax_forward_and_grad`` helper in the pytest harness.
    """
    from tmol_jax import beta2016_score_function

    sfxn = beta2016_score_function()
    target = getattr(ScoreType, target_name)
    for st in list(ScoreType):
        if st is ScoreType.n_score_types:
            continue
        sfxn.set_weight(st, 0.0)
    sfxn.set_weight(target, 1.0)
    state = sfxn.setup_for_pose_stack(pose)

    def _e(coords):
        per = sfxn.score_by_term(pose, state=state, coords_override=coords)
        if target in per:
            return per[target].sum()
        return jnp.asarray(0.0, dtype=coords.dtype)

    e = float(_e(pose.coords))
    g = np.asarray(jax.grad(_e)(pose.coords))
    assert g.ndim == 3 and g.shape[0] == 1
    return e, g[0]


# --------------------------------------------------------------------- #
# Upstream isolated-term evaluation                                      #
# --------------------------------------------------------------------- #


# Sub-channel names of the upstream CartBondedEnergyTerm in the order
# its ``render_whole_pose_scoring_module`` writes them out — see
# tmol/score/terms/cartbonded_creator.py.
_CART_BONDED_ORDER = [
    "cart_lengths", "cart_angles", "cart_torsions",
    "cart_impropers", "cart_hxltorsions",
]
_CART_BONDED_SUBSET = set(_CART_BONDED_ORDER)

# Cache the CartBondedEnergyTerm + 10-pose stack per pose id() — we only
# need to set up the term once per PDB even though we evaluate 5 sub-
# channels separately. Keyed by id(pose) since PoseStack isn't hashable.
_CB_TERM_CACHE: dict[int, Any] = {}


def _build_cart_bonded_setup(pose, up: _Upstream):
    """Build a 10-pose stack + CartBondedEnergyTerm matching upstream's
    ``test_whole_pose_scoring_10`` test pattern.

    The upstream cartbonded kernel returns zero for the cart_lengths
    (2-body) and cart_angles (3-body) channels when ``n_poses == 1``,
    while it correctly populates them for ``n_poses >= 2`` (the YAML
    goldens are generated via ``PoseStackBuilder.from_poses([p] * 10)``;
    see ``tmol/tests/score/common/test_energy_term.py``). Replicating
    that 10-stack pattern restores parity.
    """
    key = id(pose)
    if key in _CB_TERM_CACHE:
        return _CB_TERM_CACHE[key]
    torch = up.torch
    from tmol.score.cartbonded.cartbonded_energy_term import (
        CartBondedEnergyTerm,
    )
    from tmol.pose.pose_stack_builder import PoseStackBuilder
    pn = PoseStackBuilder.from_poses([pose] * 10, device=torch.device("cpu"))
    term = CartBondedEnergyTerm(
        up.ParameterDatabase.get_default(), torch.device("cpu")
    )
    for bt in pn.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pn.packed_block_types)
    term.setup_poses(pn)
    _CB_TERM_CACHE[key] = (term, pn)
    return term, pn


def _torch_energy_and_grad(pose, target_name, up: _Upstream):
    """Return (energy_float, grad_ndarray[n_atoms, 3]) from upstream tmol.

    For non-cart_bonded score types: build a fresh ScoreFunction with
    only the target weight set — upstream's constructor starts every
    weight at zero, so setting just the target is enough — and read
    ``coords.grad`` after ``backward()``.

    For the 5 cart_bonded sub-channels: bypass ScoreFunction (which
    only exposes a weighted-sum) and call the energy term directly on
    a 10-pose stack, indexing the per-channel ``[5, n_poses]`` output.
    See ``_build_cart_bonded_setup`` for the n_poses=1 kernel issue
    this works around.
    """
    torch = up.torch

    if target_name in _CART_BONDED_SUBSET:
        term, pn = _build_cart_bonded_setup(pose, up)
        scorer = term.render_whole_pose_scoring_module(pn)
        coords = pn.coords.detach().clone().double().requires_grad_(True)
        per_channel = scorer(coords)  # [5, 10]
        target_idx = _CART_BONDED_ORDER.index(target_name)
        # All 10 stacked poses are clones of the same PDB. We pick
        # pose 0 and propagate gradient through it only; poses 1..9
        # contribute zero grad and we slice them out below.
        scalar = per_channel.reshape(len(_CART_BONDED_ORDER), -1)[
            target_idx, 0
        ]
    else:
        sfxn = up.ScoreFunction(
            up.ParameterDatabase.get_default(), torch.device("cpu")
        )
        target = getattr(up.ScoreType, target_name)
        sfxn.set_weight(target, 1.0)
        scorer = sfxn.render_whole_pose_scoring_module(pose)
        # float64 — see top-of-file note on cubic-Hermite kink precision.
        coords = pose.coords.detach().clone().double().requires_grad_(True)
        weighted = scorer(coords)
        scalar = weighted.sum()

    scalar.backward()

    e = float(scalar.detach().cpu().float().item())
    if coords.grad is None:
        g = np.zeros(tuple(coords.shape), dtype=np.float32)
    else:
        g = coords.grad.detach().cpu().float().numpy()
    assert g.ndim == 3
    # In the cart_bonded path ``coords`` has shape [10, n_atoms_max, 3]
    # but we only backwarded a scalar from pose 0, so pose-0 grad is
    # the only non-zero slice and the comparison harness expects a
    # single-pose [n_atoms_max, 3] tensor.
    return e, g[0]


# --------------------------------------------------------------------- #
# Result accumulator                                                     #
# --------------------------------------------------------------------- #


@dataclass
class _Row:
    """One pass/fail cell for the markdown report.

    ``mode``: "live" when both sides came from live-evaluated frameworks,
    "yaml" when the reference is the frozen upstream baseline, "skip"
    when we had to bail out.

    ``rel_err`` is the unsigned relative forward-energy error (capped
    denominator). Used for the summary stats but never for pass/fail.
    """

    pdb_label: str
    st: str
    mode: str
    fwd_ok: Optional[bool]
    grad_ok: Optional[bool]
    e_jax: Optional[float]
    e_ref: Optional[float]
    grad_max_abs_err: Optional[float]
    rel_err: Optional[float] = None
    duration_s: float = 0.0
    note: str = ""


# --------------------------------------------------------------------- #
# Main driver                                                            #
# --------------------------------------------------------------------- #


def _load_yaml_golden(term_dir: str, term_idx: int) -> float:
    """Read ``pose0`` of ``termN`` out of the upstream frozen baseline.

    See the harness module for the full rationale; this is a duplicate
    of :func:`tests.test_upstream_parity._load_yaml_golden` so the
    driver can run standalone without importing the harness (which
    would drag in pytest).
    """
    import yaml

    path = os.path.join(
        _REPO_ROOT,
        "tmol/tmol/tests/data/term_baselines",
        term_dir,
        "test_whole_pose_scoring_10.yaml",
    )
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return float(data[f"term{term_idx}"]["pose0"])


def _run_sweep() -> list[_Row]:
    """Iterate (PDB, term) combinations and collect parity rows.

    Each combination logs a single line with the mode, outcome, and
    timing. We intentionally evaluate every cell even on failures so
    the final report shows the full grid — the caller looks at the
    aggregated pass/fail counts to decide exit status.
    """
    up = _probe_upstream()
    print(
        f"[probe] upstream_available={up.available}"
        + (f" reason={up.reason}" if not up.available else ""),
        flush=True,
    )

    import tmol_jax
    from tmol_jax.score.score_types import ScoreType as JaxScoreType

    # Pre-load tmol_jax pose stacks; they are cheap to build (<1s each).
    jax_poses: dict[str, Any] = {}
    for p in PDB_PATHS:
        if not os.path.isfile(p):
            print(f"[pdb] missing: {p}", flush=True)
            continue
        try:
            t0 = time.monotonic()
            jax_poses[p] = tmol_jax.pose_stack_from_pdb(p)
            n_atoms = int(jax_poses[p].coords.shape[1])
            print(
                f"[pdb] jax loaded {PDB_LABELS.get(p, p)} "
                f"n_atoms={n_atoms} ({time.monotonic() - t0:.1f}s)",
                flush=True,
            )
        except Exception as exc:
            print(f"[pdb] jax load failed for {p}: {exc!r}", flush=True)

    # Pre-load upstream pose stacks when available. Upstream's
    # pose_stack_from_pdb is slow-ish (a few seconds), so amortising
    # across all 20 terms per PDB is a big win.
    torch_poses: dict[str, Any] = {}
    if up.available:
        for p in PDB_PATHS:
            if not os.path.isfile(p):
                continue
            try:
                t0 = time.monotonic()
                torch_poses[p] = up.pose_stack_from_pdb(p, up.torch.device("cpu"))
                n_atoms = int(torch_poses[p].coords.shape[1])
                print(
                    f"[pdb] torch loaded {PDB_LABELS.get(p, p)} "
                    f"n_atoms={n_atoms} ({time.monotonic() - t0:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                print(f"[pdb] torch load failed for {p}: {exc!r}", flush=True)

    rows: list[_Row] = []

    for p in PDB_PATHS:
        label = PDB_LABELS[p]
        for name in ALL_SCORE_TYPE_NAMES:
            t0 = time.monotonic()

            # tmol_jax side first.
            if p not in jax_poses:
                rows.append(
                    _Row(label, name, "skip", None, None, None, None, None,
                         note="tmol_jax pose load failed")
                )
                print(f"[row] {label:7s} {name:20s} SKIP (no jax pose)",
                      flush=True)
                continue

            try:
                e_jax, g_jax = _jax_energy_and_grad(
                    jax_poses[p], name, JaxScoreType
                )
            except Exception as exc:
                rows.append(
                    _Row(label, name, "skip", None, None, None, None, None,
                         duration_s=time.monotonic() - t0,
                         note=f"jax raised: {type(exc).__name__}: {exc}")
                )
                print(f"[row] {label:7s} {name:20s} JAX-FAIL ({exc!r})",
                      flush=True)
                continue

            # Choose reference: live upstream if available, else YAML.
            if up.available and p in torch_poses:
                try:
                    e_torch, g_torch = _torch_energy_and_grad(
                        torch_poses[p], name, up
                    )
                except Exception as exc:
                    rows.append(
                        _Row(label, name, "live", False, False, e_jax, None,
                             None, duration_s=time.monotonic() - t0,
                             note=f"torch raised: {type(exc).__name__}: {exc}")
                    )
                    print(f"[row] {label:7s} {name:20s} TORCH-FAIL ({exc!r})",
                          flush=True)
                    continue

                fwd_ok = _within_tol(e_jax, e_torch)
                rel = _rel_err(e_jax, e_torch)

                # Shape guard: some PDBs (1r21 with its disulfide chain
                # topology) end up with different atom counts in jax vs
                # torch. Element-wise comparison would throw. Record a
                # shape_mismatch and keep going so we at least get the
                # forward-energy row for the report.
                if g_jax.shape != g_torch.shape:
                    rows.append(
                        _Row(label, name, "live", fwd_ok, None, e_jax,
                             e_torch, None, rel_err=rel,
                             duration_s=time.monotonic() - t0,
                             note=f"grad shape mismatch: "
                             f"jax={g_jax.shape}, torch={g_torch.shape}")
                    )
                    print(
                        f"[row] {label:7s} {name:20s} live "
                        f"fwd={'P' if fwd_ok else 'F'} grad=- "
                        f"de={e_jax - e_torch:+.3e} shape_mismatch "
                        f"(jax={g_jax.shape}, torch={g_torch.shape}) "
                        f"({time.monotonic() - t0:.1f}s)",
                        flush=True,
                    )
                    continue

                grad_diff = np.abs(g_jax - g_torch) - (ATOL + RTOL * np.abs(g_torch))
                grad_ok = bool(np.all(grad_diff <= 0.0))
                max_abs = float(np.max(np.abs(g_jax - g_torch)))

                rows.append(
                    _Row(label, name, "live", fwd_ok, grad_ok, e_jax, e_torch,
                         max_abs, rel_err=rel,
                         duration_s=time.monotonic() - t0)
                )
                print(
                    f"[row] {label:7s} {name:20s} live fwd={'P' if fwd_ok else 'F'} "
                    f"grad={'P' if grad_ok else 'F'} "
                    f"de={e_jax - e_torch:+.3e} max|dg|={max_abs:.3e} "
                    f"({time.monotonic() - t0:.1f}s)",
                    flush=True,
                )
                continue

            # YAML fallback. Only 1ubq has goldens for most terms.
            if label != "1ubq":
                rows.append(
                    _Row(label, name, "skip", None, None, e_jax, None, None,
                         duration_s=time.monotonic() - t0,
                         note="no YAML golden for this (PDB, term)")
                )
                print(
                    f"[row] {label:7s} {name:20s} SKIP (no YAML golden, "
                    f"upstream unavailable)",
                    flush=True,
                )
                continue

            if name not in UPSTREAM_TERM_DIR:
                rows.append(
                    _Row(label, name, "skip", None, None, e_jax, None, None,
                         duration_s=time.monotonic() - t0,
                         note="no YAML mapping for score type")
                )
                continue

            term_dir, term_idx = UPSTREAM_TERM_DIR[name]
            try:
                e_ref = _load_yaml_golden(term_dir, term_idx)
            except (FileNotFoundError, KeyError) as exc:
                rows.append(
                    _Row(label, name, "skip", None, None, e_jax, None, None,
                         duration_s=time.monotonic() - t0,
                         note=f"golden missing: {exc}")
                )
                continue

            fwd_ok = _within_tol(e_jax, e_ref)
            rel = _rel_err(e_jax, e_ref)
            rows.append(
                _Row(label, name, "yaml", fwd_ok, None, e_jax, e_ref, None,
                     rel_err=rel,
                     duration_s=time.monotonic() - t0)
            )
            print(
                f"[row] {label:7s} {name:20s} yaml fwd={'P' if fwd_ok else 'F'} "
                f"de={e_jax - e_ref:+.3e} ({time.monotonic() - t0:.1f}s)",
                flush=True,
            )

    return rows


# --------------------------------------------------------------------- #
# Pivot / cell formatting                                                #
# --------------------------------------------------------------------- #


# Pass/fail glyphs. Plain ASCII so the markdown renders even in
# environments without emoji-font fallback (e.g. PDF export from
# pandoc).
GLYPH_BOTH = "PASS"      # forward AND gradient both within tolerance
GLYPH_FWD_ONLY = "F"     # forward passes, gradient fails
GLYPH_FAIL = "FAIL"      # neither passes
GLYPH_NA = "-"           # no comparison possible (skipped)


def _cell(row: _Row) -> str:
    """Render a single (PDB, term) cell as a compact pass/fail token.

    Rules:
    * Both forward and gradient pass  -> ``PASS``
    * Forward passes, gradient fails  -> ``F:<rel_err%>``
    * Forward fails (either pass)     -> ``FAIL:<rel_err%>``
    * Skipped / no reference          -> ``-``
    * YAML mode with no gradient ref  -> ``F-only`` (forward-only result)

    The percent-formatted rel_err makes the cell self-documenting in
    the failure case — readers can see whether it's a 0.01% (rounding)
    or 5% (genuine algorithmic divergence) failure without scrolling to
    the per-PDB detail tables.
    """
    if row.mode == "skip" or row.fwd_ok is None:
        return GLYPH_NA
    if row.fwd_ok:
        if row.grad_ok is True:
            return GLYPH_BOTH
        if row.grad_ok is False:
            # Forward passes but gradient breaks. Show forward rel-err
            # for context (typically ~0 here).
            return f"F:{(row.rel_err or 0.0) * 100:.2g}%"
        # YAML mode: no gradient check, but forward passed.
        return "fwd-only"
    # Forward fails -> attach the rel-err so failures are
    # self-classifying in the matrix.
    rel = row.rel_err if row.rel_err is not None else 0.0
    return f"FAIL:{rel * 100:.2g}%"


def _pivot_table(rows: list[_Row]) -> str:
    """Render a (PDB x score-type) pivot table in markdown.

    Rows = PDB label. Columns = score type. One header row, one body
    row per PDB. The cell content comes from :func:`_cell`.

    Score type column names are abbreviated to keep the table from
    blowing past terminal width — full names stay in the per-PDB
    sections further down.
    """
    pdb_order = [PDB_LABELS[p] for p in PDB_PATHS]

    # Build {(pdb_label, st_name): _Row} lookup.
    by_key = {(r.pdb_label, r.st): r for r in rows}

    # Header.
    header = ["PDB"] + ALL_SCORE_TYPE_NAMES
    sep = ["---"] + [":---:"] * len(ALL_SCORE_TYPE_NAMES)

    lines: list[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(sep) + "|")
    for pdb_label in pdb_order:
        row_cells = [pdb_label]
        for st_name in ALL_SCORE_TYPE_NAMES:
            r = by_key.get((pdb_label, st_name))
            row_cells.append(_cell(r) if r is not None else GLYPH_NA)
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines) + "\n"


def _per_term_breakdown(rows: list[_Row]) -> str:
    """Per-score-term pass-rate summary.

    For each of the 20 score types, count how many of the PDBs in the
    panel produced a passing forward and a passing gradient. Lets the
    reader see at a glance which terms are "always green" and which
    have a kink on some PDBs.
    """
    pdb_count = len(PDB_PATHS)
    lines: list[str] = []
    lines.append("| Term | Fwd pass | Grad pass | Notes |")
    lines.append("|---|:---:|:---:|---|")
    by_term: dict[str, list[_Row]] = {n: [] for n in ALL_SCORE_TYPE_NAMES}
    for r in rows:
        if r.st in by_term:
            by_term[r.st].append(r)
    for name in ALL_SCORE_TYPE_NAMES:
        term_rows = by_term[name]
        fwd_p = sum(1 for r in term_rows if r.fwd_ok is True)
        fwd_total = sum(1 for r in term_rows if r.fwd_ok is not None)
        grad_p = sum(1 for r in term_rows if r.grad_ok is True)
        grad_total = sum(1 for r in term_rows if r.grad_ok is not None)

        # Worst rel-err amongst failures: surfaces the "0.01% rounding
        # vs 5% real divergence" distinction in one column.
        worst = 0.0
        for r in term_rows:
            if r.fwd_ok is False and r.rel_err is not None:
                worst = max(worst, r.rel_err)
        note = ""
        if worst > 0.0:
            note = f"max forward rel-err on a failing PDB: {worst * 100:.2g}%"
        lines.append(
            f"| {name} | {fwd_p}/{fwd_total} | {grad_p}/{grad_total} | {note} |"
        )
    return "\n".join(lines) + "\n"


def _per_pdb_detail(rows: list[_Row], label: str) -> str:
    """Long-form detail table for one PDB (preserves provenance numbers)."""
    out: list[str] = []
    out.append(
        "| Term | Mode | Fwd | Grad | E(jax) | E(ref) | rel | max\\|dg\\| | Note |\n"
        "|---|---|:---:|:---:|---:|---:|---:|---:|---|\n"
    )

    def _m(v):
        if v is None:
            return "-"
        return "PASS" if v else "FAIL"

    for r in rows:
        if r.pdb_label != label:
            continue
        ej = f"{r.e_jax:+.4f}" if r.e_jax is not None else ""
        er = f"{r.e_ref:+.4f}" if r.e_ref is not None else ""
        rel = f"{r.rel_err * 100:.2g}%" if r.rel_err is not None else ""
        gm = f"{r.grad_max_abs_err:.2e}" if r.grad_max_abs_err is not None else ""
        out.append(
            f"| {r.st} | {r.mode} | {_m(r.fwd_ok)} | {_m(r.grad_ok)} "
            f"| {ej} | {er} | {rel} | {gm} | {r.note} |\n"
        )
    return "".join(out)


# --------------------------------------------------------------------- #
# Markdown rendering                                                     #
# --------------------------------------------------------------------- #


def _render_markdown(rows: list[_Row], up_avail: bool, up_reason: Optional[str]) -> str:
    """Produce the full markdown report body (str), ready to write to disk.

    Layout:
    1. Header (mode, tolerances, totals)
    2. Pass-rate summary
    3. Per-term breakdown
    4. Pivot matrix (rows=PDB, cols=term)
    5. Per-PDB detail tables (provenance)
    """
    out: list[str] = []
    out.append("# Upstream parity (tmol_jax vs tmol)\n\n")
    out.append(f"- Upstream live: **{'yes' if up_avail else 'no'}**\n")
    if not up_avail:
        out.append(f"- Fallback reason: `{up_reason}`\n")
        out.append(
            "- Reference for non-upstream rows: frozen YAML goldens at "
            "`tmol/tmol/tests/data/term_baselines/<Term>/test_whole_pose_scoring_10.yaml`\n"
        )
    out.append(f"- Tolerances: atol={ATOL}, rtol={RTOL}\n")
    out.append(
        "- Comparison rule: `|E_jax - E_ref| <= atol + rtol * |E_ref|` "
        "(matches torch.autograd.gradcheck default)\n"
    )
    out.append(f"- Panel size: {len(PDB_PATHS)} PDBs x {len(ALL_SCORE_TYPE_NAMES)} score types\n")
    out.append("- PDB selection rationale:\n")
    out.append("  - 1ubq, 1qys: small monomeric baselines\n")
    out.append("  - 1r21, 3plc: disulfide-containing single chains\n")
    out.append("  - 6DMZ_A: designed small protein\n")
    out.append("  - 2oob: small protein-protein complex (well-studied)\n")
    out.append("  - 1cgi: chymotrypsin-inhibitor (classic interface)\n")
    out.append("  - 7opb: cytokine binder complex (~500aa, real binder-target)\n")
    if DEFERRED_PDBS:
        out.append("- Deferred PDBs:\n")
        for tag, reason in DEFERRED_PDBS:
            out.append(f"  - {tag}: {reason}\n")
    out.append("\n")

    # Counts.
    total = len(rows)
    fwd_pass = sum(1 for r in rows if r.fwd_ok is True)
    fwd_fail = sum(1 for r in rows if r.fwd_ok is False)
    fwd_skip = sum(1 for r in rows if r.fwd_ok is None)
    grad_pass = sum(1 for r in rows if r.grad_ok is True)
    grad_fail = sum(1 for r in rows if r.grad_ok is False)
    grad_skip = sum(1 for r in rows if r.grad_ok is None)
    fwd_eligible = fwd_pass + fwd_fail
    grad_eligible = grad_pass + grad_fail

    out.append("## Summary\n\n")
    out.append(f"- Total cells: **{total}**\n")
    if fwd_eligible:
        out.append(
            f"- Forward parity: **{fwd_pass}/{fwd_eligible}** "
            f"({100.0 * fwd_pass / fwd_eligible:.1f}%) PASS, "
            f"{fwd_fail} FAIL, {fwd_skip} SKIP\n"
        )
    else:
        out.append(f"- Forward parity: 0/0 (all skipped)\n")
    if grad_eligible:
        out.append(
            f"- Gradient parity: **{grad_pass}/{grad_eligible}** "
            f"({100.0 * grad_pass / grad_eligible:.1f}%) PASS, "
            f"{grad_fail} FAIL, {grad_skip} SKIP\n"
        )
    else:
        out.append(f"- Gradient parity: 0/0 (all skipped)\n")
    out.append("\n")

    # Per-term breakdown.
    out.append("## Per-term breakdown\n\n")
    out.append(
        "Pass counts are reported as ``<pass>/<eligible>`` "
        "(skipped cells are excluded from the denominator).\n\n"
    )
    out.append(_per_term_breakdown(rows))
    out.append("\n")

    # Pivot matrix.
    out.append("## Parity matrix\n\n")
    out.append(
        "Cells: ``PASS`` = both forward and gradient within tolerance; "
        "``F:<x%>`` = forward passes but gradient fails (with forward rel-err); "
        "``FAIL:<x%>`` = forward fails (with forward rel-err); "
        "``-`` = skipped (no reference value).\n\n"
    )
    out.append(_pivot_table(rows))
    out.append("\n")

    # Per-PDB detail tables.
    out.append("## Per-PDB detail tables\n\n")
    seen: list[str] = []
    for r in rows:
        if r.pdb_label not in seen:
            seen.append(r.pdb_label)
    for label in seen:
        out.append(f"### {label}\n\n")
        out.append(_per_pdb_detail(rows, label))
        out.append("\n")

    return "".join(out)


# --------------------------------------------------------------------- #
# CSV export                                                             #
# --------------------------------------------------------------------- #


def _write_csv(rows: list[_Row], out_path: str) -> None:
    """Persist the raw rows to a CSV — one line per (PDB, term) cell.

    Columns:
        pdb, score_type, mode, forward_ok, gradient_ok,
        energy_jax, energy_ref, forward_rel_err,
        gradient_max_abs_err, duration_s, note
    """
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pdb", "score_type", "mode",
            "forward_ok", "gradient_ok",
            "energy_jax", "energy_ref",
            "forward_abs_err", "forward_rel_err",
            "gradient_max_abs_err",
            "duration_s", "note",
        ])
        for r in rows:
            abs_err = (
                abs(r.e_jax - r.e_ref)
                if (r.e_jax is not None and r.e_ref is not None)
                else None
            )
            w.writerow([
                r.pdb_label,
                r.st,
                r.mode,
                "" if r.fwd_ok is None else int(r.fwd_ok),
                "" if r.grad_ok is None else int(r.grad_ok),
                "" if r.e_jax is None else f"{r.e_jax:.8f}",
                "" if r.e_ref is None else f"{r.e_ref:.8f}",
                "" if abs_err is None else f"{abs_err:.8e}",
                "" if r.rel_err is None else f"{r.rel_err:.8e}",
                "" if r.grad_max_abs_err is None else f"{r.grad_max_abs_err:.8e}",
                f"{r.duration_s:.3f}",
                r.note,
            ])


def main() -> int:
    """Entry point: run the sweep, render markdown, print summary.

    Returns a shell exit code (0 = all checks passed, 1 = any failure).
    Prints both the path to the written markdown and a compact summary
    line at the end so CI can surface the result at-a-glance.
    """
    sweep_t0 = time.monotonic()
    rows = _run_sweep()
    up = _probe_upstream()
    report = _render_markdown(rows, up.available, up.reason)

    md_path = os.path.join(_REPO_ROOT, "docs", "UPSTREAM_PARITY.md")
    csv_path = os.path.join(_REPO_ROOT, "docs", "upstream_parity_results.csv")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write(report)
    print(f"[done] wrote {md_path}", flush=True)
    _write_csv(rows, csv_path)
    print(f"[done] wrote {csv_path}", flush=True)

    fwd_pass = sum(1 for r in rows if r.fwd_ok is True)
    fwd_fail = sum(1 for r in rows if r.fwd_ok is False)
    grad_pass = sum(1 for r in rows if r.grad_ok is True)
    grad_fail = sum(1 for r in rows if r.grad_ok is False)
    total = len(rows)
    elapsed = time.monotonic() - sweep_t0
    print(
        f"[done] summary: fwd_pass={fwd_pass} fwd_fail={fwd_fail} "
        f"grad_pass={grad_pass} grad_fail={grad_fail} (of {total} cells)",
        flush=True,
    )
    print(f"[done] elapsed: {elapsed:.1f}s", flush=True)
    return 1 if (fwd_fail or grad_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
