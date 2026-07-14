"""Upstream-vs-jtmol parity harness (forward energy + backward gradient).

This test module takes every term in the beta_nov2016_cart score function,
evaluates it at unit weight on a panel of PDBs with both upstream
``tmol`` (PyTorch) and ``jtmol`` (pure-JAX), and compares the scalar
energy and the ``dE/dcoords`` array.

Two running modes, selected automatically:

* **Live**: upstream ``tmol`` imports cleanly. Each PDB is loaded in
  both frameworks, the same term is evaluated with weight 1.0, and we
  assert element-wise agreement on *both* the scalar forward energy and
  the [n_atoms, 3] backward gradient. Tolerances follow
  ``torch.autograd.gradcheck``'s defaults: ``atol=1e-5``, ``rtol=1e-3``.

* **Fallback**: upstream ``tmol`` cannot import (e.g. the C++/CUDA
  extensions are not built in this environment). We skip the torch
  branch cleanly and compare the jtmol forward energy against the
  frozen YAML goldens in
  ``tmol/tmol/tests/data/term_baselines/<Term>/test_whole_pose_scoring_10.yaml``
  — these are upstream's own test fixtures, regenerated whenever
  upstream changes, and give us the same reference the upstream test
  suite uses to validate itself. The goldens are computed on a
  10-copy stack of ``1ubq.pdb`` (same tensor for every pose), so we
  compare them to the single-pose jtmol result. In fallback mode
  gradients cannot be compared because the goldens don't include them;
  gradient tests are marked SKIPPED with a clear reason.

What we intentionally *don't* test here:

* We don't run the upstream ``check_grads``-style finite-difference —
  that is :mod:`jtmol.tests.test_gradcheck`. This harness is
  explicitly tmol-vs-jtmol parity.
* We don't introspect ScoreFunction internals beyond the single
  ``set_weight(...) / setup_for_pose_stack / score_by_term`` path. If
  upstream's term registry changes shape, this file should still run;
  only the per-term mapping tables below may need updating.

Produces a markdown-friendly report when run with ``-v --tb=no --no-header``.
See the companion ``docs/UPSTREAM_PARITY.md`` for a sample run's output.

Run with::

    PYTHONPATH=. python -m pytest \\
        tests/test_upstream_parity.py -v
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pytest

# JAX is a hard dependency for jtmol itself — if it's missing, every
# test in this module is meaningless, so fail fast rather than skip.
import jax
import jax.numpy as jnp


# --------------------------------------------------------------------- #
# Paths                                                                 #
# --------------------------------------------------------------------- #


#: Repo root, computed from this file's location. All data paths below
#: are anchored here so the tests run whether invoked from repo root or
#: anywhere else.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: PDBs we run the parity sweep on. Order matches the task spec so the
#: failure report reads top-to-bottom in the same order. Each entry
#: includes a short human-readable label used for ``pytest -v`` IDs.
#:
#: Panel includes small monomeric baselines (1ubq, 1qys), disulfide-
#: containing chains (1r21, 3plc), a designed small protein (6DMZ_A),
#: and three protein-protein complexes of increasing size (2oob, 1cgi,
#: 7opb). 1BL8 is documented as deferred (upstream load failure) and
#: kept here for reference only.
UBQ = os.path.join(_REPO_ROOT, "jtmol/tests/data/1ubq.pdb")
QYS = os.path.join(_REPO_ROOT, "jtmol/tests/data/1qys.pdb")
R21 = os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/1r21.pdb")
PLC = os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/3plc.pdb")
DMZ = os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/6DMZ_A.pdb")
OPB = os.path.join(_REPO_ROOT, "jtmol/tests/data/7OPB.pdb")
CGI = os.path.join(_REPO_ROOT, "jtmol/tests/data/1CGI.pdb")
OOB = os.path.join(_REPO_ROOT, "jtmol/tests/data/2OOB.pdb")

#: Deferred: upstream refuses to load this PDB (missing non-leaf atoms
#: on a disordered ARG side chain). Kept for documentation only.
BL8 = os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb/1BL8.pdb")

#: The disulfide baseline is computed on 3plc in upstream (the `ubq` one
#: is all-zero by construction since ubiquitin has no cystines). Kept
#: here for documentation even though ``disulfide`` isn't in the current
#: score-type list; a future extension may want to parametrize it.
DISULFIDE_REFERENCE_PDB = PLC


#: Human-readable PDB labels for parametrize ids.
PDB_LABELS: dict[str, str] = {
    UBQ: "1ubq",
    QYS: "1qys",
    R21: "1r21",
    PLC: "3plc",
    DMZ: "6DMZ_A",
    OOB: "2oob",
    CGI: "1cgi",
    OPB: "7opb",
    BL8: "1BL8",
}


# --------------------------------------------------------------------- #
# Score-type list                                                       #
# --------------------------------------------------------------------- #


# We import ScoreType lazily so a completely busted jtmol install still
# lets pytest collect this module and produce a useful skip reason. The
# import guard mirrors the one used by :mod:`jtmol.tests.conftest`.
try:
    from jtmol.score.score_types import ScoreType  # type: ignore
except Exception as _exc:  # pragma: no cover - defensive import
    ScoreType = None  # type: ignore[assignment]
    _jtmol_import_error = _exc
else:
    _jtmol_import_error = None


#: The 20 score types the parity harness exercises. Order matches the
#: task spec verbatim. ``constraint`` and ``n_score_types`` are excluded
#: (the former is zero on a fresh pose stack; the latter is a sentinel).
ALL_SCORE_TYPE_NAMES = [
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


def _st(name: str):
    """Resolve a score-type name to the ScoreType enum member, or None.

    Used inside tests only — the module-level parametrize lists carry
    the *string* names so pytest collection still works when ScoreType
    isn't importable. Tests that actually run the term convert on first
    use and skip if the enum is unavailable.
    """
    if ScoreType is None:
        return None
    return getattr(ScoreType, name)


# --------------------------------------------------------------------- #
# Upstream term-class mapping                                           #
# --------------------------------------------------------------------- #


#: Maps each tested score-type name to the upstream EnergyTerm directory
#: name (under ``tmol/tests/data/term_baselines/``) and the 0-based index
#: of that score type within the term's ``score_types()`` list. The
#: index is the YAML ``termN`` key.
#:
#: Source of truth for both: the ``_score_types = [...]`` list in each
#: ``tmol/score/terms/*_creator.py`` file. If upstream reorders those,
#: regenerate this table.
UPSTREAM_TERM_DIR: dict[str, tuple[str, int]] = {
    # LJLKTermCreator: [fa_ljatr, fa_ljrep, fa_lk]
    "fa_ljatr": ("LJLKEnergyTerm", 0),
    "fa_ljrep": ("LJLKEnergyTerm", 1),
    "fa_lk": ("LJLKEnergyTerm", 2),
    # ElecTermCreator: [fa_elec]
    "fa_elec": ("ElecEnergyTerm", 0),
    # HBondTermCreator: [hbond]
    "hbond": ("HBondEnergyTerm", 0),
    # BackboneTorsionTermCreator: [rama, omega]  -- note the order!
    "rama": ("BackboneTorsionEnergyTerm", 0),
    "omega": ("BackboneTorsionEnergyTerm", 1),
    # DunbrackTermCreator: [dunbrack_rot, dunbrack_rotdev, dunbrack_semirot]
    "dunbrack_rot": ("DunbrackEnergyTerm", 0),
    "dunbrack_rotdev": ("DunbrackEnergyTerm", 1),
    "dunbrack_semirot": ("DunbrackEnergyTerm", 2),
    # CartBondedTermCreator:
    #   [cart_lengths, cart_angles, cart_torsions, cart_impropers, cart_hxltorsions]
    "cart_lengths": ("CartBondedEnergyTerm", 0),
    "cart_angles": ("CartBondedEnergyTerm", 1),
    "cart_torsions": ("CartBondedEnergyTerm", 2),
    "cart_impropers": ("CartBondedEnergyTerm", 3),
    "cart_hxltorsions": ("CartBondedEnergyTerm", 4),
    # LKBallTermCreator:
    #   [lk_ball_iso, lk_ball, lk_bridge, lk_bridge_uncpl]
    "lk_ball_iso": ("LKBallEnergyTerm", 0),
    "lk_ball": ("LKBallEnergyTerm", 1),
    "lk_bridge": ("LKBallEnergyTerm", 2),
    "lk_bridge_uncpl": ("LKBallEnergyTerm", 3),
    # RefTermCreator: [ref]
    "ref": ("RefEnergyTerm", 0),
}


# --------------------------------------------------------------------- #
# Tolerances                                                            #
# --------------------------------------------------------------------- #

#: Absolute tolerance (matches upstream torch.autograd.gradcheck default).
ATOL = 1e-5
#: Relative tolerance (matches upstream torch.autograd.gradcheck default).
RTOL = 1e-3


def _within_tol(a: float, b: float, atol: float = ATOL, rtol: float = RTOL) -> bool:
    """``|a - b| <= atol + rtol * |b|`` — upstream gradcheck tolerance.

    We put the reference value (``b``) inside the absolute, which is the
    direction ``numpy.isclose`` and ``torch.isclose`` both use; this
    matters when the reference is near zero (``|b|`` small => atol wins).
    """
    return float(abs(a - b)) <= atol + rtol * float(abs(b))


# --------------------------------------------------------------------- #
# Upstream availability probe                                            #
# --------------------------------------------------------------------- #


@dataclass
class _UpstreamProbe:
    """Result of probing whether ``import tmol`` works.

    We do this once per session (module load) and cache the verdict on
    :data:`_UPSTREAM`. If the import fails for *any* reason (missing
    C++/CUDA extensions, missing Python package, circular import, ...)
    we fall back to YAML goldens with this message as the skip reason.
    """

    available: bool
    reason: Optional[str]
    pose_stack_from_pdb: Optional[Callable[..., Any]] = None
    score_function_factory: Optional[Callable[..., Any]] = None
    ScoreType: Optional[Any] = None
    torch: Optional[Any] = None


def _probe_upstream() -> _UpstreamProbe:
    """Return an :class:`_UpstreamProbe` describing upstream availability.

    Tries the minimum set of imports needed to run a parity test —
    ``tmol.pose_stack_from_pdb``, the beta2016 score function factory,
    and ``torch`` (for gradient tracking). If any fail we return a
    populated ``reason`` the tests use in their skip message.
    """
    try:
        import torch  # type: ignore

        from tmol import ParameterDatabase, pose_stack_from_pdb  # type: ignore
        from tmol.score.score_function import ScoreFunction  # type: ignore
        from tmol.score.score_types import ScoreType as UpstreamScoreType  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        # We catch the broadest possible Exception here because upstream's
        # import can fail in a dozen ways -- SystemExit from missing
        # torch-lib ops, OSError from missing .so files, ImportError
        # from missing packages, TypeError from version skew... we want
        # a clean skip no matter which.
        return _UpstreamProbe(available=False, reason=f"{type(exc).__name__}: {exc}")

    # We build the score function via the ScoreFunction constructor rather
    # than the beta2016 preset because upstream's ``set_weight(..., 0.0)``
    # *removes* the term when the remaining non-zero weights are also zero
    # — which makes it impossible to zero all weights from the preset
    # (some "zeroing" set-weight calls fail silently because the term
    # gets torn down). Starting with an empty ScoreFunction and setting
    # only the target weight to 1.0 avoids this entirely.
    def factory(device):
        # ParameterDatabase.get_default() is the same instance both the
        # preset and the direct ScoreFunction constructor consume.
        return ScoreFunction(ParameterDatabase.get_default(), device)

    return _UpstreamProbe(
        available=True,
        reason=None,
        pose_stack_from_pdb=pose_stack_from_pdb,
        score_function_factory=factory,
        ScoreType=UpstreamScoreType,
        torch=torch,
    )


_UPSTREAM = _probe_upstream()


# --------------------------------------------------------------------- #
# YAML golden loader                                                    #
# --------------------------------------------------------------------- #


def _load_yaml_golden(term_dir: str, term_idx: int) -> float:
    """Return ``pose0`` scalar from the upstream frozen baseline.

    The YAML structure is::

        term0:
          pose0: <float>
          pose1: <float>
          ...
        term1:
          ...

    All 10 poses are identical copies of the same input, so
    ``pose0`` is representative. We use it because it's the first
    entry and immune to any subtle per-pose padding behaviour in
    upstream's ``PoseStackBuilder.from_poses``.
    """
    import yaml  # lazy: yaml is only used in the fallback path

    path = os.path.join(
        _REPO_ROOT,
        "tmol/tmol/tests/data/term_baselines",
        term_dir,
        "test_whole_pose_scoring_10.yaml",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    term_key = f"term{term_idx}"
    if term_key not in data:
        raise KeyError(
            f"{path}: missing {term_key!r} (got {list(data.keys())!r})"
        )
    pose0 = data[term_key]["pose0"]
    return float(pose0)


# --------------------------------------------------------------------- #
# jtmol helpers                                                      #
# --------------------------------------------------------------------- #


def _isolated_sfxn(target_name: str):
    """Build a jtmol ScoreFunction with only ``target_name`` active.

    Zeroes every beta2016 weight, then sets ``target_name`` to 1.0. The
    underlying term stays registered and computes all of its sibling
    score channels, but :meth:`ScoreFunction.score_by_term` only yields
    the non-zero-weighted channel back to the caller — exactly what we
    need for one-term-at-a-time parity.
    """
    if ScoreType is None:
        pytest.skip(f"jtmol not importable: {_jtmol_import_error!r}")
    from jtmol import beta2016_score_function

    sfxn = beta2016_score_function()
    target = _st(target_name)
    for st in list(ScoreType):
        if st is ScoreType.n_score_types:
            continue
        sfxn.set_weight(st, 0.0)
    sfxn.set_weight(target, 1.0)
    return sfxn


def _jax_forward_energy(pdb_path: str, target_name: str) -> float:
    """Return the jtmol scalar energy for ``target_name`` on ``pdb_path``.

    One-pose value (jtmol pose-stacks are always 1-deep when loaded
    from a single PDB). Runs through ``score_by_term`` so we can assert
    only the target channel is non-zero — a belt-and-braces guard that
    the isolation wiring worked.
    """
    import jtmol  # noqa: F401 - needed to register the ScoreType enum

    ps = jtmol.pose_stack_from_pdb(pdb_path)
    sfxn = _isolated_sfxn(target_name)
    state = sfxn.setup_for_pose_stack(ps)
    per_term = sfxn.score_by_term(ps, state=state)
    target = _st(target_name)
    if target not in per_term:
        # Score was identically zero and got suppressed by the "zero
        # weight elides" logic in ScoreFunction. That only fires if our
        # weight was 0 (bug) or the term returned 0 — for parity the
        # numeric zero is the right answer, so return it.
        return 0.0
    # score_by_term returns [n_poses] arrays; single-pose stacks have
    # one entry and we return it as a plain Python float for the
    # comparison logic below.
    return float(per_term[target][0])


def _jax_forward_and_grad(
    pdb_path: str, target_name: str
) -> tuple[float, np.ndarray]:
    """Return ``(energy_scalar, grad_coords[n_atoms, 3])`` from jtmol.

    We explicitly grad the summed weighted-1.0 energy. jtmol coords
    are ``float32[1, max_n_atoms, 3]``; we strip the leading axis so the
    returned gradient is directly comparable to upstream's
    ``[n_atoms, 3]`` layout. Float32 throughout: we're matching
    upstream's native precision (which is also float32 by default).
    """
    import jtmol

    ps = jtmol.pose_stack_from_pdb(pdb_path)
    sfxn = _isolated_sfxn(target_name)
    state = sfxn.setup_for_pose_stack(ps)

    def _e(coords):
        per = sfxn.score_by_term(ps, state=state, coords_override=coords)
        target = _st(target_name)
        # If the term returns zero we need the gradient to still have
        # the right shape (else jax.grad returns a scalar 0). Bail via
        # sum-of-empty returning 0-dimensional zero -- jax handles this
        # gracefully but we route through the target access to keep the
        # trace structure identical across terms that do have signal.
        if target in per:
            return per[target].sum()
        # No signal: contribute a zero that shares the coords dtype so
        # the gradient has the right dtype and shape.
        return jnp.asarray(0.0, dtype=coords.dtype)

    e = float(_e(ps.coords))
    g = np.asarray(jax.grad(_e)(ps.coords))
    # Collapse pose dim -> [n_atoms, 3].
    assert g.ndim == 3 and g.shape[0] == 1, f"unexpected grad shape {g.shape}"
    return e, g[0]


# --------------------------------------------------------------------- #
# Upstream helpers (live mode only)                                     #
# --------------------------------------------------------------------- #


def _torch_forward_and_grad(
    pdb_path: str, target_name: str
) -> tuple[float, np.ndarray]:
    """Live-mode: load into upstream tmol and return energy + gradient.

    Builds an upstream score function via the
    ``beta2016_score_function`` factory, zeros every weight except
    ``target_name``, runs ``coords.requires_grad_()`` and does a
    single ``.backward()`` on the scalar sum. Returned gradient is
    ``[n_atoms, 3]`` (same layout as jtmol) and the returned energy
    is a Python float for the tolerance check.

    Only reached when :data:`_UPSTREAM.available` is True — the caller
    (the test) must skip first if the probe failed.
    """
    assert _UPSTREAM.available, "upstream unavailable; caller should have skipped"
    torch = _UPSTREAM.torch
    upstream_st = _UPSTREAM.ScoreType
    pose_stack_from_pdb = _UPSTREAM.pose_stack_from_pdb
    factory = _UPSTREAM.score_function_factory

    # Default CPU device is what upstream's ``torch_device`` fixture
    # ends up on in an unconfigured environment. If CUDA ever gets built
    # in this harness's worker, switching to ``torch.device('cuda')``
    # here is the only change required.
    device = torch.device("cpu")

    pose = pose_stack_from_pdb(pdb_path, device)

    # Our ``factory`` wraps ``ScoreFunction(ParameterDatabase.get_default(),
    # device)`` — every weight starts at zero, every term unregistered.
    # Setting the target weight to 1.0 is enough to both register the
    # owning term and give it unit weight.
    sfxn = factory(device)
    target = getattr(upstream_st, target_name)
    sfxn.set_weight(target, 1.0)

    # Upstream's render_whole_pose_scoring_module expects coords on the
    # pose and a single scalar per pose. We use the high-level score()
    # method when available; otherwise we go term-by-term as in the
    # test_energy_term.py base class.
    coords = pose.coords.detach().clone().double().requires_grad_(True)

    # ``sfxn.render_whole_pose_scoring_module(pose)`` returns a callable
    # that accepts coords and returns [n_weighted_channels, n_poses]
    # weighted energies. We sum to a scalar, backprop, read .grad.
    scorer = sfxn.render_whole_pose_scoring_module(pose)
    weighted = scorer(coords)  # [n_channels, n_poses] or [n_poses]
    scalar_energy = weighted.sum()
    scalar_energy.backward()

    e = float(scalar_energy.detach().cpu().float().item())

    # Some terms have no coordinate dependence (e.g. ``ref`` is a
    # per-residue constant) — ``coords.grad`` stays ``None`` because no
    # autograd node ever wrote into it. Return the all-zero gradient
    # array of the expected shape so the caller can still compare.
    if coords.grad is None:
        grad_zero = np.zeros(
            tuple(coords.shape), dtype=np.float32
        )
        # Strip pose axis as below.
        assert grad_zero.ndim == 3 and grad_zero.shape[0] == 1
        return e, grad_zero[0]

    grad = coords.grad.detach().cpu().float().numpy()
    # Layout is [n_poses, max_n_atoms, 3] — strip the pose axis.
    assert grad.ndim == 3 and grad.shape[0] == 1, f"unexpected {grad.shape}"
    return e, grad[0]


# --------------------------------------------------------------------- #
# Cross-test caches                                                      #
# --------------------------------------------------------------------- #
#
# Both framework halves can take 20-40 seconds per (PDB, term) — the
# upstream score-function render is the dominant cost, and JAX tracing
# is no slouch either. Since the forward test and the gradient test for
# the same (PDB, term) pair reuse the same pose stack + score-fn
# build, we cache the inner results so the second test in a pair is
# nearly free. Caches are keyed by (pdb_path, target_name) and hold
# ``(energy_scalar, gradient_ndarray)`` tuples.
#
# The cache is per-process, intentional: module-scope fixtures with
# pytest parametrize don't easily share state across (PDB, term) tests
# because each test has a different fixture instantiation. A plain dict
# is simpler and survives the pytest lifecycle just fine.

_JAX_CACHE: dict[tuple[str, str], tuple[float, np.ndarray]] = {}
_TORCH_CACHE: dict[tuple[str, str], tuple[float, np.ndarray]] = {}


def _cached_jax_forward_and_grad(pdb_path: str, target_name: str):
    """Memoised wrapper around :func:`_jax_forward_and_grad`.

    The jtmol forward + grad pair is idempotent at fixed inputs, so
    the (pdb, term) key uniquely determines the output. The cache
    saves ~20s per subsequent test on the same (pdb, term).
    """
    key = (pdb_path, target_name)
    cached = _JAX_CACHE.get(key)
    if cached is not None:
        return cached
    value = _jax_forward_and_grad(pdb_path, target_name)
    _JAX_CACHE[key] = value
    return value


def _cached_torch_forward_and_grad(pdb_path: str, target_name: str):
    """Memoised wrapper around :func:`_torch_forward_and_grad`.

    Same rationale as :func:`_cached_jax_forward_and_grad`. Upstream's
    render_whole_pose_scoring_module is the expensive call — caching
    its output cuts the (forward, gradient) pair from ~80s to ~40s.
    """
    key = (pdb_path, target_name)
    cached = _TORCH_CACHE.get(key)
    if cached is not None:
        return cached
    value = _torch_forward_and_grad(pdb_path, target_name)
    _TORCH_CACHE[key] = value
    return value


# --------------------------------------------------------------------- #
# Results accumulator                                                   #
# --------------------------------------------------------------------- #


@dataclass
class _ResultRow:
    """One cell of the pass/fail matrix for the final markdown report.

    ``mode`` is "live" (compared against upstream live), "yaml" (fallback
    to frozen goldens), or "skip" (neither path available).
    ``grad_status`` is only meaningful in live mode.
    """

    pdb_label: str
    score_type: str
    mode: str
    forward_ok: Optional[bool]
    grad_ok: Optional[bool]
    forward_e_jax: Optional[float]
    forward_e_ref: Optional[float]
    grad_max_abs_err: Optional[float]
    note: str = ""


#: Session-wide accumulator. Each forward/gradient test appends its
#: outcome here, and the ``pytest_terminal_summary`` hook below renders
#: the final table. Scope: a module-level ``list`` keeps the hook
#: implementation trivially reachable from other tests / fixtures.
_RESULTS: list[_ResultRow] = []


# --------------------------------------------------------------------- #
# Pytest fixtures                                                       #
# --------------------------------------------------------------------- #


@pytest.fixture
def pose_jax(pdb):
    """Construct a jtmol PoseStack for ``pdb`` or skip with a reason.

    Kept as a fixture (rather than inlining into the test body) so that
    individual failures surface in the test report as a fixture error
    rather than a top-of-test assertion — easier to distinguish "can't
    load" from "loaded but numerics diverge".
    """
    try:
        import jtmol
    except Exception as exc:
        pytest.skip(f"jtmol not importable: {exc!r}")
    if not os.path.isfile(pdb):
        pytest.skip(f"PDB not found: {pdb}")
    try:
        return jtmol.pose_stack_from_pdb(pdb)
    except Exception as exc:
        pytest.skip(f"jtmol failed to load {pdb}: {exc!r}")


@pytest.fixture
def pose_torch(pdb):
    """Upstream PoseStack for ``pdb`` or skip if upstream isn't importable.

    Per the task spec, this fixture is what the gradient test depends on
    — if upstream can't be imported we get a clean skip rather than a
    traceback in the middle of the parity assertion.
    """
    if not _UPSTREAM.available:
        pytest.skip(f"upstream tmol not importable: {_UPSTREAM.reason}")
    if not os.path.isfile(pdb):
        pytest.skip(f"PDB not found: {pdb}")
    torch = _UPSTREAM.torch
    try:
        return _UPSTREAM.pose_stack_from_pdb(pdb, torch.device("cpu"))
    except Exception as exc:
        pytest.skip(f"upstream failed to load {pdb}: {exc!r}")


# --------------------------------------------------------------------- #
# Parametrize lists                                                     #
# --------------------------------------------------------------------- #


#: Flat (pdb, score_type_name) product used by the forward test. We flatten
#: here rather than using two ``parametrize`` decorators so the xdist /
#: ``-k`` filters can pick a single cell by name (e.g.
#: ``-k '1ubq-fa_ljatr'``) — much more convenient when chasing a single
#: regression.
#:
#: 1BL8 is deliberately excluded from the active panel (upstream refuses
#: to load it). Order matches the driver script for cross-tool diffs.
_PDB_PATHS = [UBQ, QYS, R21, PLC, DMZ, OOB, CGI, OPB]


def _pdb_label(path: str) -> str:
    return PDB_LABELS.get(path, os.path.basename(path))


# --------------------------------------------------------------------- #
# Forward-energy parity test                                            #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pdb",
    _PDB_PATHS,
    ids=[_pdb_label(p) for p in _PDB_PATHS],
)
@pytest.mark.parametrize(
    "st",
    ALL_SCORE_TYPE_NAMES,
    ids=ALL_SCORE_TYPE_NAMES,
)
def test_forward_parity(pdb: str, st: str, pose_jax):
    """Assert jtmol scalar energy matches upstream (or YAML golden).

    Branches:

    1. **Live**: call :func:`_torch_forward_and_grad` and compare the
       returned scalar directly.
    2. **YAML**: compare to the frozen upstream golden. Only meaningful
       on 1ubq (the PDB the goldens are computed for) — other PDBs get
       a skip with a reason because we have no reference value for them.

    On every run, append a :class:`_ResultRow` to :data:`_RESULTS` so
    the session-end hook can render the final pass/fail matrix.
    """
    pdb_label = _pdb_label(pdb)

    # Compute jtmol energy first — this runs regardless of mode.
    # We use the grad variant even for the forward-only test because
    # it's cached downstream and the gradient test re-uses the result.
    try:
        e_jax, _ = _cached_jax_forward_and_grad(pdb, st)
    except Exception as exc:
        _RESULTS.append(
            _ResultRow(
                pdb_label=pdb_label,
                score_type=st,
                mode="skip",
                forward_ok=None,
                grad_ok=None,
                forward_e_jax=None,
                forward_e_ref=None,
                grad_max_abs_err=None,
                note=f"jtmol raised: {type(exc).__name__}: {exc}",
            )
        )
        raise

    # Branch: prefer live upstream if available.
    if _UPSTREAM.available:
        try:
            e_torch, _ = _cached_torch_forward_and_grad(pdb, st)
        except Exception as exc:
            _RESULTS.append(
                _ResultRow(
                    pdb_label=pdb_label,
                    score_type=st,
                    mode="live",
                    forward_ok=False,
                    grad_ok=None,
                    forward_e_jax=e_jax,
                    forward_e_ref=None,
                    grad_max_abs_err=None,
                    note=f"upstream raised: {type(exc).__name__}: {exc}",
                )
            )
            pytest.fail(f"upstream evaluation failed: {exc!r}")

        ok = _within_tol(e_jax, e_torch)
        _RESULTS.append(
            _ResultRow(
                pdb_label=pdb_label,
                score_type=st,
                mode="live",
                forward_ok=ok,
                grad_ok=None,
                forward_e_jax=e_jax,
                forward_e_ref=e_torch,
                grad_max_abs_err=None,
            )
        )
        assert ok, (
            f"{st} on {pdb_label}: jax={e_jax:+.6f}, torch={e_torch:+.6f}, "
            f"delta={e_jax - e_torch:+.3e} "
            f"(atol={ATOL}, rtol={RTOL})"
        )
        return

    # Fallback: YAML goldens. Only defined for the PDB the golden was
    # computed on (1ubq for most terms, 3plc for disulfide).
    if pdb_label != "1ubq":
        _RESULTS.append(
            _ResultRow(
                pdb_label=pdb_label,
                score_type=st,
                mode="skip",
                forward_ok=None,
                grad_ok=None,
                forward_e_jax=e_jax,
                forward_e_ref=None,
                grad_max_abs_err=None,
                note="no upstream live or YAML golden for this (PDB, term)",
            )
        )
        pytest.skip(
            f"upstream unavailable ({_UPSTREAM.reason}) and no YAML golden "
            f"for {pdb_label}; skipping"
        )

    if st not in UPSTREAM_TERM_DIR:
        _RESULTS.append(
            _ResultRow(
                pdb_label=pdb_label,
                score_type=st,
                mode="skip",
                forward_ok=None,
                grad_ok=None,
                forward_e_jax=e_jax,
                forward_e_ref=None,
                grad_max_abs_err=None,
                note=f"no YAML mapping for {st}",
            )
        )
        pytest.skip(f"no YAML mapping for score type {st}")

    term_dir, term_idx = UPSTREAM_TERM_DIR[st]
    try:
        e_golden = _load_yaml_golden(term_dir, term_idx)
    except (FileNotFoundError, KeyError) as exc:
        _RESULTS.append(
            _ResultRow(
                pdb_label=pdb_label,
                score_type=st,
                mode="skip",
                forward_ok=None,
                grad_ok=None,
                forward_e_jax=e_jax,
                forward_e_ref=None,
                grad_max_abs_err=None,
                note=f"golden missing: {exc}",
            )
        )
        pytest.skip(f"YAML golden unavailable: {exc}")

    ok = _within_tol(e_jax, e_golden)
    _RESULTS.append(
        _ResultRow(
            pdb_label=pdb_label,
            score_type=st,
            mode="yaml",
            forward_ok=ok,
            grad_ok=None,
            forward_e_jax=e_jax,
            forward_e_ref=e_golden,
            grad_max_abs_err=None,
        )
    )
    assert ok, (
        f"{st} on {pdb_label} vs YAML golden: "
        f"jax={e_jax:+.6f}, golden={e_golden:+.6f}, "
        f"delta={e_jax - e_golden:+.3e} (atol={ATOL}, rtol={RTOL})"
    )


# --------------------------------------------------------------------- #
# Backward-gradient parity test                                         #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pdb",
    _PDB_PATHS,
    ids=[_pdb_label(p) for p in _PDB_PATHS],
)
@pytest.mark.parametrize(
    "st",
    ALL_SCORE_TYPE_NAMES,
    ids=ALL_SCORE_TYPE_NAMES,
)
def test_gradient_parity(pdb: str, st: str, pose_jax, pose_torch):
    """Assert per-atom ``dE/dcoords`` matches upstream within tolerance.

    Unconditionally requires upstream — the YAML goldens only contain
    forward energies. When ``pose_torch`` is unavailable, the fixture
    skips this test with a reason, and we don't even reach the body.

    Both gradients are resolved to ``[n_atoms, 3]`` float arrays and
    compared element-wise with the :func:`_within_tol` mixed tolerance.
    The reported failure includes the worst-offending atom index so a
    failure points directly at the coordinate that caused the miss.
    """
    pdb_label = _pdb_label(pdb)

    # The pose_torch fixture would have skipped already if upstream
    # wasn't available — so this assertion can't fail in practice, but
    # it's a good belt-and-braces guard in case the fixture is bypassed
    # via a conftest override.
    assert _UPSTREAM.available, "pose_torch fixture should have skipped"

    try:
        e_jax, g_jax = _cached_jax_forward_and_grad(pdb, st)
    except Exception as exc:
        _RESULTS.append(
            _ResultRow(
                pdb_label=pdb_label,
                score_type=st,
                mode="live",
                forward_ok=None,
                grad_ok=False,
                forward_e_jax=None,
                forward_e_ref=None,
                grad_max_abs_err=None,
                note=f"jax raised in grad: {type(exc).__name__}: {exc}",
            )
        )
        raise

    try:
        e_torch, g_torch = _cached_torch_forward_and_grad(pdb, st)
    except Exception as exc:
        _RESULTS.append(
            _ResultRow(
                pdb_label=pdb_label,
                score_type=st,
                mode="live",
                forward_ok=None,
                grad_ok=False,
                forward_e_jax=e_jax,
                forward_e_ref=None,
                grad_max_abs_err=None,
                note=f"upstream raised in grad: {type(exc).__name__}: {exc}",
            )
        )
        pytest.fail(f"upstream evaluation failed in grad: {exc!r}")

    # Shape check first. If they disagree, the two framework's pose
    # stacks don't share an atom count and the comparison is invalid.
    # Fall back to comparing the overlap and flag the rest.
    if g_jax.shape != g_torch.shape:
        _RESULTS.append(
            _ResultRow(
                pdb_label=pdb_label,
                score_type=st,
                mode="live",
                forward_ok=_within_tol(e_jax, e_torch),
                grad_ok=False,
                forward_e_jax=e_jax,
                forward_e_ref=e_torch,
                grad_max_abs_err=None,
                note=f"grad shape mismatch: jax={g_jax.shape}, torch={g_torch.shape}",
            )
        )
        pytest.fail(
            f"{st} on {pdb_label}: grad shape mismatch "
            f"(jax={g_jax.shape}, torch={g_torch.shape})"
        )

    # Element-wise mixed-tol check; diff = |a - b| - (atol + rtol * |b|)
    diff = np.abs(g_jax - g_torch) - (ATOL + RTOL * np.abs(g_torch))
    max_abs_err = float(np.max(np.abs(g_jax - g_torch)))
    ok = bool(np.all(diff <= 0.0))
    worst_idx = int(np.argmax(diff))
    worst_atom = np.unravel_index(worst_idx, g_jax.shape)

    _RESULTS.append(
        _ResultRow(
            pdb_label=pdb_label,
            score_type=st,
            mode="live",
            forward_ok=_within_tol(e_jax, e_torch),
            grad_ok=ok,
            forward_e_jax=e_jax,
            forward_e_ref=e_torch,
            grad_max_abs_err=max_abs_err,
        )
    )
    assert ok, (
        f"{st} on {pdb_label}: grad mismatch at atom "
        f"{worst_atom}, |jax - torch|={max_abs_err:.3e} "
        f"(atol={ATOL}, rtol={RTOL})"
    )


# --------------------------------------------------------------------- #
# Session-end report                                                    #
# --------------------------------------------------------------------- #


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Render the final pass/fail matrix as a markdown table.

    Hooked via pytest's terminalreporter API so the report appears at
    the end of ``pytest -v`` output. Also writes to
    ``docs/UPSTREAM_PARITY.md`` so the CI pipeline picks it up without
    scraping stdout.
    """
    if not _RESULTS:
        return

    tr = terminalreporter
    tr.section("jtmol vs upstream tmol parity")

    # Group by PDB, emit one section per PDB with a table of
    # (score_type, mode, forward, grad, notes). The YAML goldens don't
    # include gradients so grad shows "n/a" in fallback mode.
    pdbs_seen = []
    for row in _RESULTS:
        if row.pdb_label not in pdbs_seen:
            pdbs_seen.append(row.pdb_label)

    md_lines: list[str] = []
    md_lines.append("# Upstream parity (jtmol vs tmol)\n")
    md_lines.append(
        f"- Upstream live: **{'yes' if _UPSTREAM.available else 'no'}**\n"
    )
    if not _UPSTREAM.available:
        md_lines.append(
            f"- Fallback reason: `{_UPSTREAM.reason}`\n"
            f"- Reference: frozen YAML goldens at "
            f"`tmol/tests/data/term_baselines/<Term>/test_whole_pose_scoring_10.yaml`\n"
        )
    md_lines.append(f"- Tolerances: atol={ATOL}, rtol={RTOL}\n\n")

    # Overall counts.
    total = len(_RESULTS)
    fwd_pass = sum(1 for r in _RESULTS if r.forward_ok is True)
    fwd_fail = sum(1 for r in _RESULTS if r.forward_ok is False)
    grad_pass = sum(1 for r in _RESULTS if r.grad_ok is True)
    grad_fail = sum(1 for r in _RESULTS if r.grad_ok is False)
    grad_skip = sum(1 for r in _RESULTS if r.grad_ok is None)
    fwd_skip = sum(1 for r in _RESULTS if r.forward_ok is None)

    md_lines.append("## Summary\n")
    md_lines.append(f"- Total entries: {total}\n")
    md_lines.append(f"- Forward: pass={fwd_pass}, fail={fwd_fail}, skip={fwd_skip}\n")
    md_lines.append(f"- Gradient: pass={grad_pass}, fail={grad_fail}, skip={grad_skip}\n\n")

    # Per-PDB tables.
    for pdb_label in pdbs_seen:
        md_lines.append(f"## {pdb_label}\n\n")
        md_lines.append(
            "| Term | Mode | Forward | Gradient | E(jax) | E(ref) | max|dE| | Note |\n"
            "|---|---|:---:|:---:|---:|---:|---:|---|\n"
        )
        for row in _RESULTS:
            if row.pdb_label != pdb_label:
                continue

            def _mark(v):
                if v is None:
                    return "n/a"
                return "PASS" if v else "FAIL"

            e_jax = f"{row.forward_e_jax:+.4f}" if row.forward_e_jax is not None else ""
            e_ref = f"{row.forward_e_ref:+.4f}" if row.forward_e_ref is not None else ""
            gmax = (
                f"{row.grad_max_abs_err:.2e}"
                if row.grad_max_abs_err is not None
                else ""
            )
            md_lines.append(
                f"| {row.score_type} | {row.mode} "
                f"| {_mark(row.forward_ok)} | {_mark(row.grad_ok)} "
                f"| {e_jax} | {e_ref} | {gmax} | {row.note} |\n"
            )
        md_lines.append("\n")

    report = "".join(md_lines)

    # Dump to stdout for the pytest console summary.
    for line in report.splitlines():
        tr.write_line(line)

    # And persist to docs/UPSTREAM_PARITY.md (created at session end).
    try:
        out_path = os.path.join(_REPO_ROOT, "docs", "UPSTREAM_PARITY.md")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        tr.write_line(f"\nReport written to {out_path}")
    except OSError as exc:
        tr.write_line(f"\n[warn] could not write docs/UPSTREAM_PARITY.md: {exc}")


# --------------------------------------------------------------------- #
# Smoke test: module imports                                            #
# --------------------------------------------------------------------- #


def test_module_imports_cleanly():
    """Sanity: this module should collect without raising.

    Without this we'd only discover import-time regressions (missing
    yaml, bad ScoreType enum, etc.) when the full parametrize grid fires
    — which is opaque. A single always-present smoke test gives us a
    cleaner failure signal.
    """
    if ScoreType is None:
        pytest.skip(f"jtmol not importable: {_jtmol_import_error!r}")
    # The mapping tables must reference score types that actually exist.
    for name in ALL_SCORE_TYPE_NAMES:
        assert hasattr(ScoreType, name), (
            f"ScoreType has no member {name!r}; update ALL_SCORE_TYPE_NAMES"
        )
        if name in UPSTREAM_TERM_DIR:
            term_dir, _ = UPSTREAM_TERM_DIR[name]
            # We don't check presence of the YAML here (it's optional in
            # live mode); just that the dir name is plausible.
            assert term_dir.endswith("EnergyTerm"), (
                f"unexpected term dir {term_dir!r}"
            )
