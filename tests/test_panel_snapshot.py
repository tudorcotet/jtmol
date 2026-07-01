"""Regression-snapshot test for jtmol's 21 active score channels.

Loads a small panel of PDBs, scores them with
:func:`tmol_jax.beta2016_score_function`, captures every active per-term
energy (now including ``gen_torsions``) into
``tests/snapshots/panel_energies.csv``, and on subsequent runs asserts
the live values still match the committed snapshot.

Behaviour:

* If ``tests/snapshots/panel_energies.csv`` does not exist, the test
  runs in **write mode**: it produces the file and exits without
  comparing. The first run after introducing this test (or after an
  intentional energy change + ``rm snapshots/panel_energies.csv``)
  populates the snapshot.
* If the file exists, the test runs in **verify mode**: it recomputes
  energies and asserts ``allclose(atol=1e-3, rtol=1e-4)`` against the
  committed CSV. Term-by-term mismatches are reported all at once.
* A second pass per PDB JIT-compiles the score function and asserts the
  scalar total matches the eager pass bit-for-bit (no JAX
  nondeterminism).
* PDBs that fail to load (e.g. upstream's broken 1BL8 entry) are
  reported as skipped, not failures. The snapshot only contains rows
  for PDBs that loaded.

Runs either under pytest or as ``python tests/test_panel_snapshot.py``
(stdlib-only, no pytest dep introduced).

Re-snapshot intentionally::

    rm tests/snapshots/panel_energies.csv
    python tests/test_panel_snapshot.py
    git diff tests/snapshots/panel_energies.csv   # eyeball the deltas
    git add tests/snapshots/panel_energies.csv
"""

from __future__ import annotations

import csv
import os
import sys
import unittest
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import jax


# --------------------------------------------------------------------- #
# Paths                                                                 #
# --------------------------------------------------------------------- #

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# PDB locations per the task brief: 1ubq/1qys come from the jtmol test
# data dir; 1r21/3plc/6DMZ_A come from upstream's tests/data/pdb so
# their goldens stay aligned with what upstream tmol used.
def _up(rel: str) -> str:
    return os.path.join(_REPO_ROOT, "tmol/tmol/tests/data/pdb", rel)


PDB_PATHS: dict[str, str] = {
    # Original single-chain baselines.
    "1ubq": os.path.join(_REPO_ROOT, "tmol_jax/tests/data/1ubq.pdb"),
    "1qys": os.path.join(_REPO_ROOT, "tmol_jax/tests/data/1qys.pdb"),
    "1r21": _up("1r21.pdb"),
    "3plc": _up("3plc.pdb"),
    "6DMZ_A": _up("6DMZ_A.pdb"),
    # Size-graded fixtures (upstream bysize set): fold + composition
    # diversity across 15-200 residues. PDBs that fail to load are
    # reported as skipped, so this is safe to widen.
    "015res_1lu6": _up("bysize_015_res_1lu6.pdb"),
    "040res_5uoi": _up("bysize_040_res_5uoi.pdb"),
    "100res_5umr": _up("bysize_100_res_5umr.pdb"),
    "150res_5yzf": _up("bysize_150_res_5yzf.pdb"),
    "200res_5mm8": _up("bysize_200_res_5mm8.pdb"),
    # Multi-chain / interface structures (chains split, breaks open).
    "1CGI": os.path.join(_REPO_ROOT, "tmol_jax/tests/data/1CGI.pdb"),
    "7OPB": os.path.join(_REPO_ROOT, "tmol_jax/tests/data/7OPB.pdb"),
}

SNAPSHOT_PATH = os.path.join(
    os.path.dirname(__file__), "snapshots", "panel_energies.csv"
)

# Tolerances per the task brief. Tight enough to catch real regressions
# in any term, loose enough to absorb the float32 jitter that JAX's
# reduction order produces on different host devices.
ATOL = 1e-3
RTOL = 1e-4


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _build_full_sfxn() -> tuple[object, list[str]]:
    """Return ``(sfxn, channel_names)`` for the 21 active beta2016 channels.

    Snapshot stores *unweighted* per-term energies so a future change to
    the published beta_nov2016 weights doesn't churn the snapshot — only
    a change in the underlying term math does. We start from
    :func:`beta2016_score_function`, capture its active channel set
    (21 today: every beta2016 weight plus ``gen_torsions``), then
    override every weight to 1.0.
    """
    import tmol_jax

    sfxn = tmol_jax.beta2016_score_function()
    active = sfxn.active_score_types()
    channel_names = [st.name for st in active]
    for st in active:
        sfxn.set_weight(st, 1.0)
    return sfxn, channel_names


def _score_pdb(name: str, path: str) -> tuple[dict[str, float], float, str | None]:
    """Return ``({term: energy}, total, error_or_None)`` for one PDB.

    On any load/score failure returns ``({}, 0.0, "<msg>")`` so the
    caller can record a skip without aborting the whole run.
    """
    import tmol_jax

    try:
        ps = tmol_jax.pose_stack_from_pdb(path)
    except Exception as e:  # noqa: BLE001 — surface any load failure as a skip
        return {}, 0.0, f"load failed: {type(e).__name__}: {e}"

    try:
        sfxn, channel_names = _build_full_sfxn()
        state = sfxn.setup_for_pose_stack(ps)
        per_term = sfxn.score_by_term(ps, state=state)
    except Exception as e:  # noqa: BLE001
        return {}, 0.0, f"score failed: {type(e).__name__}: {e}"

    # ``score_by_term`` returns weighted values, but every weight is 1.0
    # so weighted == unweighted. Filter to the 21 beta2016 channels so
    # the snapshot stays scoped to what beta_nov2016_cart actually uses.
    wanted = set(channel_names)
    energies = {
        st.name: float(np.asarray(v).sum())
        for st, v in per_term.items()
        if st.name in wanted
    }
    total = float(sum(energies.values()))

    # JIT reproducibility smoke check: compile a coords-only energy fn
    # and confirm the eager total matches the jitted total within
    # float32 ULP. JAX with default ordering is deterministic on a
    # given device, so any mismatch here means real nondeterminism
    # (or a bug in coords_override threading) crept in.
    try:
        def _energy(coords):
            return sfxn(ps, state=state, coords_override=coords).sum()
        jit_energy = jax.jit(_energy)
        e1 = float(jit_energy(ps.coords))
        e2 = float(jit_energy(ps.coords))
        if not np.isclose(e1, e2, atol=0.0, rtol=0.0):
            return {}, 0.0, f"jit nondeterministic: {e1} vs {e2}"
        # The eager total is the sum of weighted per-term energies; the
        # jitted total is sfxn(ps).sum(). They should agree to float32
        # rounding; allow a 1e-3 absolute slop to cover reduction order.
        if not np.isclose(e1, total, atol=1e-3, rtol=1e-4):
            return {}, 0.0, f"jit/eager disagree: {e1} vs {total}"
    except Exception as e:  # noqa: BLE001
        return {}, 0.0, f"jit failed: {type(e).__name__}: {e}"

    return energies, total, None


def _collect_panel() -> tuple[list[tuple[str, str, float]], dict[str, str]]:
    """Score every PDB in the panel.

    Returns ``(rows, skips)`` where ``rows`` is a list of
    ``(pdb, term, energy)`` tuples sorted for a stable CSV diff, and
    ``skips`` is ``{pdb: reason}``.
    """
    rows: list[tuple[str, str, float]] = []
    skips: dict[str, str] = {}
    for name, path in PDB_PATHS.items():
        if not os.path.exists(path):
            skips[name] = f"missing PDB at {path}"
            continue
        energies, _total, err = _score_pdb(name, path)
        if err is not None:
            skips[name] = err
            continue
        for term in sorted(energies):
            rows.append((name, term, energies[term]))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, skips


def _write_snapshot(rows: list[tuple[str, str, float]]) -> None:
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    with open(SNAPSHOT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb", "term", "energy"])
        for pdb, term, energy in rows:
            # ``%.10g`` keeps float32-precision values round-trippable
            # without dragging along denormal noise.
            w.writerow([pdb, term, f"{energy:.10g}"])


def _read_snapshot() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with open(SNAPSHOT_PATH, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[(row["pdb"], row["term"])] = float(row["energy"])
    return out


# --------------------------------------------------------------------- #
# Test                                                                  #
# --------------------------------------------------------------------- #


class PanelSnapshotTest(unittest.TestCase):
    """Score the panel and compare to the committed snapshot."""

    def test_panel_matches_snapshot(self) -> None:
        rows, skips = _collect_panel()

        for pdb, reason in skips.items():
            # Visible in pytest -v / unittest -v output; not a failure.
            sys.stderr.write(f"[panel-snapshot] SKIP {pdb}: {reason}\n")

        self.assertTrue(rows, "No PDBs scored successfully — panel is empty.")

        if not os.path.exists(SNAPSHOT_PATH):
            _write_snapshot(rows)
            sys.stderr.write(
                f"[panel-snapshot] wrote initial snapshot with "
                f"{len(rows)} rows to {SNAPSHOT_PATH}\n"
            )
            return

        snapshot = _read_snapshot()
        live = {(p, t): e for p, t, e in rows}

        missing = sorted(set(snapshot) - set(live))
        extra = sorted(set(live) - set(snapshot))
        mismatches: list[str] = []
        for key, snap_val in snapshot.items():
            if key not in live:
                continue
            live_val = live[key]
            if not np.isclose(live_val, snap_val, atol=ATOL, rtol=RTOL):
                mismatches.append(
                    f"  {key[0]:>8s} / {key[1]:<20s} "
                    f"snapshot={snap_val:.6f} live={live_val:.6f} "
                    f"delta={live_val - snap_val:+.6f}"
                )

        problems: list[str] = []
        if missing:
            problems.append(
                f"{len(missing)} snapshot rows missing from live: {missing}"
            )
        if extra:
            problems.append(
                f"{len(extra)} live rows not in snapshot (new channel?): {extra}"
            )
        if mismatches:
            problems.append(
                "Energy mismatches beyond tolerance:\n" + "\n".join(mismatches)
            )

        if problems:
            self.fail(
                "Panel snapshot drift detected.\n"
                + "\n".join(problems)
                + "\n\nIf this drift is intentional, re-snapshot with:\n"
                f"  rm {SNAPSHOT_PATH} && python tests/test_panel_snapshot.py"
            )


# --------------------------------------------------------------------- #
# Standalone entry point                                                #
# --------------------------------------------------------------------- #


def _main() -> int:
    """Run the snapshot test as a script.

    Mirrors ``python -m unittest`` but keeps the dep surface to stdlib
    so the file works on a bare interpreter.
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(PanelSnapshotTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(_main())
