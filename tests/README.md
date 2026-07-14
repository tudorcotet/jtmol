# jtmol top-level tests

Tests in this directory exercise the public API end-to-end across
multiple PDBs. Per-term math gradients and upstream-tmol parity live
under `jtmol/tests/` (covered by `pytest.ini`'s default
`testpaths`); this directory holds the integration-level regression
checks that don't depend on the upstream tmol clone for *correctness*,
only for PDB inputs.

Run any test either via pytest or standalone:

```bash
PYTHONPATH=. python -m pytest tests/test_panel_snapshot.py -v
# or
PYTHONPATH=. python tests/test_panel_snapshot.py
```

## `test_panel_snapshot.py` — 22-channel regression snapshot

Loads a 5-PDB panel (`1ubq`, `1qys`, `1r21`, `3plc`, `6DMZ_A`), scores
each with `beta2016_score_function` at unit weights on every active
channel (the 22 beta2016 channels, **including `gen_torsions`**), and
records every per-term energy in `snapshots/panel_energies.csv`.

* **First run** (no snapshot present): writes the CSV and exits OK.
* **Subsequent runs**: re-scores and asserts every value matches the
  committed CSV to `atol=1e-3, rtol=1e-4`. Drift in any term, a new
  channel, or a vanished channel all fail the test with a per-row
  diff.
* **JIT smoke check**: after the eager pass, the score function is
  `jax.jit`-compiled and called twice; both calls must agree
  bit-for-bit (no JAX nondeterminism) and match the eager total to
  float32 tolerance.
* **PDBs that fail to load** (e.g. upstream's broken `1BL8`) are
  reported as skips on stderr and excluded from the snapshot — they
  do not fail the run.

### Re-snapshot when an energy change is intentional

```bash
rm tests/snapshots/panel_energies.csv
PYTHONPATH=. python tests/test_panel_snapshot.py   # writes new snapshot
git diff tests/snapshots/panel_energies.csv         # review the deltas
git add tests/snapshots/panel_energies.csv
git commit -m "re-snapshot: <reason>"
```

The committed CSV is the source of truth. If the numbers change for a
defensible reason (term rewrite, parameter update, etc.), the snapshot
must be updated explicitly in the same commit so reviewers can see the
delta.

## `test_upstream_parity.py`, `test_multi_pdb_parity.py`

Term-by-term comparison of jtmol's outputs against upstream tmol's
YAML goldens. These require the upstream `tmol/` clone to be present
at the repo root (gitignored) and are heavier than the snapshot test;
status is summarised in `docs/PARITY.md` and `docs/MULTI_PDB_PARITY.md`.
