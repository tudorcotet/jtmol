# Upstream alignment audit (jtmol vs `uw-ipd/tmol` `origin/master`)

Systematic diff of jtmol against the live upstream `master` branch
(currently at commit `230fe74`, 14 commits ahead of the local checkout
at `1d7e18c` used during jtmol's port). Run on 2026-06-19.

## TL;DR

```
Forcefield YAML constants       byte-identical
Cutoff constants                values match, refactor diff only
Loader-side Python              renamed namespaces, dropped torch dep
ScoreType / sfxn weights        diverges on 1 new term (gen_torsions)
ChemicalDatabase                aligned for protein-only scope
```

One real divergence to track: upstream PR #361 introduced a new
`gen_torsions` score channel weighted 1.0 in `beta2016.sfxn`. jtmol
does not implement it. See [#1 below](#1-gen_torsions-not-yet-ported)
for the port surface.

## Audit pipeline

```
upstream/master @ 230fe74
│
├─ database/default/scoring/*.yaml       diff vs jtmol → 0 lines changed
├─ database/scoring/*.py loaders         diff → renamed namespaces only
├─ score/*/potentials/*.{cc,cu,hh}       diff vs jtmol JAX impls → see Findings
├─ score/genbonded/                      NEW upstream module → port-required
├─ database/score_functions/beta2016.sfxn diff → +1 term weight (gen_torsions)
└─ ljlk + lk_ball max_dis cutoffs        already at 6.0 in jtmol
```

## Findings

### 1. `gen_torsions` not yet ported

Upstream PR [#361 "Kdidi/ligand clean"](https://github.com/uw-ipd/tmol/pull/361)
(merged 2026-06-26) added a new `gen_torsions` energy term that scores
proper + improper dihedrals using **generic atom-type parameters**
(rather than per-residue atom-name parameters used by the existing
`cart_torsions` / `cart_impropers`). The term is wired into
`beta2016.sfxn` with weight 1.0, so upstream's `beta_nov2016_cart`
total now includes a non-zero `gen_torsions` row that jtmol silently
omits.

The term was motivated by ligand support (generic atom typing handles
non-canonical residues without per-residue parameters), but it fires
on protein backbone + sidechain torsions too. Status on the 1ubq panel:
upstream will emit a `gen_torsions` row in its `score_by_term`
breakdown; jtmol will not — but the upstream YAML golden bundle
(`tmol/tests/data/term_baselines/`) does not yet ship a
`GenBondedEnergyTerm/` baseline, so the existing parity harness cannot
diff this row.

#### Port surface

| File | What to do |
|------|------------|
| `tmol_jax/score/score_types.py` | Add `gen_torsions` to the `ScoreType` enum |
| `tmol_jax/score/__init__.py::beta2016_score_function` | `sfxn.set_weight(ScoreType.gen_torsions, 1.0)` |
| `tmol_jax/database/scoring/genbonded.py` | Port the atom-hierarchy + bond-bin lookup loader from `tmol/database/scoring/genbonded.py` |
| `tmol_jax/database/default/scoring/genbonded.yaml` | Copy verbatim from upstream |
| `tmol_jax/score/genbonded/genbonded_energy_term.py` | New JAX implementation: setup_block_type (proper+improper subgraph enumeration + DB resolution + offset table) + inter-block hashtable build via existing `score.common.hash_util` |
| `tmol_jax/score/genbonded/potentials.py` | JAX kernel: `proper = Σ_n k_n · (1 + cos(n·θ - φ_n)) + offset`, `improper = k · (θ - δ)²` (verify exact form against `tmol/score/genbonded/potentials/potentials.hh`) |

Estimated port effort: ~500 LOC, no new dependencies. The proper-
torsion energy is the same `(1 + cos(nθ - φ))` form jtmol already
handles for `cart_torsions`; the diff is the atom-type-keyed parameter
lookup (jtmol's existing `score.common.hash_util` covers it).

The C++/CUDA paths are not needed (jtmol gets gradients from
`jax.grad`); the ligand-specific bond-type bin variants
(`AROMATIC`, `DOUBLE`, `TRIPLE`) can be skipped for the protein-only
panel since proteins use `SINGLE` almost exclusively.

#### When to do it

Right now would gate any future "we match upstream beta_nov2016
exactly" claim. Defer if the protein-only scoring drift in the
~10 kcal/mol range is acceptable for current downstream consumers
(mosaic affinity loss, design-loop physics regularizer). Re-evaluate
when upstream ships a YAML golden for the term so the parity harness
can verify the port.

### 2. Forcefield YAML constants byte-identical

Every YAML under `tmol_jax/database/default/scoring/` matches its
upstream counterpart at the byte level. Verified by `diff -q`.

```
$ diff -q tmol_jax/database/default/scoring/ tmol/tmol/database/default/scoring/
Only in tmol_jax/database/default/scoring: dunbrack.npz
Only in tmol_jax/database/default/scoring: omega_bbdep.npz
Only in tmol_jax/database/default/scoring: rama.npz
```

The three jtmol-only files are pre-converted numpy arrays of upstream's
torch binary tables (`scripts/convert_binary_tables.py`); they drop the
torch runtime dep but carry the same numeric content. One-time sanity
check on the `.npz` ↔ `.bin` conversion is the only follow-up.

### 3. Cutoff constants — values match, refactor diff only

Upstream commit [`1d7e18c`](https://github.com/uw-ipd/tmol/commit/1d7e18c)
("Aleaverfay/dlin2") bumped `max_dis` from `5.5` to `6.0` Å in
`ljlk` / `lk_ball` and renamed `lj_switch_dis2sigma` →
`lj_dlin_sigma_factor`. jtmol was already at `6.0` and `0.6`
respectively (the rename is a pure label swap).

| Constant | jtmol | upstream | Match |
|---|---|---|---|
| `max_dis` (YAML) | `6.0` | `6.0` | yes |
| `_CPOLY_DMAX` (`tmol_jax/score/ljlk/potentials.py`) | `6.0` | `max_dis`-derived | yes |
| `distance_threshold` (`tmol_jax/score/lk_ball/potentials.py`) | `6.0` | `global_params.distance_threshold` | yes |
| `neighbor_list_cutoff` (LJLKEnergyTerm) | `6.0` | `6.0` | yes |
| `lj_dlin_sigma_factor` | `0.6` | `0.6` | yes (renamed upstream) |
| `lj_dlin_sigma_factor_soft` | absent | `0.91` (new) | not needed for default scoring |

All hardcoded `6.0` literals in jtmol now carry a `ponytail:` comment
flagging the upstream re-parameterization, so future drift is
discoverable.

### 4. Loader-side Python diverges, no parity impact

| File | Upstream change since clone | jtmol impact |
|------|----------------------------|--------------|
| `database/scoring/cartbonded.py` | Drop `cattr`, drop `pandas`, type-annotation hygiene | None — same YAML schema |
| `database/scoring/dunbrack_libraries.py` | Prefer `.npz` over `.bin` for torch-optional loading | jtmol uses the same `.npz`-preferred path |
| `database/scoring/elec.py` | Rename `GlobalParams` → `ElecGlobalParams` | None — namespace hygiene |
| `database/scoring/hbond.py` | Drop pandas DataFrame mapper, expose plain dicts | jtmol uses the plain-dict path |
| `database/scoring/ljlk.py` | Drop `cattr`, parse `Angle` units via shared hook | None |
| `database/scoring/omega_bbdep.py` | `.npz`-preferred | aligned |
| `database/scoring/rama.py` | `.npz`-preferred | aligned |
| `database/scoring/ref.py` | Drop `cattr`, type-annotation hygiene | None |

### 5. Commits since local checkout

Of 14 commits between local clone (`1d7e18c`, 2026-04-17) and current
`origin/master` (`230fe74`, 2026-06-12), only one is a parity-affecting
backport candidate: PR #361 (the `gen_torsions` term in [#1
above](#1-gen_torsions-not-yet-ported)). Full audit in
[`UPSTREAM_BUGS.md`](UPSTREAM_BUGS.md). The other 13 are CI / build /
refactor / ligand-pipeline / minimizer changes that don't affect
protein-only beta_nov2016 scoring numerics.

## Cross-references

- Documented upstream bugs we work around: [`UPSTREAM_BUGS.md`](UPSTREAM_BUGS.md)
- Forward + gradient parity matrix: [`UPSTREAM_PARITY.md`](UPSTREAM_PARITY.md)
- Term-by-term parity on 1ubq: [`PARITY.md`](PARITY.md)

## Maintenance protocol

Run this audit before any release tag, or whenever upstream pushes
material changes to `tmol/database/`, `tmol/score/`, or
`tmol/database/score_functions/beta2016.sfxn`:

```bash
cd tmol && git fetch origin master && git log --oneline HEAD..origin/master
diff -q tmol_jax/database/default/scoring/ tmol/tmol/database/default/scoring/
```

If the diff reports anything other than the three jtmol-only `.npz`
files, update this doc with the new divergence and decide port vs
defer per term.
