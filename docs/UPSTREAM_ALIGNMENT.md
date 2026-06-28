# Upstream alignment audit (jtmol vs `uw-ipd/tmol` `origin/master`)

Systematic diff of jtmol against the live upstream `master` branch
(currently at commit `230fe74`, 14 commits ahead of the local checkout
at `1d7e18c` used during jtmol's port). Run on 2026-06-19.

## TL;DR

```
Forcefield YAML constants       byte-identical
Cutoff constants                values match, refactor diff only
Loader-side Python              renamed namespaces, dropped torch dep
ScoreType / sfxn weights        aligned (gen_torsions ported, see #1)
ChemicalDatabase                aligned for protein-only scope
```

`gen_torsions` (upstream PR #361) is now ported; on the protein-only
1ubq panel it evaluates to 0.0 which matches the upstream gold value.
See [#1 below](#1-gen_torsions-ported) for the port surface.

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

### 1. `gen_torsions` ported

Upstream PR [#361 "Kdidi/ligand clean"](https://github.com/uw-ipd/tmol/pull/361)
(merged 2026-06-26) added a new `gen_torsions` channel scoring proper
+ improper dihedrals via **generic atom-type parameters**. Ported in
`tmol_jax/score/genbonded/` and `tmol_jax/database/scoring/genbonded.py`;
YAML copied verbatim from upstream to
`tmol_jax/database/default/scoring/genbonded.yaml`.

**1ubq value: 0.0**, matches upstream's gold
(`tmol/tests/score/test_score_function.py::test_score_function_all_score_types`
asserts `ScoreType.gen_torsions: n([0.0])` on `ubq_pdb`).

Why the value is zero on a pure protein:
upstream's `genbonded.yaml` `atoms:` section only carries a hierarchy
for generic ligand chem types (`CS`, `CD`, `Nad`, ...). Canonical
protein residues remain typed with Rosetta names (`Nbb`, `CAbb`,
`CObb`, `Hapo`, `CH3`, ...), types that have no fall-back entry. The
hierarchy walk yields just `[<self>]`, no torsion entry's atom tuple
ever matches, every lookup is a hash-miss, every subgraph drops. The
term is implemented for ligands and any future YAML extension; on the
protein-only panel it correctly contributes zero just like upstream.

#### What's in the port

| File | Role |
|------|------|
| `tmol_jax/score/score_types.py` | `ScoreType.gen_torsions = 22` (sentinel bumped to 23) |
| `tmol_jax/score/__init__.py` | `sfxn.set_weight(ScoreType.gen_torsions, 1.0)` |
| `tmol_jax/score/score_function.py` | Registry entry routing to `GenBondedEnergyTerm` |
| `tmol_jax/database/scoring/genbonded.py` | Loader + atom-hierarchy + bond-bin lookup (`find_torsion_params`, `find_improper_params`); port of upstream's attrs-based file, no cattr |
| `tmol_jax/database/default/scoring/genbonded.yaml` | Verbatim copy from upstream master |
| `tmol_jax/score/genbonded/potentials.py` | `gen_proper_E` (bare-cosine 4-period + negative-K lift + offset), `gen_improper_E` (harmonic) |
| `tmol_jax/score/genbonded/genbonded_energy_term.py` | Setup: per-block-type subgraph enumeration (reuses `cartbonded._find_intra_subgraphs` + a 3-neighbor improper enumeration) and DB resolution with Rosetta `calculate_offset` floor-zeroing; eval: pure-JAX gather + reduce |

#### Deliberate simplifications

- Default `bond_type = SINGLE`, `is_ring = False` (btidx 1) for every
  central bond. jtmol's chemical DB does not currently carry per-bond
  order or ring metadata. 239/263 upstream torsion entries use the `~`
  wildcard bond so this is fine in practice; only a handful of `:`
  (aromatic), `@` (any-ring), `=`, `#` entries would gain accuracy
  from real ring detection. Upgrade path: extend `Atom`/bond schema +
  thread `bond_type_int` / `is_ring` through `_subgraphs_for_block_type`.
- Inter-residue proper torsions (4-atom subgraphs spanning a peptide
  bond) are now emitted via `_inter_residue_subgraphs_for_pair` in
  `genbonded_energy_term.py`, mirroring the cartbonded helper but with
  atom-type lookup (`GenBondedDatabase.find_torsion_params`) instead of
  atom-name lookup. On 1ubq the walker enumerates ~92k candidate
  cross-block 4-atom tuples (across the full block-type palette) and
  ~75 actual peptide-bond edges; all currently drop on the
  protein-atom-type hash miss so `gen_torsions` stays at 0.0 -- same
  zero-contribution argument as the intra-block case. Inter-block
  impropers are not emitted because an improper is intrinsically rooted
  at a single 3-neighbour center on one residue.

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

### 3. Cutoff constants: values match, refactor diff only

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
| `database/scoring/cartbonded.py` | Drop `cattr`, drop `pandas`, type-annotation hygiene | None (same YAML schema) |
| `database/scoring/dunbrack_libraries.py` | Prefer `.npz` over `.bin` for torch-optional loading | jtmol uses the same `.npz`-preferred path |
| `database/scoring/elec.py` | Rename `GlobalParams` → `ElecGlobalParams` | None (namespace hygiene) |
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
