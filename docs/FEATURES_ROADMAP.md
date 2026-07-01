# jtmol feature roadmap: importing new chemistry from the Rosetta lineage

How new energy-function chemistry (metals, post-translational
modifications, nucleic acids) gets into jtmol, and the plan per feature.

## The lineage: Rosetta -> tmol -> jtmol

```
Rosetta (C++ + database/)  ──ported subset──►  tmol (PyTorch/CUDA ops + YAML)  ──re-port──►  jtmol (JAX kernels + YAML)
   beta_nov2016_cart                            uw-ipd/tmol                                    tmol_jax
   full residue-type sets                       protein-only subset                           + genbonded (gen_torsions)
   full database params                         hand-copied YAML                              byte-similar YAML
```

- **Rosetta** holds the full energy function and every residue-type set
  (protein, nucleic, metal, ligand) plus the parameter database.
- **tmol** ported only the `beta_nov2016_cart` protein terms as compiled
  ops, transcribing the params it needed into
  `tmol/tmol/database/default/scoring/*.yaml`. It has **no** nucleic /
  PTM / metal terms or residues.
- **jtmol** re-ported tmol's terms as pure-JAX kernels reading the same
  YAML, and added the `genbonded` term. Same protein-only scope.

Provenance example (ljlk): Rosetta `database/scoring/score_functions/etable/`
atom properties -> tmol `ljlk.yaml` `atom_type_parameters` -> jtmol
`tmol_jax/database/default/scoring/ljlk.yaml` -> JAX kernels in
`tmol_jax/score/ljlk/`.

## Parameter policy for NEW features: make our own

Rosetta's source is publicly readable but **not OSI-open** (commercial
use of its database requires a UW license). jtmol's existing params
arrived via tmol under that arrangement. Rather than transcribe
Rosetta's licensed database values for new chemistry, jtmol derives its
**own** parameters from public physical sources:

- Metal LJ: published 12-6 nonbonded models for divalent cations
  (e.g. Li-Merz-style radius/well-depth sets), or Shannon ionic radii ->
  sigma with standard well depths. Charges are formal (+2, +1).
- Phosphate (PTM): phosphate-group LJ/charge from public force fields
  (AMBER/CHARMM phosphate parameters are widely published).
- Nucleic: base/sugar/phosphate params from public NA force fields.

Every derived parameter carries a source comment. This keeps jtmol
license-clean and, because everything is differentiable, leaves the door
open to *fitting* these params to experimental data later rather than
inheriting fixed literature values.

## The import mechanism: two paths

**Path A - data only (no new kernel).** If a new residue scores through
terms jtmol already has (LJ/LK, elec, hbond, cartbonded, gen_torsions),
it needs only: (1) atom-type definitions in `chemical.yaml`, (2) LJ/LK
rows in `ljlk.yaml`, (3) elec charges, (4) a residue-type definition.
jtmol's **ligand params mechanism is exactly this seam** and is proven
(28/28 atom typing + full LJ/LK/elec/hbond scoring on cdk2):
`tmol_jax/ligand/params_file.py` + `params_io.extend_db_from_yaml`
inject custom atom types + a `RawResidueType`, consumed via
`build_context_from_biotite(ligand_params_files=[...])`.

**Path B - new score term.** If the chemistry needs physics jtmol
doesn't have (nucleic sugar pucker, base stacking), a new JAX score
term + its potential must be written, mirroring an existing term's
structure under `tmol_jax/score/<name>/`.

## Per-feature plan

| Feature | Path | New kernel? | Effort | Branch |
|---|---|---|---|---|
| **Metals** (Zn, Mg, Ca) | A (data only) | No | Low | `feature/metals` |
| **PTMs** (pSer/pThr/pTyr) | A (data only) | No | Low-Med | `feature/ptm` |
| **DNA / RNA** | B (new terms) | Yes (rna_torsion, rna_sugar_close, fa_stack) | High | `feature/dna-rna` |

### `feature/metals` — first target

Metal ions are single-atom residues that score through the existing
LJ/LK + elec terms; **zero new kernels**. Smallest real win, and it
unlocks metalloprotein / zinc-finger binder scoring (directly relevant
to de novo design). Adds:

- `chemical.yaml`: element rows for Zn/Mg/Ca/Fe (missing today) + atom
  types `Zn2p`/`Mg2p`/`Ca2p`; single-atom residue types ZN/MG/CA.
- `ljlk.yaml`: LJ/LK rows (our own, from published ionic-radius models).
- `elec.yaml`: formal-charge rows (+2).
- A `.tmol`-style fixture per ion + a smoke test (load a metal, confirm
  it enters the PoseStack, score finite).

Open wrinkle: a single-atom residue has no internal frame for the icoor
tree; needs a virtual-root handling path (or the builder's 1-atom case).

### `feature/ptm` — second

pSer/pThr/pTyr are canonical residues plus a phosphate group. Either
patch-expanded residue types or standalone `.tmol` restypes with the
phosphate atoms (P, O1P/O2P/O3P) typed and charged. Scores through
existing LJ/LK + elec + cartbonded + gen_torsions; no new kernel. Needs
~15 cartbonded ideal-geometry params per restype (our own, from standard
phosphate geometry).

### `feature/dna-rna` — separate project

The only feature needing new physics. Requires: nucleotide residue-type
sets (backbone + sugar + 4-5 bases each for DNA/RNA), their atom types,
AND new score terms (`rna_torsion`, `rna_sugar_close`, `fa_stack`, base
pairing). The shared etable terms (fa_atr/rep/sol) apply to NA atoms out
of the box, but the sugar-pucker / stacking physics are new JAX kernels.
Scope as its own multi-step effort after the data-only path is proven on
metals + PTMs.
