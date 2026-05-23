# Extra interface scores (`tmol_jax.score.extra`)

Differentiable JAX implementations of published protein-interface
scores that empirically correlate with experimental binding kinetics
and affinity but are not part of Rosetta's canonical
`beta_nov2016_cart`. We use them as additional `LossTerm` channels in
mosaic / Boltz-2 hallucination loops where they're cheap to evaluate
and add orthogonal gradient signal to the Rosetta interface term.

Motivation: on Adaptyv's de novo binder corpus (≈200 designs with
measured `k_on`, `k_off`, `K_d`), these scores together explain more
of the kinetic variance than the full Rosetta interface energy
alone:

| Score | Modality | Target | Pearson r | n |
|---|---|---|---:|---:|
| `BUH_per_kÅ²` | de novo (large) | `log k_off` | **+0.54** | 43 |
| `BUH_per_kÅ²` | de novo (large) | `log K_d` | +0.48 | 43 |
| `SS99` / `TransComp` | miniprotein | `log k_off` | **+0.44** | 58 |

Numbers from internal kinetics analysis (not yet published). For
context the strongest single Rosetta-channel correlation on the same
corpus is `dG_fa_elec` at r=+0.43 on miniprotein `log k_off`.

## Scores currently implemented

| Module | Method | Predicts | Diff. status |
|---|---|---|---|
| `ss99` | Selzer-Schreiber 1999 (PARE) shielded Coulomb | `k_on` (and `k_off` via dissociation-rate-limited regime) | C∞ |
| `transcomp` | Qin-Zhou 2011 transition-state ``k_on`` | `k_on` | C∞ (delegates to SS99 math, different ε / κ) |
| `buh` | Stranges-Kuhlman 2013 buried-unsat polars | `k_off` (Rosetta filter equivalent, smooth surrogate) | Smooth surrogate, sigmoid + softplus |
| `prodigy` | Vangone-Bonvin 2015 contact-based | `K_d` baseline | Smooth surrogate (sigmoid contact mask) |
| `dfire` | Zhang-Zhou 2005 atom-pair statistical potential | `K_d` (classical r ≈ 0.73 on PP complexes) | C¹ (bilinear bin interp); placeholder table — published coefficients not bundled |
| `kinetics_tuning` | Composite three-channel loss | `k_on`, `k_off`, `K_d` exposed as separate weight knobs | Smooth (composes the four above) |

### `ss99` — Selzer-Schreiber 1999 (PARE) electrostatic kon proxy

Reference: Selzer & Schreiber, *J. Mol. Biol.* 287, 1999.

Sums shielded Coulombic interactions between charged atom pairs
across the interface. ε = 80 fixed (no distance-dependent damping),
κ⁻¹ = 6.5 Å (I = 0.022 M), 6 Å ion-exclusion shell. Bound-state
coords stand in for the transition state.

```python
from tmol_jax.score.extra import ss99_energy
from tmol_jax.score.extra.ss99 import build_ss99_charges

charges = build_ss99_charges(res3_per_res, atom_names, atom_to_res,
                              is_chain_start, is_chain_end)
U = ss99_energy(coords, charges, chain_id_atoms, binder_chain_id=0)
# ln(k_on / k_on^0) = -U / RT;  more negative U → faster k_on
```

Cost: ~0.1 ms on CPU at 200 residues. Pure JAX, fully differentiable
as-stated.

### `transcomp` — Qin-Zhou 2011 transition-state kon

Reference: Qin & Zhou, *Biophysical Journal* 100, 2011.

Transition-state-theory ``k_on`` predictor. Mathematically nearly
identical to SS99 (Debye-Hückel shielded Coulomb), with the
canonical TransComp parameters: ε = 78.5, κ⁻¹ depends on ionic
strength. Reported r ≈ 0.85 across 49 wild-type complexes; reproduces
salt-dependence of ``k_on`` across 4 orders of magnitude.

```python
from tmol_jax.score.extra import transcomp_log_kon_enhancement

ln_ratio = transcomp_log_kon_enhancement(
    coords, charges, chain_id, binder_chain_id=0,
)
# k_on = k_on_basal * exp(ln_ratio)
```

Use TransComp at multiple ionic strengths (κ⁻¹ values) and take the
salt-dependence slope for a more discriminating signal than the
single-I energy.

### `buh` — Stranges-Kuhlman 2013 buried-unsat polars

Reference: Stranges & Kuhlman, *Protein Sci.* 22, 2013.

A buried polar atom whose Hbond capacity is unsatisfied predicts a
design failing to bind. Native interfaces sit at ≤ 2 BUH per 1000 Å²;
designs above 2-3 essentially never bind in the published benchmark.

The Rosetta filter uses non-differentiable VSASA and a hard Hbond
counter. Our surrogate replaces both with smooth ingredients:

- **Burial** via Rocklin-style sidechain-neighbor cones (continuous
  weighted count of Cα neighbors inside an angular cone aligned with
  the residue's Cα→Cβ axis). Smooth in coords; ~25% noisier than true
  VSASA but gradient-clean.
- **Satisfaction** via a *coords-only* Hbond proxy: for each polar
  atom, smooth count of nearby polar atoms of opposite type (donor↔
  acceptor) on a different residue. Fully differentiable end-to-end
  through coordinates; no precomputed hbond-energy vector needed.

```python
from tmol_jax.score.extra import buh_score, buh_per_kA2

# Coords-only path: pass polar_is_donor / polar_is_acceptor and
# the score figures out hbond satisfaction internally.
BUH = buh_score(coords, polar_atom_idx, polar_atom_res,
                polar_capacity, jnp.zeros_like(polar_capacity),
                res_ca_idx, res_cb_idx, atom_to_res,
                polar_is_donor=polar_is_donor,
                polar_is_acceptor=polar_is_acceptor)
BUH_per_kA2 = buh_per_kA2(BUH, interface_bsa)
```

Calibration caveat: absolute BUH values differ from the published
Stranges-Kuhlman 2013 counts because the burial proxy is Rocklin-cones
rather than VSASA (numerical offset ~30-50%). Use as a continuous
gradient surrogate; run the published Rosetta `BuriedUnsatHbonds`
filter as a non-differentiable post-hoc QC at ranking time.

### `prodigy` — Vangone-Bonvin 2015 contact-based ΔG

Reference: Vangone & Bonvin, *eLife* 4:e07454, 2015.

Linear model over inter-residue contacts grouped by polarity class
(charged/polar/apolar) plus non-interacting surface area (NIS)
fractions. We replace the hard 5.5 Å cutoff with a smooth sigmoid;
NIS terms taken as inputs (caller provides; needs SASA on separate
monomers).

```python
from tmol_jax.score.extra import prodigy_dG

dG = prodigy_dG(coords, atom_to_res, res_polarity, chain_id_per_res,
                nis_apolar=nis_ap, nis_charged=nis_cc,
                binder_chain_id=0)
# Lower (more negative) = tighter binding.
```

Cost: O(n_atoms²) dense pair-distance ~30 ms on CPU at 1200 atoms.

### `dfire` — Zhang-Zhou 2005 atom-pair statistical potential

Reference: Zhang, Liu, Zhou, Zhou. *J. Med. Chem.* 48, 2005.

Distance-scaled finite-ideal-gas reference statistical potential —
the original "must beat" baseline at r = 0.73 on PP complexes, still
top-3 twenty years later. Our implementation provides the
differentiable bin-interpolated pair-sum infrastructure but **ships
with a placeholder potential table**, not the published coefficients.

```python
from tmol_jax.score.extra import dfire_energy, make_placeholder_dfire_table

# Placeholder table (NOT the published Zhou-Yang coefficients):
table = make_placeholder_dfire_table()

E = dfire_energy(coords, atom_types, chain_id, table,
                 cross_chain_only=True, binder_chain_id=0)
```

To get the real signal, build the published table:

```bash
# (script not yet written — TODO)
python scripts/convert_dfire_tables.py --in dfire.dat --out dfire.npz
```

### `kinetics_tuning` — composite three-channel loss

The `k_on`, `k_off`, `K_d` axes have largely disjoint hot-residue
lists on a given complex (mutations that target one axis don't move
the others). `KineticsTuningLoss` exposes the three axes as separate
weight knobs so an affinity-tuning loop can decide which property to
move:

```python
from tmol_jax.score.extra.kinetics_tuning import KineticsTuningLoss

loss = KineticsTuningLoss(
    w_kon=0.0,   # don't touch kon
    w_koff=1.0,  # tighten koff
    w_kd=0.5,    # while not breaking Kd
    charges=ss99_charges, chain_id_atoms=chain_id_atoms,
    polar_atom_idx=polar_idx, polar_atom_res=polar_res,
    polar_capacity=polar_capacity,
    polar_is_donor=polar_is_donor, polar_is_acceptor=polar_is_acceptor,
    res_ca_idx=res_ca_idx, res_cb_idx=res_cb_idx,
    atom_to_res=atom_to_res,
    kd_fn=interface_dG_function,
)
```

Plus `per_residue_kinetic_gradient_norms` to identify which residues
are sensitive on each axis independently. See
[`examples/demo_kinetics_tuning.py`](../examples/demo_kinetics_tuning.py)
and [`examples/demo_alanine_scan_kinetics.py`](../examples/demo_alanine_scan_kinetics.py).

## Differentiability evidence

`examples/demo_extra_scores.py` runs each score on 1ubq split-binder
and verifies:

- the gradient is finite,
- a small step opposite the gradient lowers the loss (perturbation-
  direction sanity check — catches saturated-region gradients that
  finite-norm tests miss).

Run:

```bash
PYTHONPATH=. python examples/demo_extra_scores.py
```

## What we didn't implement (yet) and why

- **Shape Complementarity (Lawrence-Colman 1993).** Needs a
  differentiable surrogate for the Connolly surface and a smooth
  quantile. ~300 LOC. Deferred.
- **CoreHSEnergy (Agius 2013).** Needs either a Rosetta alanine-scan
  workflow (non-differentiable) or a SKEMPI v2-trained ΔΔG MLP.
  Deferred to a separate model release.
- **PRODIGY NIS with on-the-fly SASA.** NIS currently an input; a
  smooth-SASA estimator would close that gap (~50 LOC).
- **Neural ddG predictors (RDE-Net, BA-DDG, Pythia-PPI, TopoScorer).**
  All differentiable, all reach Spearman 0.5-0.8 on SKEMPI v2 / PPB-
  Affinity. Highest empirical bang/buck would be BA-DDG (ProteinMPNN
  + thermodynamic-cycle correction) since the ProteinMPNN backbone
  is already a fixture in BindCraft-style loops. Deferred: integrating
  a 3M-parameter weight bundle is a separate ship.

## Composition with the Rosetta interface loss

For an affinity-tuning loop these compose with `TmolInterfaceLoss`
(from `examples/integration_mosaic_hallucinator.py`) via the same
`LossTerm` algebra. Skeleton:

```python
loss = (
    1.0   * TmolInterfaceLoss(ps_tpl, scale=1e-2)       # Rosetta interface
    + 0.5   * ss99_loss(ps_tpl)                          # k_on signal
    + 1.0   * buh_loss(ps_tpl)                           # k_off signal
    + 0.01  * prodigy_loss(ps_tpl)                       # K_d baseline
    + 0.1   * transcomp_loss(ps_tpl)                     # TST k_on
    # + 0.1 * dfire_loss(ps_tpl, dfire_table=...)        # classical Kd baseline
)
```

The `LossTerm` wrappers (`SS99Loss`, `TransCompLoss`, `BUHLoss`,
`PRODIGYLoss`, `DFIRELoss`) live in
[`tmol_jax/score/extra/losses.py`](../tmol_jax/score/extra/losses.py).
They take the same atom37-coords convention as `TmolInterfaceLoss`
and compose via `+` / `*` with mosaic losses; lift back to the
tmol_jax flat layout internally via the `_atom37_map` /
`_atom37_is_mapped` mapping built by `build_pose_stack_template`.

## References

- Selzer & Schreiber 1999 — https://doi.org/10.1006/jmbi.1999.2615
- Qin & Zhou 2011 — https://doi.org/10.1016/j.bpj.2011.05.004
- Stranges & Kuhlman 2013 — https://doi.org/10.1002/pro.2187
- Vangone & Bonvin 2015 — https://elifesciences.org/articles/07454
- Zhang & Zhou 2005 (DFIRE) — https://pubmed.ncbi.nlm.nih.gov/15801826/
- Lawrence & Colman 1993 — https://doi.org/10.1006/jmbi.1993.1648
- Agius, Torchala et al. 2013 — https://doi.org/10.1371/journal.pcbi.1003216
- Rocklin sidechain-neighbor cones — Rocklin et al., *Science* 357, 2017
- RDE-Net — https://openreview.net/forum?id=_X9Yl1K2mD
- Pythia-PPI — Tao et al. 2025
- BA-DDG / BA-Cycle — https://arxiv.org/pdf/2410.09543 / https://arxiv.org/pdf/2507.05502
- TopoScorer / Specter — https://openreview.net/pdf?id=QNcrdCKNa5
- PPI-Graphomer — https://pmc.ncbi.nlm.nih.gov/articles/PMC12042501/
- PPB-Affinity dataset — https://preview-www.nature.com/articles/s41597-024-03997-4
