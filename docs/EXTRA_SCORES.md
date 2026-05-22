# Extra interface scores (`tmol_jax.score.extra`)

Differentiable JAX implementations of published protein-interface
scores that empirically correlate with experimental binding kinetics
and affinity but are not part of Rosetta's canonical
`beta_nov2016_cart`. We use them as additional `LossTerm` channels in
mosaic / Boltz-2 hallucination loops where they're cheap to evaluate
and add orthogonal gradient signal to the Rosetta interface term.

The motivation: on Adaptyv's internal kinetics corpus (≈200 de novo
binders + miniproteins from public competitions, with measured
`k_on`, `k_off`, `K_d`), these three scores together explain more of
the kinetic variance than the full Rosetta interface energy alone:

| Score | Modality | Target | Pearson r | n |
|---|---|---|---:|---:|
| `BUH_per_kÅ²` | de novo (large) | `log k_off` | **+0.54** | 43 |
| `BUH_per_kÅ²` | de novo (large) | `log K_d` | +0.48 | 43 |
| `SS99` | miniprotein | `log k_off` | **+0.44** | 58 |
| `binder_net_q` | miniprotein | `log K_d` | −0.41 | 58 |

Numbers from the Adaptyv kinetics analysis pipeline (not yet
published; cited from internal correlations as of 2026-05). For
context, the strongest single Rosetta-channel correlation on the same
corpus is `dG_fa_elec` at r=+0.43 on miniprotein `log k_off`.

## Scores currently implemented

### `ss99` — Selzer–Schreiber 1999 (PARE)

Reference: Selzer & Schreiber, *J. Mol. Biol.* 287, 1999. The
canonical electrostatic-only `k_on` predictor. Sums shielded Coulombic
interactions between charged atom pairs across the interface using a
fixed solvent dielectric (ε = 80, no distance-dependent damping) and
a Debye–Hückel ion-screening factor (κ⁻¹ = 6.5 Å for I = 0.022 M).
Bound-state coords stand in for the transition state.

```python
from tmol_jax.score.extra import ss99_energy
from tmol_jax.score.extra.ss99 import build_ss99_charges

charges = build_ss99_charges(res3_per_res, atom_names, atom_to_res,
                              is_chain_start, is_chain_end)
U = ss99_energy(coords, charges, chain_id_atoms, binder_chain_id=0)
# ln(k_on / k_on^0) = -U / RT;  more negative U → faster k_on
```

Cost: ~0.1 ms on CPU for a 200-residue complex (≈40 charged atoms /
side, ~1600 cross-pairs). Pure JAX, no surrogates, no soft
thresholds — differentiable as-stated.

### `buh` — Stranges–Kuhlman 2013 buried-unsat polars (smooth surrogate)

Reference: Stranges & Kuhlman, *Protein Sci.* 22, 2013. A buried polar
atom whose Hbond capacity is unsatisfied predicts a design failing to
bind. Native interfaces sit at ≤ 2 BUH per 1000 Å²; designs above 2-3
essentially never bind in their benchmark.

The Rosetta filter uses non-differentiable VSASA and a hard hbond
counter. Our surrogate keeps the spirit but replaces both with smooth
ingredients:

- **Burial** via Rocklin-style sidechain-neighbor cones (continuous
  weighted count of Cα neighbors inside an angular cone aligned with
  Cα→Cβ axis). Smooth in coords; ~25% noisier than true VSASA, but
  gradient-clean.
- **Satisfaction** via the per-atom Hbond-energy sum, softplus-shaped
  so each Hbond contributes ≈ |E_hb| when E_hb < 0, ≈ 0 when E_hb ≥ 0.
  Compared smoothly to per-atom capacity (1 for backbone N/O, 2 for
  OH groups).

```python
from tmol_jax.score.extra import buh_score, buh_per_kA2

BUH = buh_score(coords, polar_atom_idx, polar_atom_res,
                polar_capacity, polar_hb_energies,
                res_ca_idx, res_cb_idx, atom_to_res)
BUH_per_kA2 = buh_per_kA2(BUH, interface_bsa)
```

Cost: O(n_res²) pairwise cones + O(n_polar) score reduction; ~3 ms on
CPU at 200 residues.

**Calibration caveat.** Absolute BUH values differ from Stranges-Kuhlman 2013
counts because the burial proxy is Rocklin-cones rather than true VSASA
(numerical offset ~30-50%). Use this as a continuous gradient
surrogate; run the published Rosetta `BuriedUnsatHbonds` filter as a
non-differentiable post-hoc QC at ranking time.

### `prodigy` — Vangone–Bonvin 2015 contact-based ΔG

Reference: Vangone & Bonvin, *eLife* 4:e07454, 2015. Linear model
over inter-residue contacts grouped by polarity class
(charged/polar/apolar) plus non-interacting surface area (NIS)
fractions.

The published model uses a hard 5.5 Å heavy-atom distance cutoff;
we replace with a smooth sigmoid so `jax.grad` flows. The NIS terms
need a SASA estimator on the separate monomers, which is non-trivial
in pure JAX; we expose them as inputs so you can pass a static
estimate from template-build time.

```python
from tmol_jax.score.extra import prodigy_dG

dG = prodigy_dG(coords, atom_to_res, res_polarity, chain_id_per_res,
                nis_apolar=nis_ap, nis_charged=nis_cc,
                binder_chain_id=0)
# Lower (more negative) = tighter binding. RMSE ~1.9 kcal/mol on the
# published benchmark, r ≈ 0.73 on training / 0.31 on held-out.
```

Cost: O(n_atoms²) dense pair-distance ~30 ms on CPU at 1200 atoms
(could be cut to ~3 ms with a neighbor-list path; not yet wired).

**Surrogate caveat.** Our atom-pair sigmoid surrogate over-counts
relative to the published residue-pair Boolean. We apply a fixed
1/36 rescaling to land in the right ballpark; for hard PRODIGY
ranking, use the original tool (https://github.com/haddocking/prodigy)
externally.

## What we didn't implement (yet)

- **Shape Complementarity (Lawrence-Colman 1993).** Needs a
  differentiable surrogate for the Connolly surface and a smooth
  quantile over surface-dot normals. Reasonable path is Gaussian-blur
  isosurface + softmin nearest-neighbor + smoothed quantile (~300
  LOC); validate against the Rust `sc-rs` reference. Deferred.
- **CoreHSEnergy (Agius 2013).** Sum of single-point ΔΔG_Ala over
  core hotspot residues. Requires either (a) running a Rosetta
  alanine scan per design (non-differentiable, ~150 s/complex),
  or (b) training a small differentiable ΔΔG MLP on SKEMPI v2.
  Deferred to a separate model release.
- **PRODIGY NIS terms with true on-the-fly SASA.** The current
  implementation takes NIS as an input. A fully self-contained
  PRODIGY needs a smooth SASA estimator (e.g. Rocklin cones
  inverted), which would close that gap with ~50 LOC.

## Composition with the Rosetta interface loss

These scores compose with `TmolInterfaceLoss` (from
`examples/integration_mosaic_hallucinator.py`) via the same
`LossTerm` algebra. For an affinity-tuning loop:

```python
loss = (
    1.0   * TmolInterfaceLoss(ps_tpl, scale=1e-2)       # Rosetta interface
    + 0.1   * ss99_energy_loss(ps_tpl)                   # k_on signal
    + 1.0   * buh_loss(ps_tpl)                           # k_off signal
    + 0.01  * prodigy_loss(ps_tpl)                       # K_d baseline
)
```

(`LossTerm` wrappers for the three scores are not yet committed;
the score functions themselves are in `tmol_jax.score.extra` and
can be used directly inside any `eqx.Module` loss class.)

## Reproducer

```bash
PYTHONPATH=. python examples/demo_extra_scores.py
```

Runs each of the three scores on 1ubq split into binder (res 1-40)
and target (res 41-76), and verifies that `jax.grad` is finite on
each.

## References

- Selzer & Schreiber 1999 — https://doi.org/10.1006/jmbi.1999.2615
- Stranges & Kuhlman 2013 — https://doi.org/10.1002/pro.2187
- Vangone & Bonvin 2015 — https://elifesciences.org/articles/07454
- Lawrence & Colman 1993 — https://doi.org/10.1006/jmbi.1993.1648
- Agius, Torchala et al. 2013 — https://doi.org/10.1371/journal.pcbi.1003216
- Rocklin sidechain-neighbor cones — Rocklin et al., *Science* 357, 2017
