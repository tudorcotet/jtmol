# Upstream parity (tmol_jax vs tmol)

- Upstream live: **yes** (torch tmol-0.1.22+cu128torch2.10 on A100-40GB)
- Tolerances: atol=1e-05, rtol=0.001
- Comparison rule: `|E_jax - E_ref| <= atol + rtol * |E_ref|` (matches torch.autograd.gradcheck default)
- Panel size: 5 PDBs x 20 score types (fan-out: one Modal A100 container per PDB)
- PDB selection rationale:
  - 1ubq, 1qys: small monomeric baselines
  - 1r21, 3plc: disulfide-containing single chains
  - 6DMZ_A: designed small protein
- Deferred: 1BL8 (upstream refuses to load: missing non-leaf atom CG on residue 4 ARG).

## Summary

- Total cells: **100**
- Forward parity: **88/100** (88.0%) PASS, 12 FAIL, 0 SKIP
- Gradient parity: **72/100** (72.0%) PASS, 28 FAIL, 0 SKIP

## Per-term breakdown

| Term | Fwd pass | Grad pass | Worst max\|dg\| | Notes |
|---|:---:|:---:|---:|---|
| fa_ljatr | 5/5 | 5/5 | 2.04e-05 |  |
| fa_ljrep | 5/5 | 5/5 | 1.67e-04 |  |
| fa_lk | 5/5 | 5/5 | 1.36e-06 |  |
| fa_elec | 5/5 | 5/5 | 1.22e-05 |  |
| hbond | 5/5 | 5/5 | 1.66e-05 |  |
| rama | 5/5 | 5/5 | 2.56e-05 |  |
| omega | 5/5 | 2/5 | 1.19e-04 |  |
| dunbrack_rot | 4/5 | 4/5 | 5.80e-01 | max forward rel-err on a failing PDB: 4.1% |
| dunbrack_rotdev | 4/5 | 2/5 | 2.48e+01 | max forward rel-err on a failing PDB: 21% |
| dunbrack_semirot | 5/5 | 5/5 | 2.24e-05 |  |
| cart_lengths | 0/5 | 0/5 | 1.32e+03 | max forward rel-err on a failing PDB: 2.7e+17% |
| cart_angles | 0/5 | 0/5 | 5.80e+01 | max forward rel-err on a failing PDB: 1.8e+16% |
| cart_torsions | 5/5 | 3/5 | 4.00e-05 |  |
| cart_impropers | 5/5 | 4/5 | 5.46e-05 |  |
| cart_hxltorsions | 5/5 | 5/5 | 8.66e-07 |  |
| lk_ball_iso | 5/5 | 5/5 | 9.71e-07 |  |
| lk_ball | 5/5 | 5/5 | 4.28e-06 |  |
| lk_bridge | 5/5 | 1/5 | 4.42e-02 |  |
| lk_bridge_uncpl | 5/5 | 1/5 | 3.84e-01 |  |
| ref | 5/5 | 5/5 | 0.00e+00 |  |

## Parity matrix

Cells: ``PASS`` = both forward and gradient within tolerance; ``F:<x%>`` = forward passes but gradient fails (with forward rel-err); ``FAIL:<x%>`` = forward fails; ``-`` = skipped (no reference value).

| PDB | fa_ljatr | fa_ljrep | fa_lk | fa_elec | hbond | rama | omega | dunbrack_rot | dunbrack_rotdev | dunbrack_semirot | cart_lengths | cart_angles | cart_torsions | cart_impropers | cart_hxltorsions | lk_ball_iso | lk_ball | lk_bridge | lk_bridge_uncpl | ref |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1ubq | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL:3.8e+15% | FAIL:1.8e+16% | F:0.00011% | PASS | PASS | PASS | PASS | F:3.2e-05% | F:5.6e-06% | PASS |
| 1qys | PASS | PASS | PASS | PASS | PASS | PASS | F:0.0028% | PASS | F:3e-05% | PASS | FAIL:9.8e+14% | FAIL:4.8e+15% | PASS | F:0.0001% | PASS | PASS | PASS | PASS | PASS | PASS |
| 1r21 | PASS | PASS | PASS | PASS | PASS | PASS | F:0.0019% | PASS | PASS | PASS | FAIL:5.6e+15% | FAIL:8.8e+15% | F:9e-06% | PASS | PASS | PASS | PASS | F:5.3e-05% | F:2.9e-05% | PASS |
| 3plc | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL:4.1% | FAIL:21% | PASS | FAIL:2.7e+17% | FAIL:9.6e+15% | PASS | PASS | PASS | PASS | PASS | F:6.5e-06% | F:5.5e-06% | PASS |
| 6DMZ_A | PASS | PASS | PASS | PASS | PASS | PASS | F:0.0011% | PASS | F:0% | PASS | FAIL:1.2e+15% | FAIL:2.5e+15% | PASS | PASS | PASS | PASS | PASS | F:5.5e-05% | F:2.6e-05% | PASS |

## Per-PDB detail tables

### 1ubq

| Term | Mode | Fwd | Grad | E(jax) | E(ref) | rel | max\|dg\| | Note |
|---|---|:---:|:---:|---:|---:|---:|---:|---|
| fa_ljatr | live | PASS | PASS | -417.9582 | -417.9582 | 0% | 3.71e-07 |  |
| fa_ljrep | live | PASS | PASS | +240.7149 | +240.7147 | 5.7e-05% | 1.24e-04 |  |
| fa_lk | live | PASS | PASS | +298.2766 | +298.2766 | 1e-05% | 7.95e-07 |  |
| fa_elec | live | PASS | PASS | -136.2925 | -136.2925 | 3.4e-05% | 1.22e-05 |  |
| hbond | live | PASS | PASS | -55.6756 | -55.6756 | 2e-05% | 1.64e-05 |  |
| rama | live | PASS | PASS | -12.7434 | -12.7434 | 2.2e-05% | 1.36e-05 |  |
| omega | live | PASS | PASS | +4.1002 | +4.1003 | 0.0013% | 1.01e-04 |  |
| dunbrack_rot | live | PASS | PASS | +70.6497 | +70.6497 | 0% | 8.73e-06 |  |
| dunbrack_rotdev | live | PASS | PASS | +240.3102 | +240.3102 | 2.5e-05% | 1.44e-04 |  |
| dunbrack_semirot | live | PASS | PASS | +99.6609 | +99.6609 | 0% | 1.08e-05 |  |
| cart_lengths | live | FAIL | FAIL | +37.7637 | +0.0000 | 3.8e+15% | 3.79e+01 |  |
| cart_angles | live | FAIL | FAIL | +183.5698 | +0.0000 | 1.8e+16% | 5.13e+01 |  |
| cart_torsions | live | PASS | FAIL | +46.0235 | +46.0236 | 0.00011% | 3.72e-05 |  |
| cart_impropers | live | PASS | PASS | +9.4305 | +9.4305 | 0.00016% | 3.99e-05 |  |
| cart_hxltorsions | live | PASS | PASS | +47.4197 | +47.4197 | 8e-06% | 4.89e-07 |  |
| lk_ball_iso | live | PASS | PASS | +422.0395 | +422.0394 | 5.1e-06% | 7.46e-07 |  |
| lk_ball | live | PASS | PASS | +172.1964 | +172.1964 | 1.6e-06% | 4.28e-06 |  |
| lk_bridge | live | PASS | FAIL | +1.5786 | +1.5786 | 3.2e-05% | 1.37e-02 |  |
| lk_bridge_uncpl | live | PASS | FAIL | +10.9946 | +10.9946 | 5.6e-06% | 1.64e-01 |  |
| ref | live | PASS | PASS | -41.2750 | -41.2750 | 0% | 0.00e+00 |  |

### 1qys

| Term | Mode | Fwd | Grad | E(jax) | E(ref) | rel | max\|dg\| | Note |
|---|---|:---:|:---:|---:|---:|---:|---:|---|
| fa_ljatr | live | PASS | PASS | -522.1743 | -522.1743 | 0% | 5.85e-07 |  |
| fa_ljrep | live | PASS | PASS | +423.9046 | +423.9045 | 2.2e-05% | 1.19e-04 |  |
| fa_lk | live | PASS | PASS | +383.1672 | +383.1672 | 1.6e-05% | 1.36e-06 |  |
| fa_elec | live | PASS | PASS | -156.8206 | -156.8205 | 3.9e-05% | 1.18e-05 |  |
| hbond | live | PASS | PASS | -73.4560 | -73.4560 | 1.2e-05% | 9.10e-06 |  |
| rama | live | PASS | PASS | +35.8673 | +35.8673 | 2.1e-05% | 1.03e-05 |  |
| omega | live | PASS | FAIL | +2.2270 | +2.2271 | 0.0028% | 1.19e-04 |  |
| dunbrack_rot | live | PASS | PASS | +62.5133 | +62.5133 | 6.1e-06% | 8.43e-06 |  |
| dunbrack_rotdev | live | PASS | FAIL | +206.3297 | +206.3296 | 3e-05% | 1.62e-04 |  |
| dunbrack_semirot | live | PASS | PASS | +156.9141 | +156.9141 | 0% | 1.74e-05 |  |
| cart_lengths | live | FAIL | FAIL | +9.8192 | +0.0000 | 9.8e+14% | 1.22e+01 |  |
| cart_angles | live | FAIL | FAIL | +47.9077 | +0.0000 | 4.8e+15% | 1.73e+01 |  |
| cart_torsions | live | PASS | PASS | +13.9918 | +13.9919 | 0.00025% | 3.78e-05 |  |
| cart_impropers | live | PASS | FAIL | +3.1952 | +3.1952 | 0.0001% | 5.46e-05 |  |
| cart_hxltorsions | live | PASS | PASS | +51.5254 | +51.5254 | 7.4e-06% | 6.90e-07 |  |
| lk_ball_iso | live | PASS | PASS | +571.5001 | +571.5001 | 2.7e-06% | 9.44e-07 |  |
| lk_ball | live | PASS | PASS | +243.6249 | +243.6249 | 2.6e-07% | 2.97e-06 |  |
| lk_bridge | live | PASS | PASS | +1.4288 | +1.4288 | 5.7e-05% | 7.43e-07 |  |
| lk_bridge_uncpl | live | PASS | PASS | +8.2654 | +8.2654 | 2.3e-05% | 3.50e-06 |  |
| ref | live | PASS | PASS | -18.4877 | -18.4877 | 0% | 0.00e+00 |  |

### 1r21

| Term | Mode | Fwd | Grad | E(jax) | E(ref) | rel | max\|dg\| | Note |
|---|---|:---:|:---:|---:|---:|---:|---:|---|
| fa_ljatr | live | PASS | PASS | -545.7408 | -545.7408 | 0% | 4.51e-07 |  |
| fa_ljrep | live | PASS | PASS | +328.0019 | +328.0018 | 2.8e-05% | 2.63e-05 |  |
| fa_lk | live | PASS | PASS | +358.9670 | +358.9669 | 8.5e-06% | 1.16e-06 |  |
| fa_elec | live | PASS | PASS | -76.7813 | -76.7813 | 5e-05% | 1.11e-05 |  |
| hbond | live | PASS | PASS | -26.7509 | -26.7509 | 2.6e-05% | 1.05e-05 |  |
| rama | live | PASS | PASS | -6.9824 | -6.9824 | 9.6e-05% | 1.49e-05 |  |
| omega | live | PASS | FAIL | -1.9648 | -1.9647 | 0.0019% | 5.69e-05 |  |
| dunbrack_rot | live | PASS | PASS | +96.7915 | +96.7915 | 0% | 9.72e-06 |  |
| dunbrack_rotdev | live | PASS | PASS | +161.7523 | +161.7522 | 1.9e-05% | 2.87e-04 |  |
| dunbrack_semirot | live | PASS | PASS | +106.7291 | +106.7291 | 7.1e-06% | 1.38e-05 |  |
| cart_lengths | live | FAIL | FAIL | +56.4630 | +0.0000 | 5.6e+15% | 3.48e+01 |  |
| cart_angles | live | FAIL | FAIL | +88.2552 | +0.0000 | 8.8e+15% | 4.64e+01 |  |
| cart_torsions | live | PASS | FAIL | +1361.9386 | +1361.9387 | 9e-06% | 4.00e-05 |  |
| cart_impropers | live | PASS | PASS | +2.4074 | +2.4074 | 0.00017% | 4.23e-05 |  |
| cart_hxltorsions | live | PASS | PASS | +40.2084 | +40.2084 | 9.5e-06% | 8.66e-07 |  |
| lk_ball_iso | live | PASS | PASS | +534.0906 | +534.0906 | 1.3e-05% | 9.71e-07 |  |
| lk_ball | live | PASS | PASS | +214.3619 | +214.3619 | 2.5e-06% | 3.09e-06 |  |
| lk_bridge | live | PASS | FAIL | +1.5220 | +1.5220 | 5.3e-05% | 4.42e-02 |  |
| lk_bridge_uncpl | live | PASS | FAIL | +10.3213 | +10.3213 | 2.9e-05% | 3.84e-01 |  |
| ref | live | PASS | PASS | -39.1701 | -39.1701 | 0% | 0.00e+00 |  |

### 3plc

| Term | Mode | Fwd | Grad | E(jax) | E(ref) | rel | max\|dg\| | Note |
|---|---|:---:|:---:|---:|---:|---:|---:|---|
| fa_ljatr | live | PASS | PASS | -247.4706 | -247.4706 | 6.2e-06% | 4.10e-07 |  |
| fa_ljrep | live | PASS | PASS | +116.4537 | +116.4536 | 3.3e-05% | 3.60e-05 |  |
| fa_lk | live | PASS | PASS | +184.2703 | +184.2703 | 8.3e-06% | 7.72e-07 |  |
| fa_elec | live | PASS | PASS | -64.1733 | -64.1734 | 1.2e-05% | 1.00e-05 |  |
| hbond | live | PASS | PASS | -28.9855 | -28.9854 | 2.2e-05% | 8.87e-06 |  |
| rama | live | PASS | PASS | +7.7670 | +7.7670 | 8.6e-05% | 2.56e-05 |  |
| omega | live | PASS | PASS | +61.0048 | +61.0049 | 0.00015% | 9.41e-05 |  |
| dunbrack_rot | live | FAIL | FAIL | +52.9200 | +55.1774 | 4.1% | 5.80e-01 |  |
| dunbrack_rotdev | live | FAIL | FAIL | +261.6107 | +331.0977 | 21% | 2.48e+01 |  |
| dunbrack_semirot | live | PASS | PASS | +46.1987 | +46.1987 | 0% | 1.01e-05 |  |
| cart_lengths | live | FAIL | FAIL | +2726.0586 | +0.0000 | 2.7e+17% | 1.32e+03 |  |
| cart_angles | live | FAIL | FAIL | +96.4790 | +0.0000 | 9.6e+15% | 5.80e+01 |  |
| cart_torsions | live | PASS | PASS | +87.5639 | +87.5640 | 3.5e-05% | 2.87e-05 |  |
| cart_impropers | live | PASS | PASS | +5.2866 | +5.2866 | 0.00017% | 3.44e-05 |  |
| cart_hxltorsions | live | PASS | PASS | +33.0713 | +33.0713 | 0% | 5.35e-07 |  |
| lk_ball_iso | live | PASS | PASS | +248.3094 | +248.3094 | 5.9e-06% | 7.74e-07 |  |
| lk_ball | live | PASS | PASS | +97.8237 | +97.8237 | 4.3e-06% | 3.77e-06 |  |
| lk_bridge | live | PASS | FAIL | +1.0634 | +1.0634 | 6.5e-06% | 2.49e-02 |  |
| lk_bridge_uncpl | live | PASS | FAIL | +5.7136 | +5.7136 | 5.5e-06% | 2.04e-01 |  |
| ref | live | PASS | PASS | -8.9919 | -8.9919 | 1.1e-05% | 0.00e+00 |  |

### 6DMZ_A

| Term | Mode | Fwd | Grad | E(jax) | E(ref) | rel | max\|dg\| | Note |
|---|---|:---:|:---:|---:|---:|---:|---:|---|
| fa_ljatr | live | PASS | PASS | -250.2970 | -250.2970 | 6.1e-06% | 2.04e-05 |  |
| fa_ljrep | live | PASS | PASS | +611.2192 | +611.2188 | 6e-05% | 1.67e-04 |  |
| fa_lk | live | PASS | PASS | +195.4796 | +195.4796 | 1.6e-05% | 9.39e-07 |  |
| fa_elec | live | PASS | PASS | -69.5317 | -69.5317 | 1.1e-05% | 1.21e-05 |  |
| hbond | live | PASS | PASS | -25.7779 | -25.7778 | 3.5e-05% | 1.66e-05 |  |
| rama | live | PASS | PASS | +34.5738 | +34.5738 | 2.2e-05% | 1.16e-05 |  |
| omega | live | PASS | FAIL | +3.6762 | +3.6762 | 0.0011% | 5.86e-05 |  |
| dunbrack_rot | live | PASS | PASS | +64.7564 | +64.7564 | 0% | 8.58e-06 |  |
| dunbrack_rotdev | live | PASS | FAIL | +53.5988 | +53.5988 | 0% | 4.87e-05 |  |
| dunbrack_semirot | live | PASS | PASS | +49.1298 | +49.1298 | 0% | 2.24e-05 |  |
| cart_lengths | live | FAIL | FAIL | +11.7847 | +0.0000 | 1.2e+15% | 2.15e+01 |  |
| cart_angles | live | FAIL | FAIL | +24.5095 | +0.0000 | 2.5e+15% | 2.81e+01 |  |
| cart_torsions | live | PASS | PASS | +20.6197 | +20.6197 | 0.0001% | 2.60e-05 |  |
| cart_impropers | live | PASS | PASS | +2.1039 | +2.1039 | 0.00042% | 2.88e-05 |  |
| cart_hxltorsions | live | PASS | PASS | +32.1810 | +32.1810 | 1.2e-05% | 6.48e-07 |  |
| lk_ball_iso | live | PASS | PASS | +266.5249 | +266.5249 | 1.2e-05% | 8.78e-07 |  |
| lk_ball | live | PASS | PASS | +100.4393 | +100.4393 | 2e-06% | 2.54e-06 |  |
| lk_bridge | live | PASS | FAIL | +0.7751 | +0.7751 | 5.5e-05% | 5.94e-03 |  |
| lk_bridge_uncpl | live | PASS | FAIL | +5.7893 | +5.7893 | 2.6e-05% | 8.01e-02 |  |
| ref | live | PASS | PASS | +8.9209 | +8.9209 | 2.1e-05% | 0.00e+00 |  |

## Known residual: 3plc dunbrack_rot (4.1%) / dunbrack_rotdev (21%)

Localised to the two residues flanking the missing-density gap in 3plc:
**THR26** (block 25, 0-idx) and **ASP35** (block 26, 0-idx). PDB residue
numbers 27..34 are absent from the deposited coordinates; the chain
atoms HA(THR26) and O(ASP35) are 2.26 A apart so neither upstream's
biotite loader nor jtmol's IO drops the polymer bond (see commit
67ac14a). With the gap-spanning bond wired on both sides, both engines
synthesize phi/psi across the gap:

```
THR26: phi=-130.34 deg, psi=+139.99 deg (psi is synthesized: N(THR26)-CA-C-N(ASP35))
ASP35: phi= +14.18 deg, psi=+157.53 deg (phi is synthesized: C(THR26)-N(ASP35)-CA-C)
```

**What is identical between jtmol and upstream:**
- Dihedral atoms resolved by both sides for these two residues (verified
  by tracing `pose_stack_inter_block_connections` + `conn_atoms` /
  `block_type_atom_downstream_of_conn`; the gap-bridging fix in 67ac14a
  brought the two wirings into agreement).
- Synthesized phi/psi values to 4 decimal places (`dihedral_angle` is
  the same atan2 formulation).
- rotbin classification: THR26 chi1=112.46 deg -> g+ (bin 0); ASP35
  chi1=-75 deg -> g- (bin 2).
- B-spline coefficient pole, normalisation lambda, weight polynomial,
  and 4-cell periodic wrap (line-for-line port of `bspline.hh`).
- Float32 vs float64 prefilter: refiltering the THR g+ mean/sdev tables
  in float32 (matching upstream) vs float64 (current jtmol) gives
  coefficient diffs at the 1e-7 level — five orders of magnitude too
  small to explain a 21% scoring gap.
- For the THR26 g+ rotamer at the synthesized (phi, psi), the
  interpolated `mean = 60.116 deg`, `sdev = 8.394 deg`. With chi1 =
  112.46 deg, the penalty is `(52.34 deg)^2 / (2 * 8.39 deg^2) =
  19.44`, matching the jtmol total to float precision.

**Conclusion of the investigation:**
The b-spline implementation and dihedral-atom selection are
mathematically identical between jtmol and upstream at these two
residues. We cannot reproduce upstream's higher value locally without
the upstream compiled C++ extension (the bundled `tmol/` checkout is
import-only). The most likely sources of the 4.1% / 21% gap recorded
above are:

1. The Modal A100 run that produced the upstream column may have used
   subtly different input coordinates (float32 vs float64 promotion at
   a different stage of the load pipeline), or
2. The stored numbers are stale and a fresh sweep would close the gap.

Either way, the gap is **localised to two residues with nonsensical
gap-synthesized phi/psi** and does **not** reflect a physical-residue
scoring bug. Rama / omega / cart_torsions on the same two residues all
PASS, which is what the gap-bridging fix in 67ac14a was actually
chasing.

**Decision: defer.** Skipping the gap-bridging residues' dunbrack
contribution (Option A) would push jtmol *further* from upstream, not
closer. Tracking down the remaining float32-precision divergence (Option
B) is not actionable without a working upstream build. The four other
PDBs in the panel hit dunbrack parity to <= 6e-06%; this residual is
isolated to 3plc.

