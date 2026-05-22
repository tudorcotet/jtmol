# Physics: Rosetta `beta_nov2016_cart`

`tmol_jax` evaluates the full Rosetta `beta_nov2016_cart` all-atom
energy function. Each term is a channel in `ScoreType`; weights in
`beta2016_score_function()` match upstream `_non_memoized_beta2016`.
Parameters are loaded verbatim from the upstream YAMLs in
`database/default/scoring/`.

References:
- Park, Bradley, Greisen, Liu, Mulligan, Kim, Baker, DiMaio. *JCTC*
  2016, 12(12), 6201-6212 -- `beta_nov2016` optimisation.
- Alford, Leaver-Fay, Jeliazkov *et al.* (Leman et al.). *JCTC* 2017,
  13(6), 3031-3048 -- "The Rosetta All-Atom Energy Function".

## Pair terms

- **`fa_ljatr` / `fa_ljrep`** -- 12-6 LJ split into attractive /
  repulsive channels, shifted to cap the repulsive wall, faded to
  zero with a cubic-Hermite polynomial between 4.5 and 6.0 A.
- **`fa_lk`** -- isotropic Lazaridis-Karplus desolvation: pairwise
  Gaussian-in-distance with per-atom-type volumes, lambdas, and
  `dG_free`; symmetrised.
- **`lk_ball` / `lk_ball_iso` / `lk_bridge` / `lk_bridge_uncpl`** --
  anisotropic LK with explicit water sites along donor/acceptor
  hydrogen-bonding axes; `lk_ball_iso` subtracts the isotropic
  reference so only the anisotropic correction is counted.
- **`fa_elec`** -- sigmoidal-dielectric Coulombic
  `eiej * (C1/(d*eps(d)) - C2)` with
  `eps(d) = D - 0.5*(D-D0)*(2 + 2*d*S + d^2*S^2)*exp(-d*S)`. Cubic-
  Hermite fades at `min_dis` and `max_dis`. `C1 = 322.0637`.

Count-pair: path_length > 4 -> 1.0; == 4 -> 0.2; < 4 -> 0.0.

## Hydrogen bonding (`hbond`)

Four measurements -- AH distance, AHD angle, BAH angle, and a
hybridisation-dependent chi (B0-B-A-H) -- each scored by an
11th-degree polynomial bounded in its own x-range with y clamps.
sp2/sp3/ring hybridisation selects the chi polynomial; donor /
acceptor types select tables from `hbond.yaml`.

## Backbone torsion

- **`rama`** -- bilinear interpolation of a per-amino-acid
  `[n_phi, n_psi]` Ramachandran grid (typically 36x36 at 10 degrees).
- **`omega`** -- backbone-dependent harmonic spring around
  `omega0(phi, psi)` with `k(phi, psi)`, from the same bilinear lookup.

## Rotamer library (Dunbrack)

- **`dunbrack_rot`** -- `-ln P(rotamer | phi, psi)`.
- **`dunbrack_rotdev`** -- per-chi Gaussian penalty with bbdep
  mean/sigma: `(chi - mean)^2 / (2 sigma^2)`.
- **`dunbrack_semirot`** -- for PHE, TYR, HIS, ASN, ASP, GLN, GLU,
  TRP a 3-D `(phi, psi, chi_last)` log-probability table.

## Cartesian bonded

- **`cart_lengths`**: `0.5 k (d - d0)^2`.
- **`cart_angles`**: `0.5 k (theta - theta0)^2`.
- **`cart_torsions`**: `0.5 k (1 + cos(n phi - phi0))`.
- **`cart_impropers`**: harmonic in improper angle.
- **`cart_hxltorsions`**: three-cosine sum for hydroxyl torsions.

## Other

- **`disulfide`** -- S-S distance, two CA-CB-S-S' angles, a CB-S-S-CB
  dihedral (two von Mises peaks), two CA-CB-S-S dihedrals (three
  peaks each).
- **`ref`** -- 1-body per-residue reference energy from `ref.yaml`.
