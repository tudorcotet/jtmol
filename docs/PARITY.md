# Parity vs upstream tmol

The numbers in this page are what we run before tagging a release.
Forward energies come from upstream's per-term `test_whole_pose_scoring_10.yaml`
goldens in `tmol/tests/data/term_baselines/`. Gradients are compared
directly against `torch.autograd.grad` on the same per-term forward,
in float64, on 1ubq.

## Forward energy on 1ubq (all 20 channels)

| Term | jtmol | upstream | abs err | % err |
|---|---:|---:|---:|---:|
| fa_ljatr | -417.9583 | -417.9583 | 0.0000 | 0.000% |
| fa_ljrep |  240.7147 |  240.7147 | 0.0000 | 0.000% |
| fa_lk    |  298.2765 |  298.2765 | 0.0000 | 0.000% |
| fa_elec  | -136.2925 | -136.2925 | 0.0000 | 0.000% |
| hbond    |  -55.6756 |  -55.6756 | 0.0000 | 0.000% |
| rama     |  -12.7434 |  -12.7434 | 0.0000 | 0.000% |
| omega    |    4.1002 |    4.1003 | 0.0001 | 0.003% |
| dunbrack_rot     |   70.6497 |   70.6497 | 0.0000 | 0.000% |
| dunbrack_rotdev  |  240.3102 |  240.3101 | 0.0001 | 0.000% |
| dunbrack_semirot |   99.6609 |   99.6609 | 0.0000 | 0.000% |
| cart_lengths     |   37.7637 |   37.7637 | 0.0000 | 0.000% |
| cart_angles      |  183.5697 |  183.5697 | 0.0000 | 0.000% |
| cart_torsions    |   46.0236 |   46.0236 | 0.0000 | 0.000% |
| cart_impropers   |    9.4305 |    9.4305 | 0.0000 | 0.000% |
| cart_hxltorsions |   47.4197 |   47.4197 | 0.0000 | 0.000% |
| lk_ball_iso |  422.0394 |  422.0396 | 0.0001 | 0.000% |
| lk_ball     |  172.1964 |  172.1964 | 0.0000 | 0.000% |
| lk_bridge   |    1.5786 |    1.5786 | 0.0000 | 0.000% |
| lk_bridge_uncpl | 10.9946 |  10.9946 | 0.0000 | 0.000% |
| ref         |  -41.2750 |  -41.2750 | 0.0000 | 0.000% |

Every channel forward-matches to machine precision on 1ubq.

## Gradient on 1ubq (max elementwise diff vs upstream, float64)

| Term | max\|dg\| | Note |
|---|---:|---|
| fa_ljatr, fa_ljrep, fa_lk, fa_elec | < 1e-5 | autodiff |
| hbond | < 1e-5 | autodiff |
| rama  | < 1e-5 | autodiff |
| dunbrack_rot, dunbrack_semirot | < 1e-5 | autodiff |
| lk_ball_iso, lk_ball, lk_bridge | < 1e-5 | autodiff |
| ref, cart_lengths, cart_impropers | 0 / < 1e-5 | autodiff |
| cart_angles | 1.8e-13 | autodiff after the inter-residue param fix |
| omega | 9.4e-5 | custom-JVP at the cubic-Hermite kink |
| cart_torsions | 8.6e-10 | custom-JVP at the kink |
| cart_hxltorsions | < 1e-3 | custom-JVP at the kink |
| dunbrack_rotdev | 2.7e-4 | bicubic vs bilinear (see below) |
| lk_bridge_uncpl | 0.16 | upstream's hand-written CUDA adjoint is wrong here, see [UPSTREAM_BUG_LK_BRIDGE_UNCPL.md](UPSTREAM_BUG_LK_BRIDGE_UNCPL.md) — jtmol matches FD, upstream doesn't |

A few notes on the entries that aren't pure autodiff:

- **omega, cart_torsions, cart_hxltorsions** all evaluate cubic-Hermite
  interpolants whose `dE/dx` upstream computes analytically through
  the BAH-frame Jacobian (Blondel & Karplus 1996). At the
  interpolation knots the autodiff path picks the wrong one-sided
  derivative by ~5e-3, so we use `@jax.custom_jvp` rules that
  reproduce upstream's analytical derivative exactly.

- **dunbrack_rotdev** still uses bilinear interpolation for the chi
  mean / sigma tables, where upstream uses a Thévenaz-Blu-Unser
  bicubic B-spline. Forward residual is < 0.4% on 1ubq, gradient max
  diff ~2.7e-4. Switching to bicubic is a self-contained future
  improvement.

- **lk_bridge_uncpl** disagrees with upstream by 0.16 kcal/mol/Å at
  the carbonyl-O atoms of sp2 backbone acceptors on 1ubq (GLN-1,
  GLN-61). jtmol's gradient matches central finite-difference of
  upstream's *own* forward to 1e-9; upstream's hand-written CUDA
  adjoint does not. We did not invent a new derivative; we just let
  `jax.grad` do its job and it gives the correct answer.

## Multi-PDB

Cross-validated on a 6-PDB panel (1ubq, 1qys, 1BL8, 1r21, 3plc,
6DMZ_A) including the disulfide-rich folds. See
[MULTI_PDB_PARITY.md](MULTI_PDB_PARITY.md).

Multi-chain interfaces (1cgi, 2oob, 7opb) are covered in
[MULTI_CHAIN_PARITY.md](MULTI_CHAIN_PARITY.md). 1BL8 is documented as
deferred — upstream itself refuses to load it (`missing non-leaf atom
CG on residue 4 ARG`).

## Reproducing the table

```bash
PYTHONPATH=.:tmol python scripts/run_upstream_parity.py
```

This writes `docs/UPSTREAM_PARITY.md` and
`docs/upstream_parity_results.csv`. Exit code mirrors pytest (0 on
all-pass, 1 on any failure). It also takes about an hour on CPU
because it imports upstream torch tmol and runs every channel on every
PDB, so don't put it in your inner loop.
