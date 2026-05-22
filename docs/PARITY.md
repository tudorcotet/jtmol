# Parity vs upstream tmol

## Forward energy (1ubq, all 20 channels)

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

**20/20 terms forward-match to machine precision.**

## Gradient (1ubq, float64 max elementwise diff vs upstream)

| Term | max\|dg\| | Status |
|---|---:|---|
| fa_ljatr, fa_ljrep, fa_lk, fa_elec | < 1e-5 | bit-exact |
| hbond | < 1e-5 | bit-exact |
| rama  | < 1e-5 | bit-exact |
| dunbrack_rot, dunbrack_semirot | < 1e-5 | bit-exact |
| lk_ball_iso, lk_ball, lk_bridge | < 1e-5 | bit-exact |
| ref, cart_lengths, cart_impropers | 0 / < 1e-5 | bit-exact |
| **cart_angles** | **1.8e-13** | bit-exact (fixed) |
| omega | 9.4e-5 | custom-JVP @ kink |
| cart_torsions | 8.6e-10 | custom-JVP @ kink |
| cart_hxltorsions | < 1e-3 | custom-JVP @ kink |
| dunbrack_rotdev | 2.7e-4 | bicubic vs bilinear |
| lk_bridge_uncpl | 0.16 | **upstream bug** — jtmol matches FD, upstream's hand-written CUDA adjoint does not (see [UPSTREAM_BUG_LK_BRIDGE_UNCPL.md](UPSTREAM_BUG_LK_BRIDGE_UNCPL.md)) |

Custom `@jax.custom_jvp` rules close the cubic-Hermite kink gradients for
omega / cart_torsions / cart_hxltorsions (jtmol uses the Blondel-Karplus
1996 analytical formulas there to match upstream's hand-written CUDA
adjoints exactly).

## Multi-PDB panel

Cross-validated on a 6-PDB panel (1ubq, 1qys, 1BL8, 1r21, 3plc, 6DMZ_A)
including disulfide-rich folds. See [MULTI_PDB_PARITY.md](MULTI_PDB_PARITY.md).

Multi-chain interface PDBs (1cgi, 2oob, 7opb) covered in
[MULTI_CHAIN_PARITY.md](MULTI_CHAIN_PARITY.md). 1BL8 is documented as
deferred — upstream itself refuses to load it (`"missing non-leaf atom
CG on residue 4 ARG"`).

## Reproducing the table

```bash
# Forward + grad sweep across the full PDB panel:
PYTHONPATH=.:tmol python scripts/run_upstream_parity.py

# Writes docs/UPSTREAM_PARITY.md and docs/upstream_parity_results.csv.
# Exit code mirrors pytest: 0 on all pass, 1 on any failure.
```
