# Upstream parity (tmol_jax vs tmol)

- Upstream live: **yes**
- Tolerances: atol=1e-05, rtol=0.001
- Comparison rule: `|E_jax - E_ref| <= atol + rtol * |E_ref|` (matches torch.autograd.gradcheck default)

## Summary

- Total rows: **80**
- Forward: **53 PASS**, 7 FAIL, 20 SKIP
- Gradient: **38 PASS**, 22 FAIL, 20 SKIP (skip means upstream-unavailable or no-coord-dep)

## 1ubq

| Term | Mode | Fwd | Grad | E(jax) | E(ref) | max\|dg\| | Note |
|---|---|:---:|:---:|---:|---:|---:|---|
| fa_ljatr | live | PASS | PASS | -417.9583 | -417.9583 | 2.98e-07 |  |
| fa_ljrep | live | PASS | PASS | +240.7147 | +240.7147 | 7.63e-06 |  |
| fa_lk | live | PASS | PASS | +298.2765 | +298.2765 | 1.13e-06 |  |
| fa_elec | live | PASS | PASS | -136.2925 | -136.2925 | 1.29e-05 |  |
| hbond | live | PASS | PASS | -55.6756 | -55.6756 | 6.14e-06 |  |
| rama | live | PASS | PASS | -12.7434 | -12.7434 | 1.60e-05 |  |
| omega | live | PASS | FAIL | +4.1002 | +4.1003 | 5.63e-03 |  |
| dunbrack_rot | live | PASS | PASS | +70.6497 | +70.6497 | 9.89e-06 |  |
| dunbrack_rotdev | live | PASS | FAIL | +240.3102 | +240.3101 | 2.71e-04 |  |
| dunbrack_semirot | live | PASS | PASS | +99.6609 | +99.6609 | 1.48e-05 |  |
| cart_lengths | live | PASS | PASS | +37.7637 | +37.7637 | 3.81e-06 |  |
| cart_angles | live | PASS | FAIL | +183.5697 | +183.5697 | 1.02e-03 |  |
| cart_torsions | live | PASS | FAIL | +46.0236 | +46.0236 | 3.06e-02 |  |
| cart_impropers | live | PASS | PASS | +9.4305 | +9.4305 | 3.71e-05 |  |
| cart_hxltorsions | live | PASS | FAIL | +47.4197 | +47.4197 | 3.46e-03 |  |
| lk_ball_iso | live | PASS | PASS | +422.0394 | +422.0396 | 1.19e-06 |  |
| lk_ball | live | PASS | PASS | +172.1964 | +172.1964 | 2.79e-05 |  |
| lk_bridge | live | PASS | FAIL | +1.5786 | +1.5786 | 1.37e-02 |  |
| lk_bridge_uncpl | live | PASS | FAIL | +10.9946 | +10.9946 | 1.64e-01 |  |
| ref | live | PASS | PASS | -41.2750 | -41.2750 | 0.00e+00 |  |

## 1qys

| Term | Mode | Fwd | Grad | E(jax) | E(ref) | max\|dg\| | Note |
|---|---|:---:|:---:|---:|---:|---:|---|
| fa_ljatr | live | PASS | PASS | -522.1743 | -522.1744 | 3.87e-07 |  |
| fa_ljrep | live | PASS | PASS | +423.9045 | +423.9046 | 6.87e-05 |  |
| fa_lk | live | PASS | PASS | +383.1672 | +383.1670 | 1.19e-06 |  |
| fa_elec | live | PASS | PASS | -156.8206 | -156.8204 | 1.45e-05 |  |
| hbond | live | FAIL | FAIL | -73.8124 | -73.4560 | 5.01e+00 |  |
| rama | live | PASS | PASS | +35.8673 | +35.8673 | 1.67e-05 |  |
| omega | live | PASS | FAIL | +2.2271 | +2.2271 | 5.11e-02 |  |
| dunbrack_rot | live | PASS | PASS | +62.5133 | +62.5133 | 1.62e-05 |  |
| dunbrack_rotdev | live | PASS | FAIL | +206.3297 | +206.3300 | 6.19e-03 |  |
| dunbrack_semirot | live | PASS | PASS | +156.9141 | +156.9141 | 4.30e-05 |  |
| cart_lengths | live | PASS | PASS | +9.8192 | +9.8192 | 9.54e-07 |  |
| cart_angles | live | PASS | FAIL | +47.9077 | +47.9078 | 1.22e-03 |  |
| cart_torsions | live | PASS | FAIL | +13.9918 | +13.9919 | 4.88e-02 |  |
| cart_impropers | live | PASS | PASS | +3.1952 | +3.1952 | 5.32e-05 |  |
| cart_hxltorsions | live | PASS | FAIL | +51.5254 | +51.5254 | 3.00e-03 |  |
| lk_ball_iso | live | PASS | PASS | +571.5001 | +571.5002 | 9.54e-07 |  |
| lk_ball | live | PASS | PASS | +243.6249 | +243.6249 | 1.63e-05 |  |
| lk_bridge | live | PASS | PASS | +1.4288 | +1.4288 | 8.18e-06 |  |
| lk_bridge_uncpl | live | PASS | PASS | +8.2654 | +8.2654 | 3.39e-05 |  |
| ref | live | PASS | PASS | -18.4877 | -18.4877 | 0.00e+00 |  |

## 1BL8

| Term | Mode | Fwd | Grad | E(jax) | E(ref) | max\|dg\| | Note |
|---|---|:---:|:---:|---:|---:|---:|---|
| fa_ljatr | skip | — | — |  |  |  | tmol_jax pose load failed |
| fa_ljrep | skip | — | — |  |  |  | tmol_jax pose load failed |
| fa_lk | skip | — | — |  |  |  | tmol_jax pose load failed |
| fa_elec | skip | — | — |  |  |  | tmol_jax pose load failed |
| hbond | skip | — | — |  |  |  | tmol_jax pose load failed |
| rama | skip | — | — |  |  |  | tmol_jax pose load failed |
| omega | skip | — | — |  |  |  | tmol_jax pose load failed |
| dunbrack_rot | skip | — | — |  |  |  | tmol_jax pose load failed |
| dunbrack_rotdev | skip | — | — |  |  |  | tmol_jax pose load failed |
| dunbrack_semirot | skip | — | — |  |  |  | tmol_jax pose load failed |
| cart_lengths | skip | — | — |  |  |  | tmol_jax pose load failed |
| cart_angles | skip | — | — |  |  |  | tmol_jax pose load failed |
| cart_torsions | skip | — | — |  |  |  | tmol_jax pose load failed |
| cart_impropers | skip | — | — |  |  |  | tmol_jax pose load failed |
| cart_hxltorsions | skip | — | — |  |  |  | tmol_jax pose load failed |
| lk_ball_iso | skip | — | — |  |  |  | tmol_jax pose load failed |
| lk_ball | skip | — | — |  |  |  | tmol_jax pose load failed |
| lk_bridge | skip | — | — |  |  |  | tmol_jax pose load failed |
| lk_bridge_uncpl | skip | — | — |  |  |  | tmol_jax pose load failed |
| ref | skip | — | — |  |  |  | tmol_jax pose load failed |

## 1r21

| Term | Mode | Fwd | Grad | E(jax) | E(ref) | max\|dg\| | Note |
|---|---|:---:|:---:|---:|---:|---:|---|
| fa_ljatr | live | PASS | PASS | -545.7408 | -545.7408 | 4.17e-07 |  |
| fa_ljrep | live | PASS | PASS | +328.0018 | +328.0017 | 7.63e-06 |  |
| fa_lk | live | PASS | PASS | +358.9670 | +358.9668 | 1.59e-06 |  |
| fa_elec | live | FAIL | FAIL | -77.0381 | -76.7813 | 2.31e-01 |  |
| hbond | live | PASS | PASS | -26.7509 | -26.7509 | 9.16e-06 |  |
| rama | live | FAIL | FAIL | -9.4837 | -6.9824 | 5.17e+00 |  |
| omega | live | FAIL | FAIL | -1.7983 | -1.9659 | 8.18e+00 |  |
| dunbrack_rot | live | PASS | PASS | +96.7915 | +96.7915 | 9.54e-06 |  |
| dunbrack_rotdev | live | FAIL | FAIL | +175.6039 | +161.7523 | 1.65e+01 |  |
| dunbrack_semirot | live | FAIL | FAIL | +124.4064 | +106.7291 | 3.92e+00 |  |
| cart_lengths | live | PASS | PASS | +56.4630 | +56.4630 | 1.91e-06 |  |
| cart_angles | live | FAIL | FAIL | +88.2552 | +88.1178 | 4.40e+00 |  |
| cart_torsions | live | PASS | FAIL | +1361.9384 | +1361.9390 | 3.90e-02 |  |
| cart_impropers | live | PASS | PASS | +2.4074 | +2.4074 | 2.98e-05 |  |
| cart_hxltorsions | live | PASS | PASS | +40.2084 | +40.2084 | 2.38e-06 |  |
| lk_ball_iso | live | PASS | PASS | +534.0906 | +534.0906 | 7.15e-07 |  |
| lk_ball | live | PASS | PASS | +214.3619 | +214.3619 | 1.12e-05 |  |
| lk_bridge | live | PASS | FAIL | +1.5220 | +1.5220 | 4.42e-02 |  |
| lk_bridge_uncpl | live | PASS | FAIL | +10.3213 | +10.3214 | 3.84e-01 |  |
| ref | live | PASS | PASS | -39.1701 | -39.1701 | 0.00e+00 |  |

