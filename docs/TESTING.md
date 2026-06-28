# Testing jtmol

Every demo and test below should run end-to-end on a local CPU after
`uv pip install -e .[dev]` plus `uv pip install equinox jaxtyping`
(the latter are only needed by the mosaic integration demos). No Modal,
no GPU, no torch.

Run order during a normal post-change verification:

1. `pytest tmol_jax/tests` — fast, deterministic, gates correctness.
2. `examples/integration_mosaic_hallucinator.py` — confirms the mosaic
   `LossTerm` wrappers still compose with synthetic Boltz-2 outputs.
3. `examples/demo_kinetics_tuning.py` and
   `examples/demo_alanine_scan_kinetics.py` — sentinel for the
   `gen_torsions` port and per-residue gradient breakdown.
4. Optional: the remaining `examples/demo_*.py` if the change touched
   any extra-scores / steering code paths.

## Unit & gradcheck suite

`pytest tmol_jax/tests` — 47 tests covering scoring (`test_potentials.py`),
analytic-vs-FD gradients (`test_gradcheck.py`), kinematics
(`test_kinematics.py`), pose-stack invariants (`test_pose_stack.py`),
and the load-1ubq-and-score smoke test (`test_smoke.py`). One
`disulfide` gradcheck is skipped (energy ≡ 0 on stock 1ubq, no FD
signal). Success = "46 passed, 1 skipped" in roughly 5–7 minutes.

## examples/integration_mosaic_hallucinator.py

End-to-end smoke for the mosaic binder-hallucinator integration.
Builds a pose-stack template from a toy 7+7-residue
binder+target, runs `TmolInterfaceLoss`, `TmolBackboneLoss`,
`TmolClashFilter`, takes `jax.grad` w.r.t. atom37 coords, composes
them via `1.0 * iface + 0.01 * bb + 0.01 * clash`, then runs the
realistic `_sanity_test_with_real_1ubq()` (1ubq split into a
synthetic chain A 1–40 / chain B 41–76). Success: the toy block
returns a finite dG and a finite-and-non-zero gradient (`grad max
≈ 0.012`); the 1ubq block prints `dG_interface ≈ +157 kcal/mol`
with `grad max ≈ 195` and the per-term weight sweep returns five
visibly different dG values (default ≈ +157, boost electrostatics
≈ -39, boost H-bonds ≈ +91, hard-wall ≈ +749, polars-only ≈
-60). The toy block's per-weight sweep is intentionally flat
because the toy chains are placed 50 Å apart and have zero
interface contribution; the realistic 1ubq sweep is the
load-bearing check. Invoke: `.venv/bin/python
examples/integration_mosaic_hallucinator.py`.

## examples/demo_kinetics_tuning.py

Three-axis kinetic-property smoke test on the 1ubq split-binder. Computes
`U_SS99` (k_on proxy), `BUH` (k_off proxy), `dG_Kd` (Rosetta interface
energy), takes the gradient w.r.t. coords on each axis, takes one step,
and ranks per-residue hot residues. Success: each `useful_gradient` flag
is `True`, the three axes have small overlap in their top-8 hot
residues (proves the channels are not degenerate), and the per-axis
loss strictly decreases after the perturbation. Invoke:
`.venv/bin/python examples/demo_kinetics_tuning.py`.

## examples/demo_alanine_scan_kinetics.py

Per-residue alanine scan in kinetic space. For each top hot residue on
the k_on and k_off axes it mutates to ALA, recomputes U_SS99 and BUH,
and tags better/worse/≈. Success: at least one residue moves
`U_SS99` by more than 0.1 kcal/mol (e.g. residue 39 ASP→ALA
≈ +0.97) and the joint table prints without NaNs. Sensitive to
`gen_torsions` because alanine has a different torsion topology than
ARG/LYS/GLU; if the port broke, this demo fails first. Invoke:
`.venv/bin/python examples/demo_alanine_scan_kinetics.py`.

## examples/demo_affinity_tuning.py

Per-term, per-residue interface-energy breakdown on 1ubq split-binder.
Imports `examples.integration_mosaic_hallucinator`, so run with
`PYTHONPATH=.`. Success: the printed table has finite numbers, the top
10 hot residues include known clash hot-spots (large `fa_ljrep`), and
the totals match the dG reported by the integration demo. Invoke:
`PYTHONPATH=. .venv/bin/python examples/demo_affinity_tuning.py`.

## examples/demo_extra_scores.py

Five drop-in non-Rosetta scores (SS99, BUH, PRODIGY, TransComp, DFIRE
placeholder). Each prints a value, an analytic gradient norm, and the
FD-perturbed loss after a one-step gradient move. Success: every
"useful gradient" flag is `True` and every loss strictly decreases
after the step. DFIRE uses placeholder coefficients so its magnitude
is huge by design; only the sign/finite checks matter for that line.
Invoke: `.venv/bin/python examples/demo_extra_scores.py`.

## examples/demo_steering.py

Six interface-loss schemes (baseline, electrostatics+Hbond, shape, polar
only, water-mediated, clash-only), with their gradient cosine-similarity
to the baseline. Success: schemes labelled "rescaling" land at cos
> 0.95, the "polar interface" scheme lands materially below
(cos ≈ 0.3). Invoke: `PYTHONPATH=. .venv/bin/python
examples/demo_steering.py`.

## Other examples

`minimize_1ubq.py`, `differentiable_folding.py`, `cheap_trial_sweep.py`,
`trial_run.py`, `vmap_batch_scoring.py`, `vmap_over_poses.py`,
`neighbor_list_benchmark.py`, `cos_sim_extended.py` are local-CPU
runnable but slower and not part of the standard verification path.
`modal_parity_run.py` and `modal_trial_run.py` require Modal and are
explicitly out of scope for the local checklist.
