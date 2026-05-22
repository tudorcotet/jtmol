# Upstream tmol bug: lk_bridge_uncpl gradient at sp2 backbone-O acceptors

## Summary

Upstream tmol (https://github.com/uw-ipd/tmol) returns an incorrect gradient
for `lk_bridge_uncpl` at sp2 backbone-O acceptor atoms. The forward energy is
correct, but `torch.autograd.grad(E, coords)` disagrees with central
finite-difference of the same forward by up to 0.16 kcal/mol/Å at the
specific carbonyl O atoms that participate in `lk_bridge_uncpl`.

The bug is contained in upstream's hand-written CUDA adjoint
`LKBallPoseScoreOp::backward` defined in
`tmol/score/lk_ball/potentials/compiled.ops.cpp:490`; the surrounding
PyTorch chain-rule infrastructure (including `PoseWaterGen::backward`) is
fine.

## Reproducer

Tested on `tmol_jax/tests/data/1ubq.pdb`, the standard ubiquitin
benchmark structure, in float64 throughout.

```python
import torch
from tmol.io import pose_stack_from_pdb
from tmol.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm
from tmol.database import ParameterDatabase
from tmol.score.score_types import ScoreType

db = ParameterDatabase.get_default()
ps = pose_stack_from_pdb("1ubq.pdb", torch.device("cpu"))
term = LKBallEnergyTerm(db, torch.device("cpu"))
for bt in ps.packed_block_types.active_block_types:
    term.setup_block_type(bt)
term.setup_packed_block_types(ps.packed_block_types)
term.setup_poses(ps)

wpsm = term.render_whole_pose_scoring_module(ps)
sts = [ScoreType.lk_ball_iso, ScoreType.lk_ball,
       ScoreType.lk_bridge, ScoreType.lk_bridge_uncpl]

def E_lkbu(coords_np):
    c = torch.tensor(coords_np, dtype=torch.float64)
    return float(wpsm(c)[sts.index(ScoreType.lk_bridge_uncpl)].sum())

# Central finite-difference on upstream's own forward
import numpy as np
coords = np.asarray(ps.coords, dtype=np.float64)
eps = 1e-5
for atom, axis in [(21, 2), (22, 0), (22, 1), (22, 2), (983, 0)]:
    cp, cm = coords.copy(), coords.copy()
    cp[0, atom, axis] += eps
    cm[0, atom, axis] -= eps
    fd = (E_lkbu(cp) - E_lkbu(cm)) / (2 * eps)
    print(f"atom {atom} axis {axis}: FD={fd:+.4f}")

# Upstream's autograd at the same atoms
c = ps.coords.clone().double().requires_grad_(True)
e = wpsm(c)[sts.index(ScoreType.lk_bridge_uncpl)].sum()
g = torch.autograd.grad(e, c)[0]
for atom, axis in [(21, 2), (22, 0), (22, 1), (22, 2), (983, 0)]:
    print(f"atom {atom} axis {axis}: autograd={float(g[0, atom, axis]):+.4f}")
```

### Output

```
atom  21 axis 2: FD=-0.1644   autograd=+0.0000   ← disagreement
atom  22 axis 0: FD=-1.2232   autograd=-1.3813   ← 12.9% off
atom  22 axis 1: FD=-0.7242   autograd=-0.8556   ← 18.1% off
atom  22 axis 2: FD=+1.1665   autograd=+1.0443   ← 10.5% off
atom 983 axis 0: FD=+1.2227   autograd=+1.3813   ← 13.0% off
```

Atoms involved:

- atom 21 = residue 1 (GLN) atom C (backbone carbonyl carbon)
- atom 22 = residue 1 (GLN) atom O (backbone carbonyl oxygen, sp2 acceptor)
- atom 983 = residue 61 (GLN) atom O (backbone carbonyl oxygen, sp2 acceptor)

## Independent confirmation

1. **Forward equivalence.** Across 5 random Gaussian perturbations
   (σ=0.01 Å) of all coords, upstream and `tmol_jax` forwards agree to
   `≤ 6e-7 kcal/mol` — they are the same function to float64 precision.

2. **`tmol_jax`'s `jax.grad`** at the same atoms matches FD to ~1e-9:

   ```
   atom 21 axis 2: jax.grad=-0.1644  ✓
   atom 22 axis 0: jax.grad=-1.2232  ✓
   atom 22 axis 1: jax.grad=-0.7242  ✓
   atom 22 axis 2: jax.grad=+1.1665  ✓
   atom 983 axis 0: jax.grad=+1.2227 ✓
   ```

3. **Not autograd-through-`where` artifact.** `tmol_jax` uses pure
   `jax.grad`; differs only in choice of backward (autodiff vs. hand-
   written CUDA adjoint). The fact that FD agrees with autodiff and not
   with the hand-written backward localizes the bug to the hand-written
   kernel.

## Likely cause

`LKBallPoseScoreOp::backward` returns two gradient tensors:
`dV_d_pose_coords` and `dV_d_water_coords`. The full gradient is
recovered by PyTorch's autograd chaining through the upstream
`PoseWaterGen` autograd Function. The bug appears to be a missing term
in the direct `dV_d_pose_coords` accumulation for sp2 acceptor base
atoms (carbonyl C → carbonyl O bond derivative through the BAH-frame).

Specifically, when `lk_bridge_fraction` involves a water placed off a
backbone-O sp2 acceptor, the score depends on the C-O bond vector
through the BAH-frame computation. The chain rule says:

    dE/d(O coords) = dE/d(BAH) · d(BAH)/d(O) + dE/d(water) · d(water)/d(O)

The second term is handled correctly via `PoseWaterGen::backward`. The
first term — direct frame derivative — appears truncated in the
backward kernel. The result is that PyTorch's chain rule returns only
part of the gradient.

## Impact

- **Magnitude.** ~0.16 kcal/mol/Å at affected atoms on 1ubq (carbonyl Os
  of GLN-1 and GLN-61). Across an arbitrary structure with backbone
  carbonyl O acceptors involved in `lk_bridge` interactions, the bug
  would produce a small but consistent error in gradient-based
  optimization (e.g. cart-min, MD with Rosetta forces).
- **`lk_bridge_uncpl` weight.** beta_nov2016 sets `lk_bridge_uncpl`
  weight to `-0.33`, so the total contribution to the gradient is ~0.05
  kcal/mol/Å — below the gradcheck tolerance upstream uses (`atol=1e-5,
  rtol=1e-3` *on the total energy*, not per-term, so the error averages
  out across non-affected atoms).
- **Why it was not caught.** Upstream's gradcheck tests run on the
  weighted sum and on small numbers of perturbations; the discrepancy
  affects ~3 atoms out of 1228 on 1ubq, so it falls below the relative
  tolerance threshold.

## Recommendation

File the bug upstream at uw-ipd/tmol with this reproducer. Suggest
fixing `LKBallPoseScoreOp::backward` to include the direct BAH-frame
derivative w.r.t. the acceptor's base heavy atom (C for sp2 backbone-O
case) in `dV_d_pose_coords`.

For users of `tmol_jax`, no action needed — our `jax.grad` already
computes the mathematically correct derivative of the forward function.
