# Contributing to `jtmol`

Thanks for your interest. `jtmol` is a pure-JAX port of
[tmol](https://github.com/uw-ipd/tmol); it preserves the upstream scientific
surface (residue types, score terms, kinematics) while replacing PyTorch +
custom C++/CUDA kernels with idiomatic JAX. If you are porting a new piece
of upstream code, or adding a new capability, this guide tells you what
"idiomatic JAX" means in this codebase.

## Style rules (the short version)

These are hard rules — please follow them for any new code and when editing
existing code. The rest of the guide elaborates.

1. **Pytrees over classes with mutable state.** Data structures (poses,
   kinematic trees, score states) are declared with
   [`attrs`](https://www.attrs.org/) and registered as JAX pytrees using
   the `@pytree_dataclass` helper in `jtmol.types`. Arrays are fields;
   methods are pure functions. Anything that needs to flow through `jit`,
   `vmap`, `scan`, or `grad` must be a pytree.

2. **`jax.numpy`, never `torch`.** There are no PyTorch imports anywhere
   in `jtmol`. The upstream types (`torch.Tensor`) become `jax.Array`;
   the upstream ops (`torch.where`, `torch.einsum`, …) become the
   corresponding `jax.numpy` / `jax.lax` equivalents. If you find yourself
   reaching for `torch`, stop and ask on the tracker.

3. **`jax.grad`, never hand-written backward.** Upstream tmol writes CUDA
   kernels for the forward pass *and* a matching backward kernel for the
   gradient. In `jtmol` we only write the forward pass — the gradient
   comes from `jax.grad` or `jax.value_and_grad` applied to that forward
   function. Do not write custom VJPs unless numerical stability requires
   it (e.g. a polynomial near a cutoff where naive autodiff would NaN);
   if you do, document the reason in the docstring and add a numerical
   test against `jax.grad` on a perturbed input.

4. **Every energy term returns `[n_poses]` arrays.** The public contract of
   `EnergyTerm.__call__` is
   `dict[ScoreType, jax.Array[shape=(n_poses,), dtype=float32]]`. This is
   what makes `vmap` over poses natural — the caller does not need to
   reshape, and it matches the `PoseStack` batch axis one-to-one.

5. **Pure functions inside `jit`, Python loops outside.** Scheduling,
   weight ramps, and pack/min outer loops are plain Python. Everything
   that lives inside a single minimization step, energy evaluation, or
   kinematic sweep is a pure JAX function suitable for `jit`.

## Development setup

```
pip install -e ".[dev]"
pytest jtmol/tests/ -v
```

There are no C++/CUDA build steps — installation is a normal Python
install. If you see `setup.py build_ext`-style errors, you are probably
in the wrong repository (the upstream `tmol/` submodule lives alongside
the port and has its own build system).

## Adding a new energy term

Each energy term is a subclass of `jtmol.score.energy_term.EnergyTerm`.
The contract is tiny — two methods plus a list:

1. `score_types_provided(self) -> list[ScoreType]` — which channels this
   term produces. A single term can own multiple (e.g. `LJLKEnergyTerm`
   owns `fa_ljatr`, `fa_ljrep`, and `fa_lk`).
2. `setup_for_pose_stack(self, pose_stack, pbt=None) -> pytree` — run
   once per `PoseStack`. Precompute any conformation-independent arrays
   (parameter gathers, neighbor lists, bonded-atom indices, etc.) and
   return them as a pytree. Do *not* touch `pose_stack.coords` here.
3. `__call__(self, pose_stack, term_state, coords_override=None)
   -> dict[ScoreType, jax.Array]` — the hot path. Pure JAX. Returns
   `{ScoreType: float32[n_poses]}`. Every call that reads coordinates
   must honour `coords_override` (the minimizer uses it so
   `jax.grad(fn)(coords)` traces cleanly).

Skeleton:

```python
# jtmol/score/my_term/my_energy_term.py
from typing import Any, Optional

import jax
import jax.numpy as jnp

from jtmol.score.energy_term import EnergyTerm
from jtmol.score.score_types import ScoreType


class MyEnergyTerm(EnergyTerm):
    def score_types_provided(self) -> list[ScoreType]:
        return [ScoreType.my_channel]

    def setup_for_pose_stack(self, pose_stack: Any, pbt: Any = None):
        # Gather per-type parameters, bonded-atom tables, etc.
        return {"params": jnp.asarray(...)}

    def __call__(
        self,
        pose_stack: Any,
        term_state,
        coords_override: Optional[jax.Array] = None,
    ) -> dict[ScoreType, jax.Array]:
        coords = coords_override if coords_override is not None else pose_stack.coords
        # ... pure-JAX math ...
        energy_per_pose = jnp.sum(...)  # shape [n_poses]
        return {ScoreType.my_channel: energy_per_pose.astype(jnp.float32)}
```

Then register the term in two places:

1. **Add the channel to `jtmol.score.score_types.ScoreType`.** Append
   the new member *before* `n_score_types` and do not reorder existing
   members (their values are part of the on-disk contract).

2. **Register the term in `jtmol.score.score_function._TERM_FOR_TYPE`.**
   The value is a string `"package.module:ClassName"`. The score function
   lazily imports the term the first time `set_weight` is called for any
   of its channels.

3. **Re-export the term from `jtmol/score/my_term/__init__.py`** so
   `from jtmol.score.my_term import MyEnergyTerm` works. Optionally,
   re-export from `jtmol/score/__init__.py` if the term is part of a
   "classic" preset.

Finally, add a smoke test under `jtmol/tests/`:

```python
def test_my_term(pose_stack):
    from jtmol.score.my_term import MyEnergyTerm
    from jtmol import ParameterDatabase
    term = MyEnergyTerm(ParameterDatabase.get_default())
    state = term.setup_for_pose_stack(pose_stack)
    out = term(pose_stack, state)
    assert out[ScoreType.my_channel].shape == (pose_stack.n_poses,)
```

## Checklist before opening a PR

- [ ] No `import torch` / `torch.*` references added.
- [ ] New data classes use `attrs` + `@pytree_dataclass`.
- [ ] New energy terms return `float32[n_poses]` per channel.
- [ ] Gradient path goes through `jax.grad`; no custom VJP unless
      numerical stability requires it (document why if so).
- [ ] `pytest jtmol/tests/` passes (skips are OK for
      not-yet-ported layers; failures are not).
- [ ] If you added a new `ScoreType`, it is appended before
      `n_score_types` and registered in `_TERM_FOR_TYPE`.

Questions? Open an issue; we triage weekly.
