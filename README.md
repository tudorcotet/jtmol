# jtmol

**jtmol** is a pure-JAX port of [**tmol**](https://github.com/uw-ipd/tmol) —
the GPU-accelerated reimplementation of Rosetta's `beta_nov2016_cart`
energy function originally built on PyTorch + custom C++/CUDA kernels.

Same forward energies. Same gradients (modulo one documented upstream bug
we fix). Zero C++/CUDA: every kernel is `jit` + `vmap` + `lax.scan` over
pure Python/JAX, and gradients come from `jax.grad` rather than
hand-written backward kernels.

The Python import path is `tmol_jax` (kept for parity with upstream);
the public project name is **jtmol**.

```python
import tmol_jax

ps   = tmol_jax.pose_stack_from_pdb("1ubq.pdb")
sfxn = tmol_jax.beta2016_score_function()

E    = sfxn(ps)                          # total energy [n_poses]
ps_m = tmol_jax.run_cart_min(ps, sfxn)   # L-BFGS Cartesian minimize
g    = jax.grad(lambda c: sfxn(ps.replace(coords=c)).sum())(ps.coords)
```

## Install

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/tudorcotet/jtmol
cd jtmol
uv pip install -e .
```

Or with pip:

```bash
pip install -e .
```

Requires Python ≥ 3.10 and JAX ≥ 0.4.30. PyTorch is **not** needed at
runtime (only optional, for regenerating binary scoring tables).

## Status

- **Forward energy**: 20/20 channels bit-exact on a 6-PDB panel
  (`1ubq`, `1qys`, `1BL8`, `1r21`, `3plc`, `6DMZ_A`). See
  [`docs/PARITY.md`](docs/PARITY.md).
- **Gradient**: matches central finite-difference of upstream's forward
  to ~1e-9 on every term in float64. One documented upstream bug at
  sp2 backbone-O acceptors (`lk_bridge_uncpl`) — jtmol returns the
  mathematically correct gradient there, upstream's hand-written CUDA
  adjoint does not. See
  [`docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md`](docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md).
- **Multi-chain interfaces**: 1cgi, 2oob, 7opb match upstream;
  see [`docs/MULTI_CHAIN_PARITY.md`](docs/MULTI_CHAIN_PARITY.md).

## What's in the box

| Module | What it does |
|---|---|
| `tmol_jax.pose` | `Pose`, `PoseStack`, `PackedBlockTypes` (JAX pytrees) |
| `tmol_jax.chemical` | `ChemicalDatabase` from upstream YAMLs verbatim |
| `tmol_jax.kinematics` | `KinForest` + `forward_kin` / `inverse_kin` via `lax.scan` |
| `tmol_jax.score` | All 20 channels of `beta_nov2016_cart` |
| `tmol_jax.optimization` | L-BFGS (Cartesian + kinematic) |
| `tmol_jax.pack` | Rotamer packer (Metropolis-MC over jitted score) |
| `tmol_jax.relax` | FastRelax |
| `tmol_jax.io` | PDB read/write, canonical ordering |
| `examples/integration_mosaic_hallucinator.py` | `LossTerm` wrappers for [mosaic](https://github.com/escalante-bio/mosaic) / Boltz-2 hallucination |

## Quick recipes

### Score and minimize

```python
import jax
import tmol_jax

ps   = tmol_jax.pose_stack_from_pdb("1ubq.pdb")
sfxn = tmol_jax.beta2016_score_function()

E    = sfxn(ps)                          # scalar per pose
ps_m = tmol_jax.run_cart_min(ps, sfxn)   # Cartesian L-BFGS
tmol_jax.write_pose_stack_pdb(ps_m, "out.pdb")
```

### Gradients w.r.t. atomic coordinates

```python
state = sfxn.setup_for_pose_stack(ps)

def energy(coords):
    return sfxn(ps, state=state, coords_override=coords).sum()

grad_fn = jax.jit(jax.grad(energy))
g       = grad_fn(ps.coords)             # ∂E/∂xyz
```

### vmap over a batch of conformations

```python
batch_coords = jnp.stack([ps.coords + 0.05 * jax.random.normal(k, ps.coords.shape)
                          for k in jax.random.split(jax.random.PRNGKey(0), 64)])

batched_E = jax.jit(jax.vmap(energy))
Es        = batched_E(batch_coords)      # [64]
```

### Use as a binding-affinity loss in protein hallucination

See [`docs/MOSAIC_INTEGRATION.md`](docs/MOSAIC_INTEGRATION.md) for the
full recipe — three drop-in `LossTerm` subclasses (`TmolInterfaceLoss`,
`TmolBackboneLoss`, `TmolClashFilter`) that compose with mosaic /
Boltz-2 losses via `+` / `*`.

## Documentation map

| File | Topic |
|---|---|
| [`docs/PARITY.md`](docs/PARITY.md) | Term-by-term parity table vs upstream tmol |
| [`docs/UPSTREAM_PARITY.md`](docs/UPSTREAM_PARITY.md) | Auto-generated parity harness output |
| [`docs/MULTI_PDB_PARITY.md`](docs/MULTI_PDB_PARITY.md) | Single-chain multi-PDB cross-check |
| [`docs/MULTI_CHAIN_PARITY.md`](docs/MULTI_CHAIN_PARITY.md) | Multi-chain / interface PDBs |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Package layout and pytree design |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | The 20 energy channels, in one page |
| [`docs/WHY_JAX.md`](docs/WHY_JAX.md) | Design rationale: why no C++/CUDA, what `jit`+`vmap` buy |
| [`docs/PORT_NOTES.md`](docs/PORT_NOTES.md) | Gaps and behavioural differences vs upstream |
| [`docs/MOSAIC_INTEGRATION.md`](docs/MOSAIC_INTEGRATION.md) | Use jtmol as a binding-affinity loss in mosaic / Boltz-2 |
| [`docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md`](docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md) | A real gradient bug we found in upstream, with reproducer |

## Credit / upstream

All physics, parameters, residue-type definitions, scoring database
YAMLs, and rotamer libraries come **verbatim from upstream tmol** by
the [Baker Lab / UW Institute for Protein
Design](https://www.ipd.uw.edu/). jtmol is a JAX re-implementation of
their work — please cite upstream if you use jtmol:

- Upstream: <https://github.com/uw-ipd/tmol>
- Rosetta `beta_nov2016`: Park *et al.*, *J. Chem. Theory Comput.*
  2016, 12, 12, 6201–6212.

## License

Apache-2.0, matching upstream tmol. See [`LICENSE`](LICENSE).
