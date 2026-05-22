# jtmol

A pure-JAX port of [tmol](https://github.com/uw-ipd/tmol), the
GPU-accelerated reimplementation of Rosetta's `beta_nov2016_cart`
energy function from the Baker Lab. Upstream is PyTorch with
hand-written C++/CUDA kernels; jtmol replaces both with `jit`, `vmap`,
`lax.scan`, and `jax.grad` over pure Python.

The point of the port is to make Rosetta scoring composable with the
rest of the JAX ML stack (Boltz-2, mosaic, AlphaFold, ESM, etc.) so
you can use Rosetta energies as a differentiable loss in protein
design loops without the autograd-boundary friction of going through
torch. We use it as a binding-affinity loss inside mosaic binder
hallucination, see
[`docs/MOSAIC_INTEGRATION.md`](docs/MOSAIC_INTEGRATION.md).

The Python import path is `tmol_jax` (kept for parity with upstream);
the project name is **jtmol**.

```python
import tmol_jax

ps   = tmol_jax.pose_stack_from_pdb("1ubq.pdb")
sfxn = tmol_jax.beta2016_score_function()

E    = sfxn(ps)                              # [n_poses]
ps_m = tmol_jax.run_cart_min(ps, sfxn)       # L-BFGS Cartesian minimize
g    = jax.grad(lambda c: sfxn(ps.replace(coords=c)).sum())(ps.coords)
```

## Install

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/tudorcotet/jtmol
cd jtmol
uv pip install -e .
```

Or pip:

```bash
pip install -e .
```

Requires Python ≥ 3.10 and JAX ≥ 0.4.30. PyTorch is not needed at
runtime, only optionally for regenerating binary scoring tables via
`scripts/convert_binary_tables.py`.

## Status

On 1ubq, forward energies match upstream's per-term goldens to
machine precision across all 20 score channels, and `jax.grad` matches
central finite-difference of upstream's own forward to ~1e-9 in
float64 on every term we have ported. The multi-PDB panel (1qys,
1r21, 3plc, 6DMZ_A) is forward-bit-exact on the channels that don't
depend on the hbond B/B0 walk, and within ~1% on the ones that do (we
use the bond-graph walk for now; upstream uses an icoor-tree walk on
ring acceptors and sp3 hydroxyls). See
[`docs/PARITY.md`](docs/PARITY.md) for the per-term table and
[`docs/UPSTREAM_PARITY.md`](docs/UPSTREAM_PARITY.md) for the auto-
generated harness output.

One thing worth flagging: while running the gradient sweep we found
that upstream's hand-written CUDA adjoint for `lk_bridge_uncpl`
disagrees with finite-difference of its own forward at sp2 backbone-O
acceptors, by up to 0.16 kcal/mol/Å on the carbonyl O atoms of GLN-1
and GLN-61 on 1ubq. jtmol's `jax.grad` matches FD to 1e-9 there.
Reproducer and analysis in
[`docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md`](docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md).
We haven't filed it upstream yet.

## What's in the package

| Module | What it does |
|---|---|
| `tmol_jax.pose` | `Pose`, `PoseStack`, `PackedBlockTypes` as JAX pytrees |
| `tmol_jax.chemical` | `ChemicalDatabase` from upstream YAMLs verbatim |
| `tmol_jax.kinematics` | `KinForest` + `forward_kin` / `inverse_kin` via `lax.scan` |
| `tmol_jax.score` | All 20 channels of `beta_nov2016_cart` |
| `tmol_jax.optimization` | L-BFGS, Cartesian and kinematic |
| `tmol_jax.pack` | Rotamer packer (Metropolis-MC over jitted score) |
| `tmol_jax.relax` | FastRelax |
| `tmol_jax.io` | PDB read/write, canonical ordering |
| `examples/integration_mosaic_hallucinator.py` | `LossTerm` wrappers for [mosaic](https://github.com/escalante-bio/mosaic) / Boltz-2 hallucination |

## A few common recipes

### Score and minimize

```python
import jax
import tmol_jax

ps   = tmol_jax.pose_stack_from_pdb("1ubq.pdb")
sfxn = tmol_jax.beta2016_score_function()

E    = sfxn(ps)
ps_m = tmol_jax.run_cart_min(ps, sfxn)
tmol_jax.write_pose_stack_pdb(ps_m, "out.pdb")
```

### Gradients w.r.t. atomic coordinates

```python
state = sfxn.setup_for_pose_stack(ps)

def energy(coords):
    return sfxn(ps, state=state, coords_override=coords).sum()

grad_fn = jax.jit(jax.grad(energy))
g       = grad_fn(ps.coords)                  # ∂E/∂xyz
```

Going through `state` and `coords_override` lets you keep the
expensive setup (neighbor-list build, count-pair table, etc.) outside
the trace, so the jitted gradient call is just the energy math.

### vmap over a batch of conformations

```python
batch_coords = jnp.stack([
    ps.coords + 0.05 * jax.random.normal(k, ps.coords.shape)
    for k in jax.random.split(jax.random.PRNGKey(0), 64)
])

batched_E = jax.jit(jax.vmap(energy))
Es        = batched_E(batch_coords)           # [64]
```

This is the move that justifies the port. Upstream needs a Python
loop over the batch dim; we get one XLA-compiled kernel.

### As a binding-affinity loss inside protein hallucination

See [`docs/MOSAIC_INTEGRATION.md`](docs/MOSAIC_INTEGRATION.md) for the
full recipe. Three drop-in `LossTerm` subclasses (`TmolInterfaceLoss`,
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
| [`docs/WHY_JAX.md`](docs/WHY_JAX.md) | Why we re-implemented this in JAX |
| [`docs/PORT_NOTES.md`](docs/PORT_NOTES.md) | Gaps and behavioural differences vs upstream |
| [`docs/MOSAIC_INTEGRATION.md`](docs/MOSAIC_INTEGRATION.md) | Use jtmol as a binding-affinity loss in mosaic / Boltz-2 |
| [`docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md`](docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md) | A real gradient bug we found in upstream, with reproducer |

## Credit / upstream

All physics, parameters, residue-type definitions, scoring database
YAMLs, and rotamer libraries come verbatim from upstream tmol by the
[Baker Lab / UW Institute for Protein
Design](https://www.ipd.uw.edu/). jtmol is a JAX re-implementation of
their work; please cite upstream if you use jtmol in published work:

- Upstream: <https://github.com/uw-ipd/tmol>
- Rosetta `beta_nov2016`: Park *et al.*,
  *J. Chem. Theory Comput.* 2016, 12, 12, 6201–6212.

## License

Apache-2.0, matching upstream tmol. See [`LICENSE`](LICENSE).
