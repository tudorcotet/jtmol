# jtmol

A pure-JAX port of [tmol](https://github.com/uw-ipd/tmol), the
GPU-accelerated reimplementation of Rosetta's `beta_nov2016_cart`
energy function from the Baker Lab. Upstream is PyTorch with
hand-written C++/CUDA kernels; jtmol replaces both with `jit`, `vmap`,
`lax.scan`, and `jax.grad` over pure Python.

The port makes Rosetta scoring composable with the rest of the JAX ML
stack (AlphaFold, ESM, Boltz, etc.) so Rosetta energies can act as a
differentiable loss in protein design loops without the autograd-boundary
friction of going through torch.

The package imports as `jtmol`; the upstream package it ports is `tmol`.

```python
import jtmol

ps   = jtmol.pose_stack_from_pdb("1ubq.pdb")
sfxn = jtmol.beta2016_score_function()

E    = sfxn(ps)                              # [n_poses]
ps_m = jtmol.run_cart_min(ps, sfxn)       # L-BFGS Cartesian minimize
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

For protein-ligand support (loading complexes with small-molecule
ligands), install the optional `ligand` extra, which pulls in RDKit:

```bash
uv pip install -e '.[ligand]'
```

See [Protein-ligand status](#protein-ligand-status) for what this
currently covers.

## Performance

jtmol evaluates the full `beta_nov2016_cart` (22 channels) as a single
fused computation over a batch of poses. Each large pair term (LJLK,
elec, lk_ball) runs on its own O(N·k_max) cell list rather than a dense
O(N²) tensor; `jtmol.nonbonded.fast` provides a parity-exact fast engine
for batched scoring (`jtmol/tests/test_nonbonded_fast.py` gates it against
the dense path in float64).

The table below compares that engine to upstream tmol on the full 22-term
`beta2016` score, in three batched regimes. **Value+grad** is the gradient
of the summed energy w.r.t. coordinates — what an optimizer or a
differentiable design loop needs. Ratios are `tmol / jtmol`, so **> 1 means
jtmol is faster**. A100-80GB, float32, warm ABBA-interleaved timing:

| regime | what it is | forward | value+grad |
|---|---|---:|---:|
| **S1** | one structure, K=1 (single-pose latency) | 2.6× | 1.7× |
| **S2** | one topology, K=128 conformations (ensemble / relax) | 2.1× | 1.1× |
| **S3** | 128 distinct structures (heterogeneous batch) | ~0.7× | 1.1× |

Notes, in the interest of accuracy: jtmol scores on a neighbor list built
once per batch (amortized across conformations), so these ratios reflect
batched/repeated scoring, not a per-call kernel comparison. S1/S2 are
verified equal to upstream per channel on the same coordinates; S3 compares
equal-sized workloads on each backend's own representation (the two loaders
place hydrogens slightly differently), and jtmol's forward on the
heterogeneous batch is slower than tmol's streaming kernels. Every jtmol
optimization is bit-exact against the dense substrate.

Reproduce with:

```bash
python benchmarks/bench_fast_vs_tmol.py --pdb-dir /path/to/distinct/pdbs
```

(S1/S2 use the bundled 1UBQ; S3 needs a directory of distinct structures,
e.g. the panel in `benchmarks/upstream_exact_panel_v2.json`. Requires a
CUDA GPU with `jax[cuda12]` and the tmol wheel.)

## Parity

> **Parity scope.** beta2016 has 22 active weighted channels. Exact
> parity holds on the cases documented below; it is not claimed in
> general. Known limitations: jtmol does not carry typed bond order,
> ringness, or connection bond types; its default chemical YAML is not
> byte-identical to upstream; and 29 of 128 cross-checked structures
> reconstruct native coordinates outside a 1e-4 Å tolerance.

On 1ubq, forward energies match upstream's per-term goldens across all
20 score channels of `beta_nov2016_cart`, and `jax.grad` matches
central finite-difference of upstream's own forward to ~1e-9 in
float64.

**Multi-chain complexes are in parity.** 1CGI / 1PRG / 7OPB split into
their chains at the correct residue indices, `inter_residue_connections`
opens at each break, and the count-pair machinery treats the chains as
disconnected. All interface (cross-chain pairwise) channels
(`fa_ljatr/ljrep/lk`, `fa_elec`, `hbond`, `lk_ball*`) match upstream to
~1e-4. See [`docs/MULTI_CHAIN_PARITY.md`](docs/MULTI_CHAIN_PARITY.md).

The remaining per-term divergences originate upstream, not in the port:

- `cart_lengths` / `cart_angles`: jtmol matches upstream's YAML goldens
  exactly (+37.76 / +183.57 on 1ubq); the pinned `v0.1.22` wheel has a
  PBT-cache bug that makes it emit 0 (fixed in upstream master by #383).
- Cubic-Hermite kink-term gradients (`omega`, `cart_torsions`,
  `dunbrack_rotdev`, …) carry a float32-internal-adjoint floor inside
  upstream's compiled kernels, unpatchable from jtmol.

`dunbrack` rotamer scoring uses the same bicubic B-spline interpolation
as upstream (rotamer −log P, chi means, chi sigmas all evaluated through
`_bicubic_bspline_interpolate` over bspline-prefiltered tables). The one
residual is a per-PDB edge case: on 3plc, two residues bracketing a
missing-density gap (THR-26 / ASP-35) get synthesized phi/psi that land
in the grid interior, where jtmol's faithful-port value diverges from a
stale/inconsistent upstream golden (`docs/UPSTREAM_BUGS.md` #6,
root-caused upstream-side, one PDB).
Full per-term table and bug catalogue:
[`docs/PARITY.md`](docs/PARITY.md), [`docs/UPSTREAM_BUGS.md`](docs/UPSTREAM_BUGS.md).

One channel worth flagging: upstream's hand-written CUDA adjoint for
`lk_bridge_uncpl` disagrees with finite-difference of its own forward
at sp2 backbone-O acceptors, by up to 0.16 kcal/mol/Å on the carbonyl
O atoms of GLN-1 and GLN-61 on 1ubq. jtmol's `jax.grad` matches FD to
1e-9 there. Reproducer and analysis in
[`docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md`](docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md).

It is one of six upstream bugs catalogued by the port. The full list is
in [`docs/UPSTREAM_BUGS.md`](docs/UPSTREAM_BUGS.md)
with reproducers and workarounds; each one drives a category of
"expected mismatch" cell in the parity table in
[`docs/PARITY.md`](docs/PARITY.md).

## What's in the package

| Module | What it does |
|---|---|
| `jtmol.pose` | `Pose`, `PoseStack`, `PackedBlockTypes` as JAX pytrees |
| `jtmol.chemical` | `ChemicalDatabase` port; typed bond/ring/connection metadata is not yet in exact upstream parity |
| `jtmol.kinematics` | `KinForest` + `forward_kin` / `inverse_kin` via `lax.scan` |
| `jtmol.score` | The 22 active weighted channels of current upstream `beta2016.sfxn` |
| `jtmol.optimization` | L-BFGS, Cartesian and kinematic |
| `jtmol.pack` | Rotamer packer (Metropolis-MC over jitted score) |
| `jtmol.relax` | FastRelax |
| `jtmol.io` | PDB read/write, canonical ordering, biotite + ligand pose construction |
| `jtmol.ligand` | Ligand params: parse `.tmol`, build one from a MOL2 or SMILES (optional `ligand` extra) |

## Protein-ligand status

jtmol loads a protein-ligand complex, places the ligand in the
`PoseStack`, scores it across the main pairwise channels, and builds the
ligand params file itself from a MOL2 or a SMILES string. Verified on the
cdk2 complex (280 residues): the ligand block is constructed with its 28
atoms at crystal coordinates, and protein-only parity is unchanged (the
panel snapshot passes).

What works:

- **Pose construction**: `pose_stack_from_biotite(atoms,
  ligand_params_files=[...])` builds a `PoseStack` containing the ligand
  residue.
- **MOL2 or SMILES to scoreable params**:
  `jtmol.ligand.mol2_to_params("lig.mol2", "out.tmol", "LG1")` and
  `smiles_to_params(smiles, "out.tmol", "LG1")` type the atoms, build the
  ICOOR tree, compute MMFF94 charges, and write a `.tmol` that pose
  construction consumes directly. The MOL2 output is score-identical to
  the shipped cdk2 params on the real complex: every structural, LJ, LK,
  hbond, and gen-bonded term is byte-identical, and `fa_elec` differs by
  2e-4 from charge rounding. The ICOOR tree round-trips through jtmol's
  own coordinate builder. Needs the optional `ligand` extra (RDKit).
- **LJ / LK**: generic-potential ligand atom types (`CR`, `Nim`, `ClR`)
  carry parameters in `chemical.yaml` and `ljlk.yaml`; all 28 cdk2 ligand
  atoms resolve to real LJ/LK params and contribute.
- **Electrostatics**: per-atom partial charges flow into `fa_elec`. The
  cdk2 ligand contributes about −4 kcal; protein elec is unchanged.
- **Hydrogen bonds**: the generic hbond block is ported and ligand N/O
  atoms participate. The cdk2 ligand contributes about −2.6 kcal; protein
  hbond is unchanged.
- **Atom typing**: `assign_atom_types_from_mol2(path)` assigns Rosetta
  generic-potential atom types (a faithful RDKit port of upstream's
  `atom_typing.py`), verified 28/28 on `cdk2.lig.mol2`.

The param builder is breadth-checked on ten diverse ligands (fused
aromatics, a twelve-membered macrocycle, halogens, sulfur, guanidinium,
phosphate): all type, carry real MMFF94 charges, and produce
rebuild-valid ICOOR trees.

Not done yet, each scoped and none silently approximated:

- **Parameterized covalent states**: an explicit Ser-OG--ligand connection
  can be loaded in `covalent_state_mode="topology_only"` for mapped-atom and
  bond-graph validation. Scoring is locked until bond order, state-specific
  cartbonded parameters, and complete charges are consumed. See
  [the covalent-state boundary](docs/COVALENT_STATE_TOPOLOGY.md).
- **CIF perception and rotatable-bond DOFs**: the MOL2 and SMILES paths
  cover the common case. Reading a coordinate-only ligand (bond
  perception from geometry) and emitting chi-topology for flexible
  ligands are deferred. Chi-topology also needs a torsional-DOF consumer,
  which jtmol does not have yet.
- **Per-complex numerical parity of elec/hbond**: the MOL2 params are
  score-identical to the shipped cdk2 reference, but elec/hbond magnitudes
  on other complexes are sanity-checked, not golden-matched (no upstream
  golden is bundled). LJ/LK and atom typing are parity-checked.

## A few common recipes

### Score and minimize

```python
import jax
import jtmol

ps   = jtmol.pose_stack_from_pdb("1ubq.pdb")
sfxn = jtmol.beta2016_score_function()

E    = sfxn(ps)
ps_m = jtmol.run_cart_min(ps, sfxn)
jtmol.write_pose_stack_pdb(ps_m, "out.pdb")
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

Upstream needs a Python loop over the batch dimension; jtmol compiles
the whole batch into one XLA kernel.

## Documentation map

| File | Topic |
|---|---|
| [`docs/PARITY.md`](docs/PARITY.md) | Term-by-term parity table vs upstream tmol |
| [`docs/MULTI_PDB_PARITY.md`](docs/MULTI_PDB_PARITY.md) | Single-chain multi-PDB cross-check |
| [`docs/MULTI_CHAIN_PARITY.md`](docs/MULTI_CHAIN_PARITY.md) | Multi-chain / interface PDBs |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Package layout and pytree design |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | The 20 energy channels, in one page |
| [`docs/WHY_JAX.md`](docs/WHY_JAX.md) | Rationale for the JAX rewrite |
| [`docs/PORT_NOTES.md`](docs/PORT_NOTES.md) | Gaps and behavioural differences vs upstream |
| [`docs/UPSTREAM_BUGS.md`](docs/UPSTREAM_BUGS.md) | Catalogue of upstream tmol bugs surfaced by the port (6 distinct issues + 1 divergence, each with a reproducer) |
| [`docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md`](docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md) | The `lk_bridge_uncpl` adjoint bug, full reproducer |

### Worked examples

| Script | Topic |
|---|---|
| [`examples/minimize_1ubq.py`](examples/minimize_1ubq.py) | Single-PDB Cartesian L-BFGS minimization |
| [`examples/vmap_over_poses.py`](examples/vmap_over_poses.py) | `jit(vmap)` over a pose-stack batch |
| [`examples/neighbor_list_benchmark.py`](examples/neighbor_list_benchmark.py) | Dense vs cell-list timing on 1ubq |
| [`examples/differentiable_folding.py`](examples/differentiable_folding.py) | Gradient through forward kinematics |
| [`examples/vmap_batch_scoring.py`](examples/vmap_batch_scoring.py) | Batched scoring over a coordinate batch |

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
