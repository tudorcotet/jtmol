# Why JAX

jtmol reimplements tmol's `beta_nov2016_cart` energy function in pure JAX.
Upstream tmol is built on PyTorch with hand-written C++/CUDA kernels; the
math is complete and the GPU implementation is well-tuned. The JAX port
targets ML protein-design loops — Boltz-2-guided hallucination, multi-term
loss composition, diffusion-time guidance — where properties of the pure-JAX
design matter more than raw single-structure kernel throughput.

## No hand-written backward kernels

Each energy term in upstream requires a paired CUDA backward pass.
`tmol/score/ljlk/potentials/ljlk_pose_score.cuda.cu` is ~1600 lines, roughly
half of which is the adjoint. In JAX, `jax.grad` autodifferentiates the
forward expression: a term is defined once as its forward function (e.g.
`lj_atr_rep(dist, ...)`) and the gradient follows automatically.

The only derivatives written by hand are at the cubic-Hermite interpolation
knots (omega, cart_torsions, cart_hxltorsions), where the autodiff path
selects the wrong one-sided derivative at the knot. Those are supplied via
`@jax.custom_jvp` using the analytical formulas of Blondel & Karplus (1996).

## Batching over conformations with `vmap`

Upstream scores one pose stack per call. Design-loop workloads instead
evaluate many conformations at once: K rotamer configurations per protein
per step, thousands of candidates from a diffusion sampler, or MCMC over the
energy landscape. `jax.vmap` turns the scalar score into a batched kernel —
`E = sfxn(coords)` with `coords.shape == [batch, N, 3]` — in one line,
without a Python loop or a separate multi-GPU setup.

## Composition with the JAX ecosystem

jtmol scoring drops into any JAX pipeline — sampler, diffusion model, Flax
model, optax optimizer — without crossing an autograd boundary. There is no
`detach` and no `torch.tensor(jax_arr)` conversion between the score function
and the surrounding model. A differentiable binding-affinity loss composed
from several energy terms follows directly from staying inside one autodiff
system.

## Simpler deployment

Pure JAX means no custom CUDA kernels to compile per GPU architecture and no
`ninja` / `nvcc` at install time. jtmol runs wherever JAX installs.

## Performance

JIT compilation collapses all score channels into a single XLA program, so a
scorer call is one kernel launch rather than the many separate ops dispatched
per call in an op-by-op eager framework. Combined with `vmap` batching, this
fits design loops that evaluate many small-to-medium structures per step.

Limitations:

- **Large structures (>2000 residues).** Once GPU compute exceeds the
  dispatch ceiling, upstream's hand-tuned tiled CUDA kernels dominate.
- **Rotamer packing.** jtmol's packer is direct Metropolis Monte Carlo —
  correct, but without the interaction-graph caching of upstream's mature
  C++ packer.

## Parity

20/20 score channels forward-match upstream goldens on 1ubq at float32-level
agreement (~1e-4–1e-5 relative). Gradients match the central finite-difference
of upstream's own forward to ~1e-9 in float64 on every term.

One known upstream gradient bug affects sp2 backbone-O acceptors for
`lk_bridge_uncpl`; jtmol returns the mathematically correct derivative. See
[UPSTREAM_BUG_LK_BRIDGE_UNCPL.md](UPSTREAM_BUG_LK_BRIDGE_UNCPL.md). Full
per-term table: [PARITY.md](PARITY.md).
