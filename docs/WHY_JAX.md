# Why we re-implemented tmol in JAX

Upstream tmol is PyTorch + hand-written C++/CUDA kernels. The math is
all there and the GPU implementation is well-tuned, but for the ML
design loops we care about (Boltz-2-driven hallucination, mosaic loss
composition, diffusion-time guidance) we kept hitting the same three
walls.

## What the port gives us

**No hand-written backward kernels.** Every new energy term in
upstream needs a paired CUDA backward — `tmol/score/ljlk/potentials/ljlk_pose_score.cuda.cu`
is ~1600 lines, roughly half of which is the adjoint. In JAX,
`jax.grad` autodifferentiates the forward expression. We wrote one
`lj_atr_rep(dist, ...)` function and got the gradient for free. The
only place we still write derivatives by hand is at the cubic-Hermite
interpolation knots (omega, cart_torsions, cart_hxltorsions), and
that is only because the autodiff path picks the wrong one-sided
derivative at the knot. We close that gap with `@jax.custom_jvp` and
Blondel & Karplus's 1996 analytical formulas.

**Parallelism over conformations is one line of `vmap`.** Upstream
scores one pose stack at a time. For the use cases we actually care
about — training a generative model that proposes K rotamer configs
per protein per step, evaluating 10,000 design candidates from a
diffusion sampler, MCMC over the energy landscape — we want
`E = sfxn(coords)` with `coords.shape = [batch, N, 3]` as a batched
kernel. `jax.vmap` makes that one line; the alternative is a Python
loop or a separate multi-GPU setup.

**Composition with the rest of the JAX stack is free.** Drop jtmol
scoring anywhere inside a JAX pipeline — sampler, diffusion model,
Flax protein model, optax optimizer — without crossing an autograd
boundary. No `detach`, no `torch.tensor(jax_arr)` dance. This is the
single biggest reason we did the port; the binding-affinity loss in
the mosaic hallucinator (see [MOSAIC_INTEGRATION.md](MOSAIC_INTEGRATION.md))
would be a much bigger lift without it.

**Deployment is simpler.** Pure JAX means no custom CUDA kernels to
compile per GPU architecture, no `ninja` / `nvcc` at install time.
Works wherever JAX installs.

## What's been measured

Head-to-head on A100-40GB with the upstream PyTorch wheel
(`tmol-0.1.22+cu128torch2.10`) and the JAX port at parity. Same
protein, same energy function (`beta_nov2016_cart`, 22 channels),
K=128 perturbed poses per call. Full details:
[BENCHMARKS.md](BENCHMARKS.md), [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

Per-pose latency across protein size, neighbor-list path on LJLK,
elec, and lk_ball:

| residues | atoms | jtmol fwd (ms/pose) | upstream fwd | speedup | jtmol grad | upstream grad | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  75 | 1116 | 0.25 | 28.3 | 111× | 1.17 | 31.5 | 27× |
| 200 | 3039 | 1.00 | 28.9 |  29× | 4.47 | 34.2 |  8× |
| 600 | 8800 | 1.41 | 28.8 |  20× | 13.9 | 32.5 |  2× |

Per-pose forward latency stays under 1.5 ms across an 8× range of
protein size. Upstream stays at ~30 ms because Python + PyTorch
op-dispatch overhead dominates compute at every size tested. The
speedup is from JIT compilation collapsing all 22 channels into a
single XLA program (one kernel launch per call) rather than the
~200 separate ops PyTorch dispatches per scorer call. Detailed
analysis in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Where upstream still beats jtmol

- **Larger structures (>2000 residues).** Once upstream's GPU compute
  exceeds the dispatch ceiling, its hand-tuned tiled CUDA kernels
  start to dominate. The sweep above does not reach this regime.
- **Mature rotamer packer with interaction-graph caching.** jtmol's
  packer is direct Metropolis-MC. Correct, but upstream has years of
  C++ tuning that jtmol's pure-JAX packer does not.

## Parity

20/20 score channels forward-match upstream goldens on 1ubq to
machine precision. Gradients match central finite-difference of
upstream's own forward to ~1e-9 in float64 on every term. Along the
way we found one upstream gradient bug at sp2 backbone-O acceptors
for `lk_bridge_uncpl` — jtmol returns the mathematically correct
derivative; see
[UPSTREAM_BUG_LK_BRIDGE_UNCPL.md](UPSTREAM_BUG_LK_BRIDGE_UNCPL.md).
Full per-term table: [PARITY.md](PARITY.md).
