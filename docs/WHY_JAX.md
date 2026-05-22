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

## What we've measured

Apples-to-apples on CPU (M3 Max, 1ubq, 1228 atoms):

| Operation | dense | neighbor list |
|---|---:|---:|
| Single energy | 8.9 ms | 1.5 ms |
| Single gradient | 8.7 ms | 3.3 ms |
| Intermediate memory | 5.78 MiB | 457 KiB |

The neighbor-list path (cell list, 6 Å cutoff) iterates real pairs
only; the dense path builds the full `[N, N]` distance matrix and
becomes infeasible past ~5000 atoms, where the NL path keeps scaling
linearly.

Scoring 100 perturbed 1ubq conformations on CPU:

| Approach | Wall time | Per-pose |
|---|---:|---:|
| Python `for` loop | ~150 ms | 1.5 ms |
| `jax.vmap(jit(sfxn))` | ~5–10 ms | 0.05–0.1 ms |

On GPU the vmap collapses to a single kernel launch over the batch
dim, so per-pose cost goes nearly flat as batch size grows. We
haven't benchmarked the full GPU sweep yet — when we do we'll add it
here.

## Where jtmol still loses to upstream

- **Single-structure peak throughput on a hand-tuned CUDA GPU.**
  Upstream's register-level shared-memory kernels are 2–3× faster
  than the vmap-of-dense path for one giant pose with no batch. The
  neighbor-list path narrows this but doesn't yet close it.
- **Mature rotamer packer with interaction-graph caching.** Our
  packer is direct Metropolis-MC. Correct, but upstream has years of
  C++ tuning we don't.

## Parity

20/20 score channels forward-match upstream goldens on 1ubq to
machine precision. Gradients match central finite-difference of
upstream's own forward to ~1e-9 in float64 on every term. Along the
way we found one upstream gradient bug at sp2 backbone-O acceptors
for `lk_bridge_uncpl` — jtmol returns the mathematically correct
derivative; see
[UPSTREAM_BUG_LK_BRIDGE_UNCPL.md](UPSTREAM_BUG_LK_BRIDGE_UNCPL.md).
Full per-term table: [PARITY.md](PARITY.md).
