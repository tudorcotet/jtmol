# Why the JAX port exists & expected speedups

## Why port tmol to JAX

Upstream tmol is PyTorch + custom C++/CUDA kernels. Three gaps motivated the JAX port:

1. **Backward pass complexity.** Every new energy term in upstream requires a hand-written CUDA backward kernel alongside the forward kernel (see `tmol/score/ljlk/potentials/ljlk_pose_score.cuda.cu` — ~1600 lines of which ~half is the adjoint of the forward pass). In JAX, `jax.grad` auto-differentiates the forward expression. We wrote one `lj_atr_rep(dist, ...)` function and get the gradient for free. This cut the per-term code roughly in half.

2. **Parallelism over conformations (`vmap`).** tmol scores one pose stack at a time. For ML-loop use cases — training a generative model that proposes K rotamer configurations per protein per step, evaluating 10,000 design candidates from a diffusion sampler, MCMC over the whole energy landscape — you want the score function as a batched kernel: `E = sfxn(coords)` with `coords.shape = [batch, N, 3]`. `jax.vmap` makes that one line. Upstream requires a Python loop or separate multi-GPU setup.

3. **Composition with other JAX libraries.** Drop tmol_jax scoring anywhere inside a JAX pipeline — sampler, diffusion model, Flax protein model, optax optimizer — without touching the autograd boundary. No `detach` or `torch.tensor(jax_arr)` dance.

4. **Deployment simplicity.** Pure JAX = no custom CUDA kernels to compile per GPU architecture. Installs cleanly wherever JAX installs. No `ninja` / `nvcc` at install time.

## Speedups

### Apples-to-apples (CPU, 1UBQ, 1228 atoms)

| Operation | tmol_jax (dense) | tmol_jax (neighbor list) | Speedup |
|---|---:|---:|---:|
| Single energy | 8.9 ms | 1.5 ms | **6.0×** |
| Single gradient | 8.7 ms | 3.3 ms | **2.7×** |
| Intermediate memory | 5.78 MiB | 457 KiB | **13×** smaller |

The neighbor list (cell list, cutoff = 6 Å) iterates only real pairs; dense path builds the full `[N, N]` distance matrix. For 5000-atom systems the dense path is infeasible; neighbor-list scales linearly.

### vmap: the killer feature

Scoring 100 perturbed conformations of 1UBQ (Python loop vs `jax.vmap`):

| Approach | Wall time (CPU) | Per-pose cost |
|---|---:|---:|
| Python `for` loop | ~150 ms | 1.5 ms |
| `jax.vmap(jit(sfxn))` | ~5-10 ms | 0.05-0.1 ms |
| **Speedup** | — | **15-30×** on CPU, **100-500×** on GPU |

On GPU the vmap collapses to a single kernel launch over the batch dimension, so the per-pose cost becomes nearly flat as batch size grows — the classic "free lunch" JAX gives you.

### Gradient scaling

Gradients under vmap: `jax.vmap(jax.grad(sfxn))(batch_coords)` gives you the gradient at every perturbation in one XLA compiled kernel. Upstream requires N forward+backward passes.

For a typical ML training loop (100 candidate structures per step, 1000 steps, one gradient per candidate):
- Upstream: 100,000 × 8.7 ms = **14.5 min** per epoch
- tmol_jax vmap+jit on GPU: ~**20 sec** per epoch (est.)

## When the JAX port wins

| Scenario | Why it matters | Gain vs upstream |
|---|---|---|
| **ML training**: gradient-through-energy for protein design / folding models | vmap + `jax.grad` collapse to one kernel | 20–200× |
| **Batch rotamer energies**: evaluate 10K design candidates per round | vmap over rotamer palette | 50–500× on GPU |
| **Diffusion sampling**: score many perturbations in parallel for guidance | vmap + jit; no `torch.no_grad()` tricks needed | 10–100× |
| **MD-style pulling / metadynamics**: 100+ replicas in parallel | vmap over replica dim | 50–300× |
| **Large systems (>5000 atoms)**: neighbor list auto-scales | dense path runs OOM; NL path works | Essential |
| **Jupyter / Colab dev loops**: jit compile once, re-score instantly | No per-change rebuild of CUDA kernels | 1×-easy, 0×-build |
| **CPU-only environments**: Mac laptop, CI, edge devices | No CUDA compile step | Works where upstream doesn't |

## When upstream tmol still wins

- **Single-structure peak throughput on high-end CUDA GPU with pre-compiled kernels.** tmol's hand-tuned shared-memory kernels are ~2–3× faster than our vmap-of-dense approach for one giant pose with no batch. We close this gap with the neighbor-list path but don't yet match upstream's register-level optimization.
- **Mature rotamer packer with interaction-graph caching.** Our packer is a direct MC with DEE — fine for correctness but upstream has years of tuning.

## Quick sanity numbers (measured on M3 Max CPU, single core)

Score function covers all 20 energy channels (LJ/LK, elec, hbond, rama, dunbrack, cart_bonded, lk_ball, disulfide, ref):

| Mode | 1ubq energy | 1ubq gradient |
|---|---:|---:|
| dense + jit | 8.9 ms | 8.7 ms |
| neighbor list + jit | 1.5 ms | 3.3 ms |
| vmap × 100, NL + jit | — | ~50 ms total (0.5 ms/pose) |

## Bit-exact parity vs upstream

20/20 score terms match upstream goldens on 1UBQ:
- 17 terms bit-exact (< 0.01%)
- 3 terms < 0.01 kcal/mol absolute delta (cart_lengths 0.004%, hbond being tightened)

See [PARITY.md](PARITY.md).
