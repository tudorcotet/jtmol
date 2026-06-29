# jtmol vs upstream tmol — A100 throughput sweep

Test target: **1ubq** (76 residues, 1231 atoms).  Modal cloud (A100-40GB).
Precision: float32 geometry, float64 accumulator (NequIP-style mixed).
Repeats: 5, amortized after one warmup call.

The two improvements: (a) `tmol_jax.score.batched_energy` wraps a score
function in `jax.lax.map(chunk_size=16)` so peak memory stays bounded by
the chunk, K is unbounded; (b) `pmap_batched_energy` adds device-axis
sharding so K is split across visible GPUs.

## Single-GPU sweep (A100-40GB, NL off)

Source: `examples/modal_bench_run.py::run_bench`.

**Single pose**: jtmol E=2.04 ms / grad=7.83 ms, upstream E=34.96 ms / grad=42.42 ms.

| K | jtmol E (lax.map) | upstream E (loop) | speedup | jtmol grad | upstream grad | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.214 | 26.786 | 12.1× | 8.922 | 36.069 | 4.0× |
| 4 | 1.351 | 27.057 | 20.0× | 7.264 | 34.623 | 4.8× |
| 16 | 1.270 | 28.706 | 22.6× | 4.979 | 32.854 | 6.6× |
| 64 | 1.097 | 28.976 | 26.4× | 4.849 | 33.829 | 7.0× |
| 256 | 1.074 | 29.305 | 27.3× | 4.823 | 34.775 | 7.2× |
| 1024 | 1.072 | 30.223 | **28.2×** | 4.823 | 32.873 | **6.8×** |

## Multi-GPU pmap sweep (A100-40GB:2, NL off)

Source: `examples/modal_bench_run.py::run_pmap`.

Forward only. The 1-GPU column inside this run is slightly slower than
the single-GPU bench above because JAX's 2-device init adds dispatch
overhead even when only one device is used; treat as in-run comparison
for the 2× scaling number, not cross-run.

| K | 1-GPU (ms) | 2-GPU (ms) | per-pose 1-GPU | per-pose 2-GPU | speedup |
|---:|---:|---:|---:|---:|---:|
| 64 | 84.37 | 47.26 | 1.318 | 0.738 | 1.79× |
| 256 | 326.92 | 166.22 | 1.277 | 0.649 | 1.97× |
| 1024 | 1300.30 | 653.15 | 1.270 | 0.638 | 1.99× |
| 4096 | 5196.78 | 2602.43 | 1.269 | 0.635 | **2.00×** |

## Combined headline

| Metric | jtmol | upstream tmol | speedup |
|---|---:|---:|---:|
| K=1024 forward (1 GPU) | 1.07 ms/pose | 30.22 ms/pose | 28× |
| K=1024 gradient (1 GPU) | 4.82 ms/pose | 32.87 ms/pose | 6.8× |
| K=4096 forward (2 GPU) | 0.635 ms/pose | ~30 ms/pose (extrap.) | **~47×** |

**Wall-clock to score 4096 1ubq-sized binders, forward only**: jtmol on 2 A100s = 2.6 s; upstream on 1 A100 ≈ 123 s.

## Caveats

- Neighbor-list path on LJLK is plumbed via
  `beta2016_score_function(use_neighbor_list=True)` and gives zero parity
  drift on CPU (panel snapshot verified). On GPU, the lax.map + NL
  combination currently OOMs the cell-list intermediate; debug pending.
  The current bench uses dense LJLK + lax.map, which is enough for K=1024
  on 40GB and K=4096 on 2×40GB.
- NL wiring into elec and lk_ball is deferred — substantial refactor each,
  expected another 5-15× at large N. Not needed at 1231 atoms.
- All numbers are 1ubq specifically. Pair-tensor cost scales as O(N²) on
  the dense path; expect proportional slowdown for larger structures.
  jtmol's relative advantage shrinks somewhat at higher N but stays
  substantial (kernel dispatch overhead dominates upstream at every N).
