## pmap multi-GPU sweep (1ubq, n_atoms=1231, 2 devices)

GPUs: ['cuda:0', 'cuda:1'].  Forward only.  Precision: float32 (geometry), float64 (accumulator).

| K | 1-GPU (ms) | 2-GPU (ms) | per-pose 1-GPU | per-pose 2-GPU | speedup |
|---:|---:|---:|---:|---:|---:|
| 64 | 84.37 | 47.26 | 1.318 | 0.738 | 1.79× |
| 256 | 326.92 | 166.22 | 1.277 | 0.649 | 1.97× |
| 1024 | 1300.30 | 653.15 | 1.270 | 0.638 | 1.99× |
| 4096 | 5196.78 | 2602.43 | 1.269 | 0.635 | 2.00× |