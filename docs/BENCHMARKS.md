# jtmol vs upstream tmol — A100 throughput sweep (1ubq, n_atoms=1231)

GPU: NVIDIA A100-SXM4-40GB.  Precision: float32.  Repeats: 5.

**Single pose**: jtmol E=2.04 ms / grad=7.83 ms
            upstream E=34.96 ms / grad=42.42 ms

## Per-pose cost (ms)

| K | jtmol E (vmap) | upstream E (loop) | speedup | jtmol grad | upstream grad | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.214 | 26.786 | 12.1× | 8.922 | 36.069 | 4.0× |
| 4 | 1.351 | 27.057 | 20.0× | 7.264 | 34.623 | 4.8× |
| 16 | 1.270 | 28.706 | 22.6× | 4.979 | 32.854 | 6.6× |
| 64 | 1.097 | 28.976 | 26.4× | 4.849 | 33.829 | 7.0× |
| 256 | 1.074 | 29.305 | 27.3× | 4.823 | 34.775 | 7.2× |
| 1024 | 1.072 | 30.223 | 28.2× | 4.823 | 32.873 | 6.8× |
