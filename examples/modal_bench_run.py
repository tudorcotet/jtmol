"""Modal: head-to-head jtmol vs upstream tmol scaling benchmark.

Runs on a single A100 GPU. Scores 1ubq across batch sizes
K in {1, 4, 16, 64, 256, 1024} with both backends, prints a markdown
table of per-pose wall-clock cost.

Usage::

    modal run examples/modal_bench_run.py::run_bench

Upstream tmol has no vmap, so its K>1 path is a Python loop of K
single-pose calls (which is what users actually do today). jtmol uses
``jit(vmap(sfxn))`` over the batch axis.
"""

from __future__ import annotations

import pathlib

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

TMOL_WHEEL_URL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.22/"
    "tmol-0.1.22+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
)
TMOL_SRC_REPO = "https://github.com/uw-ipd/tmol.git"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("torch==2.10.*", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("jax[cuda12]>=0.4.30")
    .pip_install(
        "numpy>=1.24", "scipy>=1.11", "attrs>=22.0", "pyyaml>=6.0",
        "networkx>=3.0", "pandas>=2.0", "biotite>=0.41.0", "typing_extensions>=4.0",
    )
    .pip_install(TMOL_WHEEL_URL)
    .run_commands(f"git clone --depth 1 --branch master {TMOL_SRC_REPO} /opt/tmol_src")
    .env({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/jtmol",
        ignore=[
            ".git/**", "tmol/**", "scratch_pdbs/**", "**/__pycache__/**",
            "*.egg-info/**", "build/**", "dist/**", ".venv/**", "venv/**",
            ".pytest_cache/**", ".ruff_cache/**",
        ],
    )
)

app = modal.App("tmol-jax-bench", image=image)

BATCH_SIZES = [1, 4, 16, 64, 256, 1024]
N_REPEATS = 5
CHUNK_SIZE = 16  # inner vmap batch under jax.lax.map; bounds peak memory


@app.function(gpu="A100-40GB", timeout=60 * 45)
def bench_one_pose(pdb_label: str = "1ubq") -> dict:
    import os, sys, time, traceback
    target = "/root/jtmol/tmol"
    if not os.path.islink(target):
        os.symlink("/opt/tmol_src", target)
    if "/root/jtmol" not in sys.path:
        sys.path.insert(0, "/root/jtmol")

    import warnings
    warnings.filterwarnings("ignore")

    import jax
    jax.config.update("jax_enable_x64", False)  # float32 for throughput parity
    import jax.numpy as jnp
    import torch
    import numpy as np

    pdb_path = f"/root/jtmol/tmol_jax/tests/data/{pdb_label}.pdb"

    out: dict = {"pdb": pdb_label, "rows": [], "errors": []}
    out["env"] = {
        "jax_devices": str(jax.devices()),
        "torch_cuda": torch.cuda.is_available(),
        "torch_device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
    }
    print(f"env: {out['env']}", flush=True)

    # =================================================================
    # jtmol side
    # =================================================================
    try:
        import tmol_jax
        from tmol_jax import beta2016_score_function

        jpose = tmol_jax.pose_stack_from_pdb(pdb_path)
        n_atoms = int(jpose.coords.shape[1])
        out["n_atoms"] = n_atoms
        print(f"n_atoms={n_atoms}", flush=True)

        # NL is opt-in via beta2016_score_function(use_neighbor_list=True).
        # Disabled in the bench: it currently OOMs on GPU when combined with
        # lax.map chunking (the lax.map carry inflates intermediate state of
        # the NL pair gather). Standalone NL on CPU is fine (panel snapshot
        # passes, zero parity drift). Tracked separately; lax.map alone
        # already solved the dense-path OOM at K=1024.
        sfxn = beta2016_score_function()
        state = sfxn.setup_for_pose_stack(jpose)
        base = jnp.asarray(jpose.coords)

        def energy(coords):
            return sfxn(jpose, state=state, coords_override=coords).sum()

        jit_e = jax.jit(energy)
        jit_g = jax.jit(jax.grad(energy))

        # warmup single
        jax.block_until_ready(jit_e(base))
        jax.block_until_ready(jit_g(base))

        # single-pose timing
        t0 = time.perf_counter()
        for _ in range(N_REPEATS):
            jax.block_until_ready(jit_e(base))
        jtmol_single_e = (time.perf_counter() - t0) / N_REPEATS * 1000

        t0 = time.perf_counter()
        for _ in range(N_REPEATS):
            jax.block_until_ready(jit_g(base))
        jtmol_single_g = (time.perf_counter() - t0) / N_REPEATS * 1000

        out["jtmol_single_e_ms"] = jtmol_single_e
        out["jtmol_single_g_ms"] = jtmol_single_g
        print(f"jtmol single: E={jtmol_single_e:.2f}ms grad={jtmol_single_g:.2f}ms", flush=True)

        # batched sweep via jax.lax.map (chunked vmap, f32 coord cast).
        # This is the lazy fix for the f64-dense OOM: each chunk is
        # vmap'd (one XLA kernel) and chunks share the trace via lax.map,
        # so peak memory is bounded by CHUNK_SIZE, not K.
        from tmol_jax.score import batched_energy, batched_value_and_grad
        for K in BATCH_SIZES:
            try:
                key = jax.random.PRNGKey(K)
                noise = jax.random.normal(key, (K,) + base.shape) * 0.05
                batch = base[None] + noise

                bfn_e = jax.jit(batched_energy(sfxn, jpose, state=state, chunk_size=CHUNK_SIZE, score_dtype=jnp.float32))
                bfn_g = jax.jit(batched_value_and_grad(sfxn, jpose, state=state, chunk_size=CHUNK_SIZE, score_dtype=jnp.float32))

                # warmup compile
                jax.block_until_ready(bfn_e(batch))
                jax.block_until_ready(bfn_g(batch))

                t0 = time.perf_counter()
                for _ in range(N_REPEATS):
                    jax.block_until_ready(bfn_e(batch))
                e_ms = (time.perf_counter() - t0) / N_REPEATS * 1000

                t0 = time.perf_counter()
                for _ in range(N_REPEATS):
                    jax.block_until_ready(bfn_g(batch))
                g_ms = (time.perf_counter() - t0) / N_REPEATS * 1000

                row = {
                    "K": K,
                    "jtmol_total_e_ms": e_ms,
                    "jtmol_total_g_ms": g_ms,
                    "jtmol_per_pose_e_ms": e_ms / K,
                    "jtmol_per_pose_g_ms": g_ms / K,
                }
                out["rows"].append(row)
                print(
                    f"jtmol K={K:5d}: E={e_ms:8.2f}ms ({e_ms/K:6.3f}/pose)  "
                    f"grad={g_ms:8.2f}ms ({g_ms/K:6.3f}/pose)",
                    flush=True,
                )
            except Exception as exc:
                out["errors"].append(f"jtmol K={K}: {exc!r}")
                print(f"jtmol K={K} FAILED: {exc!r}", flush=True)
                continue
    except Exception as exc:
        out["errors"].append(f"jtmol setup: {exc!r}\n{traceback.format_exc()}")
        print(f"jtmol setup FAILED: {exc!r}", flush=True)

    # =================================================================
    # upstream tmol side
    # =================================================================
    try:
        # Hide the JAX path so torch tmol doesn't fight over the GPU mem.
        # Don't add /root/jtmol/tmol (that would shadow the pip-installed
        # wheel — same dance as the parity script).
        import tmol
        from tmol import pose_stack_from_pdb as up_pose_stack
        from tmol.score.score_function import ScoreFunction as UpScoreFunction

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_pose = up_pose_stack(pdb_path, device)
        print(f"upstream loaded: n_atoms={torch_pose.coords.shape[1]} device={torch_pose.coords.device}", flush=True)

        sfxn_up = UpScoreFunction(tmol.ParameterDatabase.get_default(), device)
        # beta_nov2016_cart channel weights
        from tmol.score.score_types import ScoreType as UpST
        weights = {
            UpST.fa_ljatr: 1.0, UpST.fa_ljrep: 0.55, UpST.fa_lk: 1.0,
            UpST.fa_elec: 1.0, UpST.hbond: 1.0,
            UpST.lk_ball_iso: -0.38, UpST.lk_ball: 0.92, UpST.lk_bridge: -0.33, UpST.lk_bridge_uncpl: -0.33,
            UpST.rama: 0.5, UpST.omega: 0.48,
            UpST.dunbrack_rot: 0.76, UpST.dunbrack_rotdev: 0.69, UpST.dunbrack_semirot: 0.78,
            UpST.cart_lengths: 0.5, UpST.cart_angles: 0.5, UpST.cart_torsions: 0.5,
            UpST.cart_impropers: 0.5, UpST.cart_hxltorsions: 0.5,
            UpST.ref: 1.0,
        }
        for st, w in weights.items():
            sfxn_up.set_weight(st, w)

        scorer = sfxn_up.render_whole_pose_scoring_module(torch_pose)

        def up_energy(coords_t):
            # coords_t shape: [1, n_atoms, 3]
            return scorer(coords_t).sum()

        base_t = torch_pose.coords.detach().clone().requires_grad_(True)

        # warmup
        for _ in range(2):
            e = up_energy(base_t)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # single-pose timing. Forward measured under no_grad so upstream
        # does not build an autograd tape it never uses -- jtmol's forward
        # path (batched_energy) builds no tape either, so this keeps the
        # forward-vs-forward comparison fair. Gradient timing below keeps
        # grad enabled on both sides.
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(N_REPEATS):
                e = up_energy(base_t)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
        up_single_e = (time.perf_counter() - t0) / N_REPEATS * 1000

        t0 = time.perf_counter()
        for _ in range(N_REPEATS):
            base_t.grad = None
            e = up_energy(base_t)
            e.backward()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        up_single_g = (time.perf_counter() - t0) / N_REPEATS * 1000

        out["upstream_single_e_ms"] = up_single_e
        out["upstream_single_g_ms"] = up_single_g
        print(f"upstream single: E={up_single_e:.2f}ms grad={up_single_g:.2f}ms", flush=True)

        # K-sweep: upstream is a Python loop of K single-pose calls
        for row in out["rows"]:
            K = row["K"]
            try:
                # generate K perturbed coord tensors
                rng = np.random.default_rng(K)
                noise_np = rng.normal(0, 0.05, (K,) + tuple(base_t.shape)).astype(np.float32)
                coord_batch = [
                    torch.from_numpy(base_t.detach().cpu().numpy() + noise_np[i]).to(device).requires_grad_(True)
                    for i in range(K)
                ]

                # warmup
                for c in coord_batch[: min(K, 2)]:
                    _ = up_energy(c)
                torch.cuda.synchronize() if torch.cuda.is_available() else None

                # E loop (forward-only, no autograd tape)
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(N_REPEATS):
                        for c in coord_batch:
                            _ = up_energy(c)
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                e_ms = (time.perf_counter() - t0) / N_REPEATS * 1000

                # grad loop
                t0 = time.perf_counter()
                for _ in range(N_REPEATS):
                    for c in coord_batch:
                        c.grad = None
                        e = up_energy(c)
                        e.backward()
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                g_ms = (time.perf_counter() - t0) / N_REPEATS * 1000

                row["upstream_total_e_ms"] = e_ms
                row["upstream_total_g_ms"] = g_ms
                row["upstream_per_pose_e_ms"] = e_ms / K
                row["upstream_per_pose_g_ms"] = g_ms / K
                print(
                    f"upstream K={K:5d}: E={e_ms:9.2f}ms ({e_ms/K:6.3f}/pose)  "
                    f"grad={g_ms:9.2f}ms ({g_ms/K:6.3f}/pose)",
                    flush=True,
                )
                # K=1024 will be slow; bail early if a single K takes >5 min
                if g_ms > 300_000:
                    print(f"  bailing: K={K} grad took >5min/iter, larger K would not finish", flush=True)
                    break
            except Exception as exc:
                out["errors"].append(f"upstream K={K}: {exc!r}")
                print(f"upstream K={K} FAILED: {exc!r}", flush=True)
                continue
    except Exception as exc:
        out["errors"].append(f"upstream setup: {exc!r}\n{traceback.format_exc()}")
        print(f"upstream setup FAILED: {exc!r}", flush=True)

    return out


def _format_markdown(result: dict) -> str:
    lines = []
    lines.append(f"# jtmol vs upstream tmol — A100 throughput sweep ({result['pdb']}, n_atoms={result.get('n_atoms','?')})")
    lines.append("")
    lines.append(f"GPU: {result['env'].get('torch_device','?')}.  Precision: float32.  Repeats: {N_REPEATS}.")
    lines.append("")
    if "jtmol_single_e_ms" in result:
        lines.append(f"**Single pose**: jtmol E={result['jtmol_single_e_ms']:.2f} ms / grad={result['jtmol_single_g_ms']:.2f} ms")
    if "upstream_single_e_ms" in result:
        lines.append(f"            upstream E={result['upstream_single_e_ms']:.2f} ms / grad={result['upstream_single_g_ms']:.2f} ms")
    lines.append("")
    lines.append("## Per-pose cost (ms)")
    lines.append("")
    lines.append("| K | jtmol E (vmap) | upstream E (loop) | speedup | jtmol grad | upstream grad | speedup |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["rows"]:
        je = row.get("jtmol_per_pose_e_ms")
        ue = row.get("upstream_per_pose_e_ms")
        jg = row.get("jtmol_per_pose_g_ms")
        ug = row.get("upstream_per_pose_g_ms")
        e_sp = f"{ue/je:.1f}×" if (je and ue) else "—"
        g_sp = f"{ug/jg:.1f}×" if (jg and ug) else "—"
        lines.append(
            f"| {row['K']} | "
            f"{je:.3f} | {ue:.3f} | {e_sp} | "
            f"{jg:.3f} | {ug:.3f} | {g_sp} |"
            if (je and ue and jg and ug) else
            f"| {row['K']} | "
            f"{je:.3f if je else 0} | {'—' if not ue else f'{ue:.3f}'} | {e_sp} | "
            f"{jg:.3f if jg else 0} | {'—' if not ug else f'{ug:.3f}'} | {g_sp} |"
        )
    lines.append("")
    if result.get("errors"):
        lines.append("## Errors")
        for e in result["errors"]:
            lines.append(f"- {e}")
    return "\n".join(lines)


@app.function(gpu="A100-40GB:2", timeout=60 * 30)
def bench_multi_gpu(pdb_label: str = "1ubq") -> dict:
    """pmap'd jtmol on 2× A100. K in {64, 256, 1024, 4096}.

    Each device handles K/n_dev poses; within each device the
    ``lax.map`` chunking is the same as the single-GPU bench. We
    expect ~2× speedup at large K once data-transfer + barrier
    overhead is amortized.
    """
    import os, sys, time, traceback
    target = "/root/jtmol/tmol"
    if not os.path.islink(target):
        os.symlink("/opt/tmol_src", target)
    if "/root/jtmol" not in sys.path:
        sys.path.insert(0, "/root/jtmol")

    import warnings
    warnings.filterwarnings("ignore")

    import jax
    import jax.numpy as jnp
    import numpy as np

    out: dict = {"pdb": pdb_label, "rows": [], "errors": []}
    n_dev = jax.device_count()
    out["n_devices"] = n_dev
    out["devices"] = [str(d) for d in jax.devices()]
    print(f"n_devices={n_dev} devices={jax.devices()}", flush=True)

    pdb_path = f"/root/jtmol/tmol_jax/tests/data/{pdb_label}.pdb"

    try:
        import tmol_jax
        from tmol_jax import beta2016_score_function
        from tmol_jax.score import batched_energy, pmap_batched_energy

        jpose = tmol_jax.pose_stack_from_pdb(pdb_path)
        n_atoms = int(jpose.coords.shape[1])
        out["n_atoms"] = n_atoms

        sfxn = beta2016_score_function(use_neighbor_list=True)
        state = sfxn.setup_for_pose_stack(jpose)
        base = jnp.asarray(jpose.coords)

        K_list = [64, 256, 1024, 4096]
        # only K's divisible by n_dev
        K_list = [K for K in K_list if K % n_dev == 0]

        for K in K_list:
            try:
                key = jax.random.PRNGKey(K)
                noise = jax.random.normal(key, (K,) + base.shape) * 0.05
                batch = base[None] + noise

                # Single-GPU baseline (lax.map only)
                single_fn = jax.jit(batched_energy(sfxn, jpose, state=state, chunk_size=CHUNK_SIZE, score_dtype=jnp.float32))
                jax.block_until_ready(single_fn(batch))  # warmup
                t0 = time.perf_counter()
                for _ in range(N_REPEATS):
                    jax.block_until_ready(single_fn(batch))
                single_ms = (time.perf_counter() - t0) / N_REPEATS * 1000

                # Multi-GPU (pmap)
                multi_fn = pmap_batched_energy(sfxn, jpose, state=state, chunk_size=CHUNK_SIZE, score_dtype=jnp.float32)
                jax.block_until_ready(multi_fn(batch))  # warmup
                t0 = time.perf_counter()
                for _ in range(N_REPEATS):
                    jax.block_until_ready(multi_fn(batch))
                multi_ms = (time.perf_counter() - t0) / N_REPEATS * 1000

                row = {
                    "K": K,
                    "single_gpu_ms": single_ms,
                    "multi_gpu_ms": multi_ms,
                    "single_per_pose_ms": single_ms / K,
                    "multi_per_pose_ms": multi_ms / K,
                    "speedup": single_ms / multi_ms if multi_ms > 0 else 0,
                }
                out["rows"].append(row)
                print(
                    f"K={K:5d}: 1-GPU={single_ms:8.2f}ms ({single_ms/K:6.3f}/pose)  "
                    f"{n_dev}-GPU={multi_ms:8.2f}ms ({multi_ms/K:6.3f}/pose)  "
                    f"speedup={single_ms/multi_ms:.2f}×",
                    flush=True,
                )
            except Exception as exc:
                out["errors"].append(f"K={K}: {exc!r}")
                print(f"K={K} FAILED: {exc!r}", flush=True)
                continue
    except Exception as exc:
        out["errors"].append(f"setup: {exc!r}\n{traceback.format_exc()}")
        print(f"setup FAILED: {exc!r}", flush=True)

    return out


def _format_multi_gpu_md(result: dict) -> str:
    lines = []
    n_dev = result.get("n_devices", "?")
    lines.append(f"## pmap multi-GPU sweep ({result['pdb']}, n_atoms={result.get('n_atoms','?')}, {n_dev} devices)")
    lines.append("")
    lines.append(f"GPUs: {result.get('devices', '?')}.  Forward only.  Precision: float32 (geometry), float64 (accumulator).")
    lines.append("")
    lines.append(f"| K | 1-GPU (ms) | {n_dev}-GPU (ms) | per-pose 1-GPU | per-pose {n_dev}-GPU | speedup |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in result["rows"]:
        lines.append(
            f"| {row['K']} | {row['single_gpu_ms']:.2f} | {row['multi_gpu_ms']:.2f} | "
            f"{row['single_per_pose_ms']:.3f} | {row['multi_per_pose_ms']:.3f} | {row['speedup']:.2f}× |"
        )
    if result.get("errors"):
        lines.append("")
        lines.append("Errors:")
        for e in result["errors"]:
            lines.append(f"- {e}")
    return "\n".join(lines)


@app.local_entrypoint()
def run_bench(pdb: str = "1ubq"):
    result = bench_one_pose.remote(pdb)
    md = _format_markdown(result)
    print()
    print(md)
    repo = pathlib.Path(__file__).resolve().parents[1]
    out_path = repo / "docs" / "BENCHMARKS.md"
    out_path.write_text(md)
    print(f"\nwrote {out_path}")


# PDB size sweep: paths shipped with the upstream tmol test data.
SIZE_SWEEP_PDBS = [
    ("75res_2mtq", "/root/jtmol/tmol/tmol/tests/data/pdb/bysize_075_res_2mtq.pdb"),
    ("100res_5umr", "/root/jtmol/tmol/tmol/tests/data/pdb/bysize_100_res_5umr.pdb"),
    ("150res_5yzf", "/root/jtmol/tmol/tmol/tests/data/pdb/bysize_150_res_5yzf.pdb"),
    ("200res_5mm8", "/root/jtmol/tmol/tmol/tests/data/pdb/bysize_200_res_5mm8.pdb"),
    ("300res_6f8b", "/root/jtmol/tmol/tmol/tests/data/pdb/bysize_300_res_6f8b.pdb"),
    ("400res_6azu", "/root/jtmol/tmol/tmol/tests/data/pdb/bysize_400_res_6azu.pdb"),
    ("500res_5v2d", "/root/jtmol/tmol/tmol/tests/data/pdb/bysize_500_res_5v2d.pdb"),
    ("600res_5m4a", "/root/jtmol/tmol/tmol/tests/data/pdb/bysize_600_res_5m4a.pdb"),
]


def _adaptive_chunk_size(n_atoms: int) -> int:
    """Heuristic: cap dense pair tensor at ~1M elements per chunk slot.

    N=1200 (1ubq) -> 64; N=2400 (200res) -> 16; N=4800 (300res) -> 4;
    N=9600 (600res) -> 1. Keeps peak memory bounded on 40GB at f32.
    """
    target = 8_000_000  # elements per chunk slot
    raw = max(1, target // (n_atoms * n_atoms))
    # snap to power of 2 for nice XLA dispatch
    for c in (64, 32, 16, 8, 4, 2, 1):
        if c <= raw:
            return c
    return 1


@app.function(gpu="A100-40GB", timeout=60 * 90)
def bench_size_sweep(k: int = 128, use_neighbor_list: bool = False) -> dict:
    """Per-pose throughput across protein sizes (75-600 residues).

    Uses upstream tmol's size-graded test fixtures so results are
    reproducible from the same wheel + git clone. chunk_size is
    adapted to N to keep dense pair memory bounded.

    Pass ``use_neighbor_list=True`` to enable the LJLK + elec NL paths;
    that flips ``chunking_mode`` to ``python`` because lax.map + NL
    triggers an XLA autotuner workspace blow-up on GPU.
    """
    import os, sys, time, traceback
    target = "/root/jtmol/tmol"
    if not os.path.islink(target):
        os.symlink("/opt/tmol_src", target)
    if "/root/jtmol" not in sys.path:
        sys.path.insert(0, "/root/jtmol")

    import warnings
    warnings.filterwarnings("ignore")
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch

    out: dict = {"k": k, "rows": [], "errors": [], "device": "unknown", "use_neighbor_list": use_neighbor_list}
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)

    import tmol_jax
    from tmol_jax import beta2016_score_function
    from tmol_jax.score import batched_energy
    chunking_mode = "python" if use_neighbor_list else "lax_map"

    import tmol
    from tmol import pose_stack_from_pdb as up_pose_stack
    from tmol.score.score_function import ScoreFunction as UpScoreFunction
    from tmol.score.score_types import ScoreType as UpST

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for label, pdb_path in SIZE_SWEEP_PDBS:
        if not os.path.isfile(pdb_path):
            out["errors"].append(f"{label}: missing {pdb_path}")
            continue
        try:
            # --------- jtmol side ---------
            jpose = tmol_jax.pose_stack_from_pdb(pdb_path)
            n_atoms = int(jpose.coords.shape[1])
            chunk = _adaptive_chunk_size(n_atoms)
            print(f"[{label}] n_atoms={n_atoms}, chunk_size={chunk}", flush=True)

            sfxn = beta2016_score_function(use_neighbor_list=use_neighbor_list)
            state = sfxn.setup_for_pose_stack(jpose)
            base = jnp.asarray(jpose.coords)

            key = jax.random.PRNGKey(n_atoms)
            noise = jax.random.normal(key, (k,) + base.shape) * 0.05
            batch = base[None] + noise

            from tmol_jax.score import batched_value_and_grad
            bfn_e = batched_energy(sfxn, jpose, state=state, chunk_size=chunk, score_dtype=jnp.float32, chunking_mode=chunking_mode)
            bfn_g = batched_value_and_grad(sfxn, jpose, state=state, chunk_size=chunk, score_dtype=jnp.float32, chunking_mode=chunking_mode)
            if chunking_mode == "lax_map":
                bfn_e = jax.jit(bfn_e)
                bfn_g = jax.jit(bfn_g)

            jax.block_until_ready(bfn_e(batch))
            jax.block_until_ready(bfn_g(batch))

            t0 = time.perf_counter()
            for _ in range(3):
                jax.block_until_ready(bfn_e(batch))
            jt_e_ms = (time.perf_counter() - t0) / 3 * 1000 / k

            t0 = time.perf_counter()
            for _ in range(3):
                jax.block_until_ready(bfn_g(batch))
            jt_g_ms = (time.perf_counter() - t0) / 3 * 1000 / k

            # --------- upstream side ---------
            torch_pose = up_pose_stack(pdb_path, device)
            sfxn_up = UpScoreFunction(tmol.ParameterDatabase.get_default(), device)
            weights = {
                UpST.fa_ljatr: 1.0, UpST.fa_ljrep: 0.55, UpST.fa_lk: 1.0,
                UpST.fa_elec: 1.0, UpST.hbond: 1.0,
                UpST.lk_ball_iso: -0.38, UpST.lk_ball: 0.92,
                UpST.lk_bridge: -0.33, UpST.lk_bridge_uncpl: -0.33,
                UpST.rama: 0.5, UpST.omega: 0.48,
                UpST.dunbrack_rot: 0.76, UpST.dunbrack_rotdev: 0.69, UpST.dunbrack_semirot: 0.78,
                UpST.cart_lengths: 0.5, UpST.cart_angles: 0.5, UpST.cart_torsions: 0.5,
                UpST.cart_impropers: 0.5, UpST.cart_hxltorsions: 0.5,
                UpST.ref: 1.0,
            }
            for st, w in weights.items():
                sfxn_up.set_weight(st, w)
            scorer = sfxn_up.render_whole_pose_scoring_module(torch_pose)

            base_t = torch_pose.coords.detach().clone().requires_grad_(True)
            # warmup
            for _ in range(2):
                _ = scorer(base_t).sum()
            torch.cuda.synchronize()

            # single-pose upstream (loop K times). Cap upstream's K at min(k, 32)
            # for big proteins to stay under the 90 min function timeout.
            up_k = min(k, 16) if n_atoms > 6000 else min(k, 64)
            # Forward measured under no_grad: upstream otherwise builds an
            # autograd tape on every forward call that it never uses, which
            # inflates the forward time vs jtmol's tape-free forward path.
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(3):
                    for _ in range(up_k):
                        _ = scorer(base_t).sum()
                    torch.cuda.synchronize()
            up_e_ms = (time.perf_counter() - t0) / 3 * 1000 / up_k

            t0 = time.perf_counter()
            for _ in range(3):
                for _ in range(up_k):
                    base_t.grad = None
                    e = scorer(base_t).sum()
                    e.backward()
                torch.cuda.synchronize()
            up_g_ms = (time.perf_counter() - t0) / 3 * 1000 / up_k

            row = {
                "label": label,
                "n_atoms": n_atoms,
                "chunk": chunk,
                "jtmol_e_ms": jt_e_ms,
                "jtmol_g_ms": jt_g_ms,
                "upstream_e_ms": up_e_ms,
                "upstream_g_ms": up_g_ms,
                "speedup_e": up_e_ms / jt_e_ms,
                "speedup_g": up_g_ms / jt_g_ms,
            }
            out["rows"].append(row)
            print(
                f"[{label}] N={n_atoms} chunk={chunk}  "
                f"jtmol E={jt_e_ms:7.3f} ms/pose  up E={up_e_ms:7.3f} ms/pose  "
                f"speedup E={up_e_ms/jt_e_ms:5.1f}× G={up_g_ms/jt_g_ms:5.1f}×",
                flush=True,
            )
        except Exception as exc:
            out["errors"].append(f"{label}: {exc!r}")
            print(f"[{label}] FAILED: {exc!r}", flush=True)

    return out


def _format_size_sweep_md(result: dict) -> str:
    lines = []
    lines.append(f"## Size sweep ({result.get('device', '?')}, K={result['k']})")
    lines.append("")
    lines.append("| Label | N atoms | chunk | jtmol E | upstream E | speedup E | jtmol grad | upstream grad | speedup grad |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in result["rows"]:
        lines.append(
            f"| {r['label']} | {r['n_atoms']} | {r['chunk']} | "
            f"{r['jtmol_e_ms']:.3f} | {r['upstream_e_ms']:.3f} | {r['speedup_e']:.1f}× | "
            f"{r['jtmol_g_ms']:.3f} | {r['upstream_g_ms']:.3f} | {r['speedup_g']:.1f}× |"
        )
    if result.get("errors"):
        lines.append("")
        lines.append("Errors:")
        for e in result["errors"]:
            lines.append(f"- {e}")
    return "\n".join(lines)


@app.local_entrypoint()
def run_size_sweep(k: int = 128, nl: bool = False):
    result = bench_size_sweep.remote(k, nl)
    md = _format_size_sweep_md(result)
    print()
    print(md)
    repo = pathlib.Path(__file__).resolve().parents[1]
    out_name = "BENCHMARKS_SIZE_NL.md" if nl else "BENCHMARKS_SIZE.md"
    out_path = repo / "docs" / out_name
    out_path.write_text(md)
    print(f"\nwrote {out_path}")


@app.local_entrypoint()
def run_pmap(pdb: str = "1ubq"):
    result = bench_multi_gpu.remote(pdb)
    md = _format_multi_gpu_md(result)
    print()
    print(md)
    repo = pathlib.Path(__file__).resolve().parents[1]
    out_path = repo / "docs" / "BENCHMARKS_PMAP.md"
    out_path.write_text(md)
    print(f"\nwrote {out_path}")
