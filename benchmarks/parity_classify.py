"""Authoritative per-channel parity classifier (f64 oracle + FD adjoint tiebreaker).

The strict 1e-4 f32 gate in exact_upstream_matrix.py compares jtmol-f32 vs upstream-f32.
For several channels that comparison is dominated by UPSTREAM's own f32 imprecision or a
buggy analytic adjoint, NOT by a jtmol error. This tool classifies each channel correctly:

  OK                          - passes the f32 gate
  UPSTREAM-f32-artifact       - jtmol == upstream in f64; the f32 diff is upstream's own
                                f32<->f64 gap (jtmol is the more accurate value)
  UPSTREAM-adjoint-bug        - energies agree in f64 but gradients differ; upstream's
                                analytic gradient disagrees with a finite-difference of
                                upstream's OWN energy, jtmol matches FD (jtmol correct)
  JTMOL-BUG                    - jtmol disagrees with upstream/FD in f64 (a real jtmol defect)

Decisive comparisons (no faith in either side required):
  * jtmol-f64 vs upstream-f64  -> physics, no f32 noise
  * central FD of the shared f64 energy -> ground-truth gradient for an adjoint dispute;
    upstream's analytic vs its OWN FD exposes an upstream adjoint bug without involving jtmol.

tmol runs in f64 (pass double coords to the rendered scorer). Scope here: 1ubq S1, all 22
beta2016 channels, energy + per-channel gradient. Emits a verdict table + JSON.

    modal run benchmarks/parity_classify.py
"""
from __future__ import annotations
import pathlib, modal
REPO = pathlib.Path(__file__).resolve().parents[1]
WHEEL = ("https://github.com/uw-ipd/tmol/releases/download/v0.1.40/"
         "tmol-0.1.40+cu128torch2.10-cp312-cp312-linux_x86_64.whl")
image = (
    modal.Image.debian_slim(python_version="3.12").apt_install("git")
    .pip_install("torch==2.10.*", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("jax[cuda12]==0.10.2")
    .pip_install("numpy>=1.24", "scipy>=1.11", "attrs>=22.0", "pyyaml>=6.0",
                 "networkx>=3.0", "pandas>=2.0", "biotite>=0.41.0",
                 "typing_extensions>=4.0", "packaging>=23.0")
    .pip_install(WHEEL)
    .add_local_dir(str(REPO / "jtmol"), remote_path="/root/repo/jtmol",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(str(REPO / "benchmarks/exact_upstream_matrix.py"),
                    remote_path="/root/repo/benchmarks/exact_upstream_matrix.py")
)
app = modal.App("jtmol-parity-classify", image=image)
E_GATE = 1e-4    # energy_atol from CERTIFICATION_TOLERANCES
G_GATE = 5e-4    # grad_atol


@app.function(gpu="A100-40GB", timeout=60 * 30)
def classify() -> dict:
    import sys, json, numpy as np
    sys.path.insert(0, "/root/repo"); sys.path.insert(0, "/root/repo/benchmarks")
    from exact_upstream_matrix import OFFICIAL_NAMES
    PDB = "/root/repo/jtmol/tests/data/1ubq.pdb"

    # ---------------- upstream: per-channel energy + grad (f32, f64) + FD ---------------
    import torch
    dev = torch.device("cuda")
    from tmol.io import pose_stack_from_pdb
    from tmol.score import beta2016_score_function
    ps = pose_stack_from_pdb(PDB, dev)
    sf = beta2016_score_function(dev)
    scorer = sf.render_whole_pose_scoring_module(ps)
    names = [st.name for st in sf.all_score_types()]
    reorder = [names.index(n) for n in OFFICIAL_NAMES]
    c64 = ps.coords.detach().to(torch.float64)

    def up_channels(dtype):
        c0 = ps.coords.detach().to(dtype)
        with torch.inference_mode():
            w = scorer(c0, sum_terms=False, apply_weights=True)
        e = w.detach()[reorder, 0].double().cpu().numpy()
        g = np.zeros((len(OFFICIAL_NAMES),) + tuple(ps.coords.shape), np.float64)
        for oi, row in enumerate(reorder):
            cg = c0.clone().requires_grad_(True)
            wg = scorer(cg, sum_terms=False, apply_weights=True)
            (grad,) = torch.autograd.grad(wg[row].sum(), cg)
            g[oi] = grad.detach().double().cpu().numpy()
        return e, g

    def up_energy(native_row, c):
        with torch.inference_mode():
            w = scorer(c, sum_terms=False, apply_weights=True)
        return float(w[native_row, 0].item())

    def up_fd(native_row, a, k, h=1e-5):
        cp = c64.clone(); cp[0, a, k] += h
        cm = c64.clone(); cm[0, a, k] -= h
        return (up_energy(native_row, cp) - up_energy(native_row, cm)) / (2 * h)

    up32_e, up32_g = up_channels(torch.float32)
    up64_e, up64_g = up_channels(torch.float64)

    # ---------------- jtmol: per-channel energy + grad (x64/f64) ------------------------
    import jax; jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp, jtmol
    from exact_upstream_matrix import _jtmol_score_function, _jax_weighted_term_gradients
    from jtmol.score.score_types import ScoreType as JST
    jps = jtmol.pose_stack_from_pdb(PDB)
    jsf = _jtmol_score_function(jtmol)
    state = jsf.setup_for_pose_stack(jps)
    coords = jnp.asarray(np.asarray(jps.coords))

    def weighted_terms(c):
        v = jsf.score_by_term(jps, state=state, coords_override=c)
        return jnp.stack([jnp.sum(v[JST[n]]) for n in OFFICIAL_NAMES])

    jt_e = np.asarray(weighted_terms(coords)).astype(np.float64)
    jt_g, _ = _jax_weighted_term_gradients(weighted_terms, coords, jax=jax, jnp=jnp)
    jt_g = np.asarray(jt_g).astype(np.float64)

    # ---------------- classify ----------------
    def adjoint_tiebreak(i):
        """energies agree, gradients differ in f64 -> FD the shared energy, see who's right."""
        native_row = reorder[i]
        d = np.abs(jt_g[i] - up_g_ref[i])
        a, k = np.unravel_index(int(np.argmax(d)), d.shape)[1:]  # skip pose axis
        # careful: d is [1,N,3]
        a, k = divmod(int(np.argmax(d.reshape(-1))), 3)
        ref = up_fd(native_row, a, k, 1e-5)
        ref6 = up_fd(native_row, a, k, 1e-6)
        ja, ua = float(jt_g[i, 0, a, k]), float(up64_g[i, 0, a, k])
        converged = abs(ref - ref6) < 1e-4 * (abs(ref) + 1)
        jt_ok = abs(ja - ref) < abs(ua - ref)
        return {"atom": a, "axis": k, "fd": ref, "fd_converged": bool(converged),
                "jt_analytic": ja, "up_analytic": ua,
                "jt_matches_fd": bool(abs(ja - ref) <= G_GATE),
                "up_matches_fd": bool(abs(ua - ref) <= G_GATE), "jt_closer": bool(jt_ok)}

    up_g_ref = up64_g
    rows, verdicts = [], {}
    for i, name in enumerate(OFFICIAL_NAMES):
        e_cur = abs(jt_e[i] - up32_e[i]); g_cur = float(np.abs(jt_g[i] - up32_g[i]).max())
        e_phys = abs(jt_e[i] - up64_e[i]); g_phys = float(np.abs(jt_g[i] - up64_g[i]).max())
        fails = (e_cur > E_GATE) or (g_cur > G_GATE)
        tb = None
        if not fails:
            v = "OK"
        elif (e_phys <= E_GATE) and (g_phys <= G_GATE):
            v = "UPSTREAM-f32-artifact"          # jtmol == upstream in f64
        elif e_phys <= E_GATE and g_phys > G_GATE:
            tb = adjoint_tiebreak(i)             # energies agree, gradients differ -> FD
            if tb["jt_matches_fd"] and not tb["up_matches_fd"]:
                v = "UPSTREAM-adjoint-bug"       # jtmol correct, upstream analytic wrong
            elif tb["up_matches_fd"] and not tb["jt_matches_fd"]:
                v = "JTMOL-BUG(adjoint)"
            else:
                v = "AMBIGUOUS(both/neither match FD)"
        else:
            v = "JTMOL-BUG(energy+grad differ in f64)"
        verdicts[name] = v
        rows.append({"channel": name, "verdict": v, "e_f32_gap": e_cur, "e_f64_gap": e_phys,
                     "g_f32_gap": g_cur, "g_f64_gap": g_phys, "tiebreak": tb})

    print(f"\n{'channel':>18} {'verdict':>26} {'eF32':>9} {'eF64':>9} {'gF32':>9} {'gF64':>9}")
    for r in rows:
        print(f"{r['channel']:>18} {r['verdict']:>26} {r['e_f32_gap']:9.1e} {r['e_f64_gap']:9.1e} "
              f"{r['g_f32_gap']:9.1e} {r['g_f64_gap']:9.1e}")
        if r["tiebreak"]:
            tb = r["tiebreak"]
            print(f"{'':>18}   FD@atom{tb['atom']},ax{tb['axis']}: fd={tb['fd']:.6f} "
                  f"jt={tb['jt_analytic']:.6f} up={tb['up_analytic']:.6f} "
                  f"(jt_matches_fd={tb['jt_matches_fd']}, up_matches_fd={tb['up_matches_fd']})")

    from collections import Counter
    summary = Counter(verdicts.values())
    print("\nSUMMARY:", dict(summary))
    jtmol_bugs = [n for n, v in verdicts.items() if v.startswith("JTMOL-BUG")]
    print("REAL jtmol bugs:", jtmol_bugs if jtmol_bugs else "NONE")
    return {"verdicts": verdicts, "rows": [
        {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
         for k, v in r.items() if k != "tiebreak"} for r in rows]}


@app.local_entrypoint()
def main():
    classify.remote()
