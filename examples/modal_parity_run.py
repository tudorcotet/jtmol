"""Modal: refresh ``docs/UPSTREAM_PARITY.md`` on a GPU box.

This version fans out one Modal function per PDB so the wall-clock is
bounded by the slowest single PDB (~15-30 min on A100) instead of the
sum across all PDBs (which kept tripping the 1h/2h container timeout
on the monolithic harness).

Architecture:

* The image builds once (torch 2.10 + cu128 + matching tmol wheel +
  JAX[cuda12] + the upstream tmol source tree at /opt/tmol_src so the
  test-data paths resolve).
* ``run_one_pdb(pdb_key)`` evaluates every score channel for a single
  PDB (forward + gradient on both jtmol and live upstream torch tmol)
  and returns a list of structured rows.
* ``run_parity`` (the local entrypoint) ``.map``s ``run_one_pdb`` over
  the PDB list, gathers the rows, and writes the unified markdown +
  CSV into local ``docs/``.

Usage::

    modal run examples/modal_parity_run.py::run_parity

Each per-PDB function has a 1-hour budget. If one fails, the others
still complete and we ship the partial report.
"""

from __future__ import annotations

import pathlib

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Pre-built upstream tmol wheel that matches our Python + torch + CUDA
# stack. Pinned to v0.1.22 cu128 + torch 2.10 because that's the
# canonical pair shipped at the time of writing.
TMOL_WHEEL_URL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.22/"
    "tmol-0.1.22+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
)

TMOL_SRC_REPO = "https://github.com/uw-ipd/tmol.git"
TMOL_SRC_BRANCH = "master"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install(
        "torch==2.10.*",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "jax[cuda12]>=0.4.30",
    )
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.11",
        "attrs>=22.0",
        "pyyaml>=6.0",
        "networkx>=3.0",
        "pandas>=2.0",
        "biotite>=0.41.0",
        "typing_extensions>=4.0",
    )
    .pip_install(TMOL_WHEEL_URL)
    .run_commands(
        f"git clone --depth 1 --branch {TMOL_SRC_BRANCH} {TMOL_SRC_REPO} /opt/tmol_src",
    )
    .env(
        {
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/jtmol",
        ignore=[
            ".git/**",
            "tmol/**",
            "scratch_pdbs/**",
            "**/__pycache__/**",
            "*.egg-info/**",
            "build/**",
            "dist/**",
            ".venv/**",
            "venv/**",
            ".pytest_cache/**",
            ".ruff_cache/**",
        ],
    )
)

app = modal.App("tmol-jax-upstream-parity", image=image)


# Score types and their upstream term mapping (forward + gradient).
ALL_SCORE_TYPE_NAMES: list[str] = [
    "fa_ljatr", "fa_ljrep", "fa_lk",
    "fa_elec",
    "hbond",
    "rama", "omega",
    "dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot",
    "cart_lengths", "cart_angles", "cart_torsions",
    "cart_impropers", "cart_hxltorsions",
    "lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl",
    "ref",
]


# (label, remote-absolute-path) — both paths exist inside the container
# only. Local entrypoint passes the label, the remote function looks
# the path up.
PDB_PATHS_REMOTE: dict[str, str] = {
    "1ubq": "/root/jtmol/tmol_jax/tests/data/1ubq.pdb",
    "1qys": "/root/jtmol/tmol_jax/tests/data/1qys.pdb",
    # Upstream's master branch ships these as uppercase filenames; our
    # local mirror has lowercase symlinks but Modal pulls master fresh.
    "1r21": "/root/jtmol/tmol/tmol/tests/data/pdb/1R21.pdb",
    "3plc": "/root/jtmol/tmol/tmol/tests/data/pdb/3plc.pdb",
    "6DMZ_A": "/root/jtmol/tmol/tmol/tests/data/pdb/6DMZ_A.pdb",
}

PDB_ORDER: list[str] = ["1ubq", "1qys", "1r21", "3plc", "6DMZ_A"]

ATOL = 1e-5
RTOL = 1e-3


def _ensure_tmol_symlink() -> None:
    """Place upstream tmol source at ``/root/jtmol/tmol`` for test-data paths.

    The wheel ships compiled extensions but not the PDBs / YAML
    goldens; the harness expects them at filesystem paths under
    ``tmol/tmol/tests/data/``. We cloned upstream to ``/opt/tmol_src``
    in the image build, symlink it into place once per container.
    """
    import os
    import shutil

    target = "/root/jtmol/tmol"
    if os.path.islink(target):
        return
    if os.path.exists(target):
        shutil.rmtree(target)
    os.symlink("/opt/tmol_src", target)


@app.function(
    gpu="A100-40GB",
    timeout=60 * 60,  # 1 hour per PDB — fan-out keeps total wall-clock <45 min.
)
def run_one_pdb(pdb_label: str) -> dict:
    """Evaluate every score channel for one PDB and return raw rows.

    Returns a dict::

        {
            "pdb_label": "1ubq",
            "rows": [{"st": ..., "fwd_ok": bool, ...}, ...],
            "stdout": "captured stream log",
            "error": "" or "exc text",
        }
    """
    import io
    import os
    import sys
    import time
    import traceback

    # Set up paths.
    _ensure_tmol_symlink()
    repo = "/root/jtmol"
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # Critically, do NOT add /root/jtmol/tmol to sys.path — that would
    # shadow the pip-installed tmol wheel (the only one with compiled
    # C++/CUDA extensions). Filesystem access still works for the
    # YAML goldens / PDBs.

    # Capture stdout to a string so we can return it for visibility.
    log_buf = io.StringIO()

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, s):
            for st in self._streams:
                st.write(s)
                st.flush()

        def flush(self):
            for st in self._streams:
                st.flush()

    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, log_buf)

    pdb_path = PDB_PATHS_REMOTE.get(pdb_label)
    rows: list[dict] = []
    error_text = ""

    try:
        print(f"[{pdb_label}] starting", flush=True)

        if pdb_path is None or not os.path.isfile(pdb_path):
            raise FileNotFoundError(f"PDB not found in container: {pdb_path}")

        # Imports inside the try so any per-PDB-container failure is
        # captured rather than crashing the whole map.
        import warnings
        warnings.filterwarnings("ignore")

        import numpy as np
        import jax
        # Float64 throughout: cubic-Hermite kinks in cart_torsions /
        # omega / cart_hxltorsions accumulate ~1e-2 relative error in
        # float32, which masks the analytical-adjoint parity we get
        # in float64 (verified ~1e-9 vs upstream's Blondel-Karplus
        # adjoint on every kink term).
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import torch

        print(
            f"[{pdb_label}] jax={jax.__version__} devices={jax.devices()} "
            f"torch={torch.__version__} cuda={torch.cuda.is_available()}",
            flush=True,
        )

        import tmol
        import tmol_jax

        print(
            f"[{pdb_label}] tmol={tmol.__file__} tmol_jax={tmol_jax.__file__}",
            flush=True,
        )

        from tmol import ParameterDatabase, pose_stack_from_pdb as up_pose_stack
        from tmol.score.score_function import ScoreFunction as UpScoreFunction
        from tmol.score.score_types import ScoreType as UpScoreType

        from tmol_jax.score.score_types import ScoreType as JaxScoreType
        from tmol_jax import beta2016_score_function

        # Load both poses once per PDB.
        t0 = time.monotonic()
        jax_pose = tmol_jax.pose_stack_from_pdb(pdb_path)
        n_atoms_jax = int(jax_pose.coords.shape[1])
        print(
            f"[{pdb_label}] jax pose n_atoms={n_atoms_jax} "
            f"({time.monotonic() - t0:.1f}s)",
            flush=True,
        )

        t0 = time.monotonic()
        torch_pose = up_pose_stack(pdb_path, torch.device("cpu"))
        n_atoms_torch = int(torch_pose.coords.shape[1])
        print(
            f"[{pdb_label}] torch pose n_atoms={n_atoms_torch} "
            f"({time.monotonic() - t0:.1f}s)",
            flush=True,
        )

        def _within_tol(a: float, b: float) -> bool:
            return float(abs(a - b)) <= ATOL + RTOL * float(abs(b))

        def _rel_err(jax_val: float, ref_val: float) -> float:
            denom = max(abs(ref_val), 1e-12)
            return abs(jax_val - ref_val) / denom

        def _jax_energy_and_grad(name):
            sfxn = beta2016_score_function()
            target = getattr(JaxScoreType, name)
            for st in list(JaxScoreType):
                if st is JaxScoreType.n_score_types:
                    continue
                sfxn.set_weight(st, 0.0)
            sfxn.set_weight(target, 1.0)
            state = sfxn.setup_for_pose_stack(jax_pose)

            def _e(coords):
                per = sfxn.score_by_term(jax_pose, state=state, coords_override=coords)
                if target in per:
                    return per[target].sum()
                return jnp.asarray(0.0, dtype=coords.dtype)

            e = float(_e(jax_pose.coords))
            g = np.asarray(jax.grad(_e)(jax_pose.coords))
            assert g.ndim == 3 and g.shape[0] == 1
            return e, g[0]

        # Upstream's ``set_weight(st, 0.0)`` calls a broken
        # ``remove_term_for_score_type`` (#TODO in their source), so
        # we can never set any weight to 0. The CartBondedEnergyTerm
        # has 5 sub-channels and ``render_whole_pose_scoring_module``
        # returns one entry per registered sub-channel; setting only
        # ``cart_lengths`` to 1.0 fails to materialize cart_lengths
        # (returns 0) because the term's 2/3-body subgraph enumeration
        # gates on the full sub-channel set. Fix: register all 5 at
        # 1.0 (never 0.0) and pick the target's entry from the
        # 5-element output vector. The cart_bonded sub-channels are
        # emitted in the order declared by
        # ``tmol/score/cartbonded/cartbonded_energy_term.py``.
        CART_BONDED_ORDER = [
            "cart_lengths", "cart_angles", "cart_torsions",
            "cart_impropers", "cart_hxltorsions",
        ]
        CART_BONDED_SUBSET = set(CART_BONDED_ORDER)

        def _torch_energy_and_grad(name):
            sfxn = UpScoreFunction(ParameterDatabase.get_default(), torch.device("cpu"))
            target = getattr(UpScoreType, name)
            if name in CART_BONDED_SUBSET:
                for sub_name in CART_BONDED_ORDER:
                    sub_st = getattr(UpScoreType, sub_name)
                    sfxn.set_weight(sub_st, 1.0)
                target_idx = CART_BONDED_ORDER.index(name)
            else:
                sfxn.set_weight(target, 1.0)
                target_idx = None
            scorer = sfxn.render_whole_pose_scoring_module(torch_pose)
            # float64 throughout so the cubic-Hermite kink derivative
            # (cart_torsions, omega, cart_hxltorsions) reaches its
            # true Blondel-Karplus analytical-adjoint precision
            # instead of being floor-limited by float32.
            coords = torch_pose.coords.detach().clone().double().requires_grad_(True)
            weighted = scorer(coords)
            if target_idx is not None:
                # Pick the target's per-channel value from the
                # 5-element cart_bonded vector.
                scalar = weighted.reshape(-1)[target_idx]
            else:
                scalar = weighted.sum()
            scalar.backward()
            e = float(scalar.detach().cpu().float().item())
            if coords.grad is None:
                g = np.zeros(tuple(coords.shape), dtype=np.float64)
            else:
                g = coords.grad.detach().cpu().double().numpy()
            assert g.ndim == 3 and g.shape[0] == 1
            return e, g[0]

        for name in ALL_SCORE_TYPE_NAMES:
            row_t0 = time.monotonic()
            try:
                e_jax, g_jax = _jax_energy_and_grad(name)
            except Exception as exc:
                rows.append(dict(
                    pdb_label=pdb_label, st=name, mode="skip",
                    fwd_ok=None, grad_ok=None,
                    e_jax=None, e_ref=None,
                    grad_max_abs_err=None, rel_err=None,
                    duration_s=time.monotonic() - row_t0,
                    note=f"jax raised: {type(exc).__name__}: {exc}",
                ))
                print(f"[{pdb_label}] {name:20s} JAX-FAIL: {exc!r}", flush=True)
                continue

            try:
                e_torch, g_torch = _torch_energy_and_grad(name)
            except Exception as exc:
                rows.append(dict(
                    pdb_label=pdb_label, st=name, mode="live",
                    fwd_ok=False, grad_ok=False,
                    e_jax=e_jax, e_ref=None,
                    grad_max_abs_err=None, rel_err=None,
                    duration_s=time.monotonic() - row_t0,
                    note=f"torch raised: {type(exc).__name__}: {exc}",
                ))
                print(f"[{pdb_label}] {name:20s} TORCH-FAIL: {exc!r}", flush=True)
                continue

            fwd_ok = _within_tol(e_jax, e_torch)
            rel = _rel_err(e_jax, e_torch)

            if g_jax.shape != g_torch.shape:
                rows.append(dict(
                    pdb_label=pdb_label, st=name, mode="live",
                    fwd_ok=fwd_ok, grad_ok=None,
                    e_jax=e_jax, e_ref=e_torch,
                    grad_max_abs_err=None, rel_err=rel,
                    duration_s=time.monotonic() - row_t0,
                    note=f"grad shape mismatch: jax={g_jax.shape}, torch={g_torch.shape}",
                ))
                print(
                    f"[{pdb_label}] {name:20s} fwd={'P' if fwd_ok else 'F'} "
                    f"grad=- shape_mismatch (jax={g_jax.shape}, torch={g_torch.shape}) "
                    f"({time.monotonic() - row_t0:.1f}s)",
                    flush=True,
                )
                continue

            grad_diff = np.abs(g_jax - g_torch) - (ATOL + RTOL * np.abs(g_torch))
            grad_ok = bool(np.all(grad_diff <= 0.0))
            max_abs = float(np.max(np.abs(g_jax - g_torch)))

            rows.append(dict(
                pdb_label=pdb_label, st=name, mode="live",
                fwd_ok=fwd_ok, grad_ok=grad_ok,
                e_jax=e_jax, e_ref=e_torch,
                grad_max_abs_err=max_abs, rel_err=rel,
                duration_s=time.monotonic() - row_t0,
                note="",
            ))
            print(
                f"[{pdb_label}] {name:20s} fwd={'P' if fwd_ok else 'F'} "
                f"grad={'P' if grad_ok else 'F'} de={e_jax - e_torch:+.3e} "
                f"max|dg|={max_abs:.3e} ({time.monotonic() - row_t0:.1f}s)",
                flush=True,
            )

        print(f"[{pdb_label}] done ({len(rows)} rows)", flush=True)

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(f"[{pdb_label}] FATAL: {error_text}", flush=True)
    finally:
        sys.stdout = real_stdout

    return {
        "pdb_label": pdb_label,
        "rows": rows,
        "stdout": log_buf.getvalue(),
        "error": error_text,
    }


# --------------------------------------------------------------------- #
# Local-side rendering (mirrors scripts/run_upstream_parity.py)         #
# --------------------------------------------------------------------- #


GLYPH_BOTH = "PASS"
GLYPH_NA = "-"


def _cell(row: dict) -> str:
    if row is None:
        return GLYPH_NA
    if row["mode"] == "skip" or row["fwd_ok"] is None:
        return GLYPH_NA
    if row["fwd_ok"]:
        if row["grad_ok"] is True:
            return GLYPH_BOTH
        if row["grad_ok"] is False:
            rel = (row.get("rel_err") or 0.0)
            return f"F:{rel * 100:.2g}%"
        return "fwd-only"
    rel = row.get("rel_err") or 0.0
    return f"FAIL:{rel * 100:.2g}%"


def _pivot_table(rows: list[dict]) -> str:
    by_key = {(r["pdb_label"], r["st"]): r for r in rows}
    header = ["PDB"] + ALL_SCORE_TYPE_NAMES
    sep = ["---"] + [":---:"] * len(ALL_SCORE_TYPE_NAMES)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(sep) + "|"]
    for label in PDB_ORDER:
        cells = [label]
        for st in ALL_SCORE_TYPE_NAMES:
            r = by_key.get((label, st))
            cells.append(_cell(r))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _per_term_breakdown(rows: list[dict]) -> str:
    by_term: dict[str, list[dict]] = {n: [] for n in ALL_SCORE_TYPE_NAMES}
    for r in rows:
        if r["st"] in by_term:
            by_term[r["st"]].append(r)
    lines = ["| Term | Fwd pass | Grad pass | Worst max\\|dg\\| | Notes |",
             "|---|:---:|:---:|---:|---|"]
    for name in ALL_SCORE_TYPE_NAMES:
        term_rows = by_term[name]
        fwd_p = sum(1 for r in term_rows if r["fwd_ok"] is True)
        fwd_total = sum(1 for r in term_rows if r["fwd_ok"] is not None)
        grad_p = sum(1 for r in term_rows if r["grad_ok"] is True)
        grad_total = sum(1 for r in term_rows if r["grad_ok"] is not None)
        worst_dg = 0.0
        for r in term_rows:
            if r.get("grad_max_abs_err") is not None:
                worst_dg = max(worst_dg, r["grad_max_abs_err"])
        worst_rel = 0.0
        for r in term_rows:
            if r["fwd_ok"] is False and r.get("rel_err") is not None:
                worst_rel = max(worst_rel, r["rel_err"])
        note = ""
        if worst_rel > 0.0:
            note = f"max forward rel-err on a failing PDB: {worst_rel * 100:.2g}%"
        lines.append(
            f"| {name} | {fwd_p}/{fwd_total} | {grad_p}/{grad_total} | "
            f"{worst_dg:.2e} | {note} |"
        )
    return "\n".join(lines) + "\n"


def _per_pdb_detail(rows: list[dict], label: str) -> str:
    out = [
        "| Term | Mode | Fwd | Grad | E(jax) | E(ref) | rel | max\\|dg\\| | Note |\n",
        "|---|---|:---:|:---:|---:|---:|---:|---:|---|\n",
    ]

    def _m(v):
        if v is None:
            return "-"
        return "PASS" if v else "FAIL"

    for r in rows:
        if r["pdb_label"] != label:
            continue
        ej = f"{r['e_jax']:+.4f}" if r["e_jax"] is not None else ""
        er = f"{r['e_ref']:+.4f}" if r["e_ref"] is not None else ""
        rel = f"{r['rel_err'] * 100:.2g}%" if r.get("rel_err") is not None else ""
        gm = f"{r['grad_max_abs_err']:.2e}" if r.get("grad_max_abs_err") is not None else ""
        out.append(
            f"| {r['st']} | {r['mode']} | {_m(r['fwd_ok'])} | {_m(r['grad_ok'])} "
            f"| {ej} | {er} | {rel} | {gm} | {r.get('note', '')} |\n"
        )
    return "".join(out)


def _render_markdown(rows: list[dict], failed_pdbs: list[tuple[str, str]]) -> str:
    out: list[str] = []
    out.append("# Upstream parity (tmol_jax vs tmol)\n\n")
    out.append("- Upstream live: **yes** (torch tmol-0.1.22+cu128torch2.10 on A100-40GB)\n")
    out.append(f"- Tolerances: atol={ATOL}, rtol={RTOL}\n")
    out.append(
        "- Comparison rule: `|E_jax - E_ref| <= atol + rtol * |E_ref|` "
        "(matches torch.autograd.gradcheck default)\n"
    )
    out.append(
        f"- Panel size: {len(PDB_ORDER)} PDBs x {len(ALL_SCORE_TYPE_NAMES)} score types "
        "(fan-out: one Modal A100 container per PDB)\n"
    )
    out.append("- PDB selection rationale:\n")
    out.append("  - 1ubq, 1qys: small monomeric baselines\n")
    out.append("  - 1r21, 3plc: disulfide-containing single chains\n")
    out.append("  - 6DMZ_A: designed small protein\n")
    out.append(
        "- Deferred: 1BL8 (upstream refuses to load: missing non-leaf atom CG "
        "on residue 4 ARG).\n"
    )
    if failed_pdbs:
        out.append("- Per-PDB container failures (partial results):\n")
        for tag, reason in failed_pdbs:
            out.append(f"  - {tag}: `{reason.splitlines()[0]}`\n")
    out.append("\n")

    total = len(rows)
    fwd_pass = sum(1 for r in rows if r["fwd_ok"] is True)
    fwd_fail = sum(1 for r in rows if r["fwd_ok"] is False)
    fwd_skip = sum(1 for r in rows if r["fwd_ok"] is None)
    grad_pass = sum(1 for r in rows if r["grad_ok"] is True)
    grad_fail = sum(1 for r in rows if r["grad_ok"] is False)
    grad_skip = sum(1 for r in rows if r["grad_ok"] is None)
    fwd_eligible = fwd_pass + fwd_fail
    grad_eligible = grad_pass + grad_fail

    out.append("## Summary\n\n")
    out.append(f"- Total cells: **{total}**\n")
    if fwd_eligible:
        out.append(
            f"- Forward parity: **{fwd_pass}/{fwd_eligible}** "
            f"({100.0 * fwd_pass / fwd_eligible:.1f}%) PASS, "
            f"{fwd_fail} FAIL, {fwd_skip} SKIP\n"
        )
    else:
        out.append("- Forward parity: 0/0 (all skipped)\n")
    if grad_eligible:
        out.append(
            f"- Gradient parity: **{grad_pass}/{grad_eligible}** "
            f"({100.0 * grad_pass / grad_eligible:.1f}%) PASS, "
            f"{grad_fail} FAIL, {grad_skip} SKIP\n"
        )
    else:
        out.append("- Gradient parity: 0/0 (all skipped)\n")
    out.append("\n")

    out.append("## Per-term breakdown\n\n")
    out.append(_per_term_breakdown(rows))
    out.append("\n")

    out.append("## Parity matrix\n\n")
    out.append(
        "Cells: ``PASS`` = both forward and gradient within tolerance; "
        "``F:<x%>`` = forward passes but gradient fails (with forward rel-err); "
        "``FAIL:<x%>`` = forward fails; "
        "``-`` = skipped (no reference value).\n\n"
    )
    out.append(_pivot_table(rows))
    out.append("\n")

    out.append("## Per-PDB detail tables\n\n")
    for label in PDB_ORDER:
        out.append(f"### {label}\n\n")
        out.append(_per_pdb_detail(rows, label))
        out.append("\n")

    return "".join(out)


def _write_csv(rows: list[dict], out_path: pathlib.Path) -> None:
    import csv

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pdb", "score_type", "mode",
            "forward_ok", "gradient_ok",
            "energy_jax", "energy_ref",
            "forward_abs_err", "forward_rel_err",
            "gradient_max_abs_err",
            "duration_s", "note",
        ])
        for r in rows:
            ej, er = r.get("e_jax"), r.get("e_ref")
            abs_err = abs(ej - er) if (ej is not None and er is not None) else None
            w.writerow([
                r["pdb_label"],
                r["st"],
                r["mode"],
                "" if r["fwd_ok"] is None else int(r["fwd_ok"]),
                "" if r["grad_ok"] is None else int(r["grad_ok"]),
                "" if ej is None else f"{ej:.8f}",
                "" if er is None else f"{er:.8f}",
                "" if abs_err is None else f"{abs_err:.8e}",
                "" if r.get("rel_err") is None else f"{r['rel_err']:.8e}",
                "" if r.get("grad_max_abs_err") is None else f"{r['grad_max_abs_err']:.8e}",
                f"{r.get('duration_s', 0.0):.3f}",
                r.get("note", ""),
            ])


@app.local_entrypoint()
def run_parity() -> None:
    """Fan out one container per PDB; gather results; write docs/."""
    import time

    print(f"[local] launching {len(PDB_ORDER)} parallel A100 containers: {PDB_ORDER}")
    t0 = time.monotonic()

    all_rows: list[dict] = []
    failed_pdbs: list[tuple[str, str]] = []
    log_blocks: list[str] = []

    # .map preserves order and runs in parallel; each container has its
    # own 1h timeout. If one raises, we still get the others.
    try:
        results = list(run_one_pdb.map(PDB_ORDER, return_exceptions=True))
    except Exception as exc:
        print(f"[local] FATAL: .map crashed: {exc!r}")
        raise

    for label, res in zip(PDB_ORDER, results):
        if isinstance(res, Exception):
            failed_pdbs.append((label, repr(res)))
            log_blocks.append(f"=== {label} (container raised) ===\n{res!r}\n")
            print(f"[local] {label}: CONTAINER FAILED: {res!r}")
            continue
        rows = res.get("rows", [])
        all_rows.extend(rows)
        if res.get("error"):
            failed_pdbs.append((label, res["error"]))
        log_blocks.append(f"=== {label} ===\n{res.get('stdout', '')}\n")
        n_fwd_pass = sum(1 for r in rows if r["fwd_ok"] is True)
        n_grad_pass = sum(1 for r in rows if r["grad_ok"] is True)
        print(f"[local] {label}: {len(rows)} rows, fwd_pass={n_fwd_pass}, grad_pass={n_grad_pass}")

    elapsed = time.monotonic() - t0
    print(f"[local] all containers done in {elapsed:.1f}s wall-clock")

    docs = REPO_ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    log_path = docs / "upstream_parity_modal.log"
    log_path.write_text("\n".join(log_blocks))
    print(f"[local] wrote {log_path} ({len(log_path.read_text())} bytes)")

    if not all_rows:
        print("[local] WARNING: no rows collected; not overwriting docs/.")
        return

    md_text = _render_markdown(all_rows, failed_pdbs)
    md_path = docs / "UPSTREAM_PARITY.md"
    md_path.write_text(md_text)
    print(f"[local] wrote {md_path} ({len(md_text)} bytes)")

    csv_path = docs / "upstream_parity_results.csv"
    _write_csv(all_rows, csv_path)
    print(f"[local] wrote {csv_path}")

    # Print summary block.
    summary_lines: list[str] = []
    for line in md_text.splitlines():
        summary_lines.append(line)
        if line.startswith("## Parity matrix"):
            break
    print("\n=== UPSTREAM_PARITY.md (top) ===")
    print("\n".join(summary_lines))
