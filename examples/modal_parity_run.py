"""Modal: refresh ``docs/UPSTREAM_PARITY.md`` on a GPU box.

The local CPU sweep takes ~1 hour and the markdown in the repo was
generated *before* the cart_angles inter-residue-subgraph fix +
custom-JVP work landed, so several rows are stale. This script:

* Builds an image with Python 3.12, JAX[cuda12], torch 2.8 (CUDA 12), the
  matching pre-built upstream tmol wheel from the GitHub releases page,
  and the upstream tmol *source tree* cloned at the master HEAD (we
  need it for ``tmol/tests/data/pdb/*.pdb`` and the per-term YAML
  goldens — neither is shipped in the wheel).
* Mounts the local jtmol checkout into the container so the harness
  measures the live source, not a published version.
* Runs ``python scripts/run_upstream_parity.py`` once on an A100-40GB,
  reads back the two files it writes, and pastes them into the local
  ``docs/`` directory.

Usage::

    modal run examples/modal_parity_run.py::run_parity

Total wall-clock budget: ~30 min including image build (~5-8 min) and
the sweep itself (5-15 min on A100 once JIT is warm).
"""

from __future__ import annotations

import os
import pathlib

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Pre-built upstream tmol wheel that matches our Python + torch + CUDA
# stack. Pinned to v0.1.22 cu128 + torch 2.10 because that's the
# canonical pair shipped at the time of writing — the older cu124/torch
# 2.8 combination was not available on the PyPI mirror Modal pulls from.
TMOL_WHEEL_URL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.22/"
    "tmol-0.1.22+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
)

# Pin to a specific upstream commit so the test-data layout is stable.
# Master HEAD at session date — adjust if it drifts.
TMOL_SRC_REPO = "https://github.com/uw-ipd/tmol.git"
TMOL_SRC_BRANCH = "master"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    # Core scientific stack. Install torch first with its own index so we
    # get the cu128 build that matches the tmol wheel's ABI.
    .pip_install(
        "torch==2.10.*",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # JAX with CUDA 12 plugin wheels (pulled from PyPI).
    .pip_install(
        "jax[cuda12]>=0.4.30",
    )
    # jtmol runtime deps (mirrors pyproject.toml).
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
    # Upstream tmol wheel (pre-compiled C++/CUDA extensions).
    .pip_install(TMOL_WHEEL_URL)
    # Upstream test data + YAML goldens. The wheel doesn't ship these.
    .run_commands(
        f"git clone --depth 1 --branch {TMOL_SRC_BRANCH} {TMOL_SRC_REPO} /opt/tmol_src",
    )
    .env(
        {
            # Don't gobble all GPU memory; jtmol's allocations are tiny.
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5",
            # Keep JAX from preallocating; helps with debug prints.
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    # Mount the local jtmol checkout. We exclude obvious build/cache
    # cruft so the upload stays small; the harness only needs source,
    # tests, scripts, and docs.
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/jtmol",
        ignore=[
            ".git/**",
            "tmol/**",        # upstream clone (multi-GB, gitignored)
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


@app.function(
    gpu="A100-40GB",
    timeout=2 * 60 * 60,  # 2 hours; harness was slower than expected on first try.
)
def run_sweep() -> dict[str, str]:
    """Execute the parity harness inside the container, return both outputs.

    Returns a dict with keys ``UPSTREAM_PARITY.md``, ``upstream_parity_results.csv``,
    and ``stdout`` (the captured run log, useful for debugging on the
    local side).
    """
    import os
    import shutil
    import subprocess
    import sys

    repo = "/root/jtmol"

    # The harness expects upstream source to be at /root/jtmol/tmol so it
    # can resolve test-data paths like
    # ``tmol/tmol/tests/data/pdb/1r21.pdb``. We cloned upstream into
    # /opt/tmol_src; symlink it into place.
    target = os.path.join(repo, "tmol")
    if os.path.exists(target) or os.path.islink(target):
        # Defensive: shouldn't happen because we excluded tmol/ from the
        # mount, but be idempotent.
        if os.path.islink(target):
            os.unlink(target)
        else:
            shutil.rmtree(target)
    os.symlink("/opt/tmol_src", target)

    # PYTHONPATH includes only the repo root so that ``import tmol_jax``
    # works. Importantly, do NOT add ``/root/jtmol/tmol`` to the path —
    # that source clone would shadow the pip-installed tmol wheel (which
    # is the only one carrying the compiled C++/CUDA extensions). The
    # harness still finds the upstream test data via the filesystem path
    # ``/root/jtmol/tmol/tmol/tests/data/...`` (no Python import needed).
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}"

    # Sanity probes before launching the (long) sweep, so we can fail
    # fast with a useful message if either side is broken.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0,'/root/jtmol'); "
            "import jax; print('jax', jax.__version__, 'devices', jax.devices()); "
            "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); "
            "import tmol; print('tmol OK', tmol.__file__); "
            "import tmol_jax; print('tmol_jax OK', tmol_jax.__file__)",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo,
    )
    probe_log = probe.stdout + "\n--- stderr ---\n" + probe.stderr

    if probe.returncode != 0:
        return {
            "UPSTREAM_PARITY.md": "",
            "upstream_parity_results.csv": "",
            "stdout": "[FATAL] import probe failed:\n" + probe_log,
        }

    # Run the actual sweep. ``check=False`` because the harness returns
    # 1 if any cell fails — we still want the markdown/csv it produces.
    #
    # Stream stdout/stderr live to the container's logs (visible via
    # the Modal dashboard) so we can see progress instead of waiting for
    # the function to complete blindly. The output also gets tee'd to a
    # file we read back at the end.
    log_file = "/tmp/parity_stream.log"
    print("[remote] starting harness; streaming to /tmp/parity_stream.log", flush=True)
    with open(log_file, "w") as fh:
        result = subprocess.run(
            [sys.executable, "-u", "scripts/run_upstream_parity.py"],
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=repo,
        )
    streamed = open(log_file).read() if os.path.exists(log_file) else ""
    run_log = (
        probe_log
        + "\n=== run_upstream_parity.py ===\n"
        + streamed
        + f"\n[exit {result.returncode}]\n"
    )
    # Tail of the streamed log to the container's stdout so we see it
    # in Modal's live logs even when the function eventually returns.
    print("[remote] harness finished, exit", result.returncode, flush=True)
    print(streamed[-4000:] if streamed else "(empty harness output)", flush=True)

    md_path = os.path.join(repo, "docs", "UPSTREAM_PARITY.md")
    csv_path = os.path.join(repo, "docs", "upstream_parity_results.csv")

    md = open(md_path).read() if os.path.exists(md_path) else ""
    csv = open(csv_path).read() if os.path.exists(csv_path) else ""

    return {
        "UPSTREAM_PARITY.md": md,
        "upstream_parity_results.csv": csv,
        "stdout": run_log,
    }


@app.local_entrypoint()
def run_parity() -> None:
    """Trigger the remote sweep, write its outputs into local ``docs/``."""
    print("[local] launching remote sweep on A100-40GB...")
    out = run_sweep.remote()

    md = out["UPSTREAM_PARITY.md"]
    csv = out["upstream_parity_results.csv"]
    log = out["stdout"]

    log_path = REPO_ROOT / "docs" / "upstream_parity_modal.log"
    log_path.write_text(log)
    print(f"[local] wrote {log_path} ({len(log)} bytes)")

    if not md:
        print("[local] WARNING: empty UPSTREAM_PARITY.md returned. "
              "Check the log file above. Local files NOT overwritten.")
        return

    md_path = REPO_ROOT / "docs" / "UPSTREAM_PARITY.md"
    csv_path = REPO_ROOT / "docs" / "upstream_parity_results.csv"
    md_path.write_text(md)
    csv_path.write_text(csv)

    print(f"[local] wrote {md_path} ({len(md)} bytes)")
    print(f"[local] wrote {csv_path} ({len(csv)} bytes)")

    # Print the markdown summary block so the operator sees the
    # headline numbers without opening a file.
    summary_lines = []
    for line in md.splitlines():
        summary_lines.append(line)
        if line.startswith("## Per-term breakdown"):
            break
    print("\n=== UPSTREAM_PARITY.md (top) ===")
    print("\n".join(summary_lines))
