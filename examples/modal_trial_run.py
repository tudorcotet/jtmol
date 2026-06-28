"""Modal deployment for the real Boltz-2 + tmol_jax binder-design trial.

Runs 30 designs (1 target × 3 weight schemes × 10 each) on A100 40GB,
total ~30 GPU-hours ≈ $37.

Usage:
    modal run examples/modal_trial_run.py::run_trial

Before running:
    pip install modal
    modal setup

This is designed to drop straight into the same pipeline as
https://gist.github.com/nboyd/7b30c4033f834fc532a106d902fbdffa —
we just add our tmol_jax interface loss to the existing sp_loss.
"""

from __future__ import annotations

import modal


def install_tmol_jax():
    """Install tmol_jax into the Modal image (from this repo)."""
    # Copy the tmol_jax source into the image during build.
    # In practice you'd publish tmol_jax as a proper package; for the
    # trial we add it via modal.Mount.
    pass


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # Base mosaic + Boltz-2 deps (from the gist)
    .run_commands("git clone https://github.com/escalante-bio/mosaic.git")
    .workdir("mosaic")
    .run_commands("uv pip install --system -r pyproject.toml")
    .run_commands("uv pip install --system jax[cuda12]")
    .run_commands("uv pip install --system .")
    # tmol_jax deps
    .pip_install(
        "attrs>=22.0",
        "biotite>=0.41.0",
        "networkx>=3.0",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "typing_extensions>=4.0",
    )
    .run_function(
        lambda: __import__("mosaic.models.boltz2", fromlist=["Boltz2"]).Boltz2()
    )  # Pre-download Boltz-2 weights into image
    .run_commands("uv pip install --system equinox")
    .env({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"})
)

# Mount the tmol_jax source tree so the trial uses our port.
# Adjust local_path to point to wherever you clone this repo.
tmol_jax_mount = modal.Mount.from_local_dir(
    local_path=".",  # run modal from the repo root
    remote_path="/root/tmol_jax_src",
    condition=lambda p: "tmol_jax/" in p or p.endswith(".py") or p.endswith(".yaml"),
)

app = modal.App(
    "tmol-jax-binder-trial",
    image=image,
    mounts=[tmol_jax_mount],
)

# ---------------------------------------------------------------- #
# Trial parameters                                                 #
# ---------------------------------------------------------------- #

# Use the nipah target from the gist as default. For the paper we'd
# sweep multiple targets; for the trial keep it to one.
BINDER_LENGTH = 120  # shorter than the gist's 220 to keep trial cheap
TARGET_SEQUENCE = (
    "QNYTRSTDNQAVIKDALQGIQQQIKGLADKIGTEIGPKVSLIDTSSTITIPANIGLLGSKISQSTASINENVNEK"
)  # first 75 residues of nipah F (truncated for trial speed)

DESIGNS_PER_SCHEME = 10
TRIAL_SCHEMES = {
    "baseline":          None,
    "electrostatics_5x": {"fa_elec": 5.0},
    "hbond_5x":          {"hbond": 5.0},
}


@app.function(
    gpu="A100-40GB",
    timeout=int(4 * 60 * 60),  # 4 hr per worker
    volumes={
        "/results": modal.Volume.from_name("tmol-jax-trial", create_if_missing=True),
    },
)
def design_one(scheme_name: str, seed: int):
    """Run one design with the given weight scheme."""
    import sys, time, uuid
    sys.path.insert(0, "/root/tmol_jax_src")

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import numpy as np

    from mosaic.models.boltz2 import Boltz2
    import mosaic.losses.structure_prediction as sp
    from mosaic.common import TOKENS
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.losses.transformations import NoCys
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.structure_prediction import TargetChain
    from mosaic.optimizers import simplex_APGM

    # Our tmol_jax interface loss
    from examples.integration_mosaic_hallucinator import (
        build_pose_stack_template,
        TmolInterfaceLoss,
        TmolClashFilter,
    )

    folder = Boltz2()
    mpnn = load_mpnn_sol(0.05)

    weights = TRIAL_SCHEMES[scheme_name]

    bias = jnp.zeros((BINDER_LENGTH, 20)).at[:, TOKENS.index("C")].set(-1e6)

    # Build tmol_jax template for the binder+target length (no specific sequence
    # yet — use all-alanine as a placeholder; sidechains are sequence-dependent
    # but the atom37 scaffold is what tmol_jax consumes)
    ps_tpl = build_pose_stack_template(
        sequence="A" * BINDER_LENGTH,
        target_sequence=TARGET_SEQUENCE,
    )

    # Build the design loss: mosaic base + our tmol interface term
    tmol_iface = TmolInterfaceLoss(
        ps_tpl, term_weights=weights, scale=1e-3, binder_chain_id=0
    )
    tmol_clash = TmolClashFilter(ps_tpl, scale=1e-2)

    sp_loss = (
        sp.BinderTargetContact()
        + sp.WithinBinderContact()
        + 10.0 * InverseFoldingSequenceRecovery(
            mpnn, temp=jnp.array(0.001), bias=bias
        )
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
        # ─── NEW terms ────────────────────────────────────────────
        + 1.0 * tmol_iface          # binding-affinity proxy
        + 0.01 * tmol_clash         # clash filter
    )

    features, _ = folder.binder_features(
        binder_length=BINDER_LENGTH,
        chains=[TargetChain(sequence=TARGET_SEQUENCE, use_msa=True)],
    )

    loss = NoCys(
        folder.build_multisample_loss(
            loss=sp_loss,
            features=features,
            recycling_steps=1,
            num_samples=4,
        )
    )

    @eqx.filter_jit
    def evaluate_loss(loss, pssm, key):
        return loss(pssm, key=key)

    # Seed random PSSM
    _pssm = np.random.RandomState(seed).uniform(0.25, 0.75) * jax.random.gumbel(
        key=jax.random.key(seed), shape=(BINDER_LENGTH, 19)
    )

    t0 = time.time()
    # 1. Soft PSSM optimization
    _, pssm = simplex_APGM(
        loss_function=loss,
        x=jax.nn.softmax(_pssm),
        n_steps=100,
        stepsize=0.2 * np.sqrt(BINDER_LENGTH),
        momentum=0.3,
        scale=1.00,
        logspace=False,
        max_gradient_norm=1.0,
    )
    # 2. Sharpen to discrete
    pssm, _ = simplex_APGM(
        loss_function=loss,
        x=jnp.log(pssm + 1e-5),
        n_steps=50,
        stepsize=0.5 * np.sqrt(BINDER_LENGTH),
        momentum=0.0,
        scale=1.25,
        logspace=True,
        max_gradient_norm=1.0,
    )
    pssm, _ = simplex_APGM(
        loss_function=loss,
        x=jnp.log(pssm + 1e-5),
        n_steps=15,
        stepsize=0.5 * np.sqrt(BINDER_LENGTH),
        momentum=0.0,
        scale=1.4,
        logspace=True,
        max_gradient_norm=1.0,
    )
    pssm = NoCys.sequence(pssm)
    seq = pssm.argmax(-1)
    seq_str = "".join(TOKENS[i] for i in seq)
    t_design = time.time() - t0

    # Score with full-sequence reprediction
    boltz_features, boltz_writer = folder.target_only_features(
        chains=[
            TargetChain(sequence=seq_str, use_msa=False),
            TargetChain(sequence=TARGET_SEQUENCE, use_msa=True),
        ]
    )
    ranking_loss = folder.build_multisample_loss(
        loss=1.0 * sp.IPTMLoss()
             + 0.5 * sp.TargetBinderIPSAE()
             + 0.5 * sp.BinderTargetIPSAE(),
        features=boltz_features,
        recycling_steps=3,
        num_samples=6,
    )
    rl_value, _ = evaluate_loss(
        ranking_loss, jax.nn.one_hot(seq, 20), key=jax.random.key(seed + 1)
    )

    result = {
        "scheme": scheme_name,
        "seed": seed,
        "sequence": seq_str,
        "ranking_loss": float(rl_value.item()),
        "design_time_sec": t_design,
    }

    # Persist to volume
    wid = str(uuid.uuid4())[:8]
    out_path = f"/results/{scheme_name}_{seed}_{wid}.txt"
    with open(out_path, "w") as f:
        f.write(f"# {scheme_name} seed={seed} rank_loss={result['ranking_loss']:.4f}\n")
        f.write(f"# t_design={t_design:.1f}s\n")
        f.write(f"{seq_str}\n")

    return result


@app.local_entrypoint()
def run_trial():
    """Launch the full 30-design trial."""
    jobs = []
    for scheme in TRIAL_SCHEMES:
        for seed in range(DESIGNS_PER_SCHEME):
            jobs.append((scheme, seed))

    print(f"Launching {len(jobs)} designs across {len(TRIAL_SCHEMES)} schemes...")
    results = list(design_one.starmap(jobs))

    # Summary
    print(f"\n## Trial summary")
    for scheme in TRIAL_SCHEMES:
        xs = [r for r in results if r["scheme"] == scheme]
        if not xs:
            continue
        losses = [r["ranking_loss"] for r in xs]
        times = [r["design_time_sec"] for r in xs]
        print(
            f"  {scheme:20s}  n={len(xs):2d}  "
            f"loss={np.mean(losses):+.3f}±{np.std(losses):.3f}  "
            f"t={np.mean(times):.0f}s"
        )
    total_gpu_hrs = sum(r["design_time_sec"] for r in results) / 3600
    print(f"\nTotal GPU-hours: {total_gpu_hrs:.1f}")
    print(f"Estimated cost (A100 40GB @ $1.23/hr): ${total_gpu_hrs * 1.23:.2f}")
