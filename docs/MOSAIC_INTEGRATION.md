# Using jtmol as a binding-affinity loss in mosaic / Boltz-2 design

jtmol (the JAX-native port of Rosetta `beta_nov2016_cart` energy) plugs
into the [escalante-bio/mosaic](https://github.com/escalante-bio/mosaic)
hallucination loop as an additional `LossTerm`. The headline use case is
to add **physics-based binding-affinity gradient signal** alongside
mosaic's ML structure-prediction losses, without giving up any of the
existing iPTM / PAE / pLDDT pressure.

This guide assumes:
* you have mosaic installed and a working hallucination script (e.g.
  the [Nick Boyd gist](https://gist.github.com/nboyd/7b30c4033f834fc532a106d902fbdffa) hallucinator),
* you have `jtmol` installed (`uv pip install -e .` or `pip install -e .` from this repo).

## 1. The three loss terms

`examples/integration_mosaic_hallucinator.py` exposes three drop-in
`LossTerm` subclasses:

```python
from examples.integration_mosaic_hallucinator import (
    build_pose_stack_template,
    TmolInterfaceLoss,    # binding-affinity proxy (binder ↔ target only)
    TmolBackboneLoss,     # cart_bonded + rama + omega; design-time
    TmolClashFilter,      # fa_ljrep alone; sharp gradient out of clashes
)
```

Each composes with mosaic losses via `+` / `*` (same algebra as
`mosaic.common.LossTerm`):

```python
loss = 0.05 * sp.PLDDTLoss() + 1.0 * tmol_interface
```

## 2. Recipe — drop into the nipah-binder hallucinator

```python
import jax.numpy as jnp
import equinox as eqx

# mosaic imports (unchanged from the original gist)
from mosaic.models.boltz2 import Boltz2
import mosaic.losses.structure_prediction as sp
from mosaic.common import TOKENS
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import NoCys
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.structure_prediction import TargetChain
from mosaic.optimizers import simplex_APGM

# jtmol imports
from examples.integration_mosaic_hallucinator import (
    build_pose_stack_template,
    TmolInterfaceLoss,
    TmolClashFilter,
)

BINDER_LENGTH = 220
TARGET_SEQUENCE = "..."  # nipah F sequence

# Build the jtmol PoseStack template ONCE per design run (sequence-
# dependent, coord-independent — the design loop swaps in new atom37
# coords every step via jnp.take, which is jit/grad-safe).
ps_tpl = build_pose_stack_template(
    sequence="X" * BINDER_LENGTH,        # placeholder during design
    target_sequence=TARGET_SEQUENCE,
)

# Binding-affinity proxy with per-term knobs
tmol_interface = TmolInterfaceLoss(
    ps_tpl,
    term_weights={                       # ← tune per the experiment
        "fa_elec": 5.0,                  #   bias toward electrostatic
        "hbond": 5.0,                    #   complementarity / H-bonds
    },
    scale=1e-3,                          # Rosetta energies are 100s
    binder_chain_id=0,                   # of kcal/mol; scale down
)

# Cheap clash filter
tmol_clash = TmolClashFilter(ps_tpl, scale=1e-2)

# mosaic standard design loss + our two new terms
folder = Boltz2()
mpnn = load_mpnn_sol(0.05)
bias = jnp.zeros((BINDER_LENGTH, 20)).at[:, TOKENS.index("C")].set(-1e6)

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
    # ─── jtmol additions ───────────────────────────
    + 1.0 * tmol_interface       # binding-affinity proxy
    + 0.01 * tmol_clash          # clash filter
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

# … rest of the design loop is unchanged from the gist.
```

That's it. `simplex_APGM` picks up the physics gradient automatically
because both Boltz-2 and jtmol are pure JAX — `jax.grad` flows through
both layers.

## 3. The per-term knob (the experimental axis)

Different weight schemes implement different design hypotheses.
Empirically (jtmol's own H1 sweep over 9 PDBs × 11 schemes) most
LJ-containing reweightings share gradient directions
(cosine similarity > 0.93 to baseline); the schemes that genuinely
**point in a different direction** are the ones that zero out
LJ entirely:

```python
# Hypothesis: salt-bridge-driven affinity
TmolInterfaceLoss(ps_tpl, term_weights={"fa_elec": 5.0, "hbond": 5.0})

# Hypothesis: shape complementarity (mostly hydrophobic burial)
TmolInterfaceLoss(ps_tpl, term_weights={"fa_ljatr": 2.0, "fa_ljrep": 2.0})

# Hypothesis: polar-only interface (zeros LJ)
TmolInterfaceLoss(ps_tpl, term_weights={
    "fa_ljatr": 0.0, "fa_ljrep": 0.0, "fa_lk": 0.0,
})

# Hypothesis: water-bridged binding (relevant for elongated polar
# epitopes where waters mediate the contact)
TmolInterfaceLoss(ps_tpl, term_weights={
    "lk_ball": 3.0, "lk_bridge": 3.0, "lk_bridge_uncpl": 3.0,
})
```

Sweep these as a design hyperparameter. mosaic's existing ranking loss
(IPTM + IPSAE) is unaffected — apply at the end as before.

## 4. Ranking-time full physics

At the post-design reprediction step, use the full beta_nov2016 score
function instead of just the interface term:

```python
from examples.integration_mosaic_hallucinator import TmolFullAtomLoss

ps_tpl_full = build_pose_stack_template(
    sequence=designed_seq_str,
    target_sequence=TARGET_SEQUENCE,
)
tmol_full = TmolFullAtomLoss(ps_tpl_full, scale=1e-3)

ranking_loss = folder.build_multisample_loss(
    loss=1.0 * sp.IPTMLoss()
         + 0.5 * sp.TargetBinderIPSAE()
         + 0.5 * sp.BinderTargetIPSAE()
         + 0.1 * tmol_full,          # physics correctness check
    features=boltz_features,
    recycling_steps=3,
    num_samples=6,
)
```

This gives a physics-correctness component to the ranking score so
that designs which look great to the ML model but have hidden geometric
issues (clashes, broken bonds) get penalized.

## 5. Cost overhead

jtmol adds a small fixed cost per Boltz-2 step:

| operation | ms / step (CPU, M3 Max, 1228 atoms) |
|---|---:|
| forward (all 20 terms, neighbor-list) | ~1.5 |
| gradient | ~3.3 |
| Boltz-2 step (B200 GPU, ~700 res) | ~5000–10000 |

So jtmol is **rounding error** vs Boltz-2 — full physics-augmented
design has the same wall-clock as vanilla mosaic.

## 6. Parity guarantees

jtmol is parity-verified against upstream PyTorch tmol:

* **Forward energy:** 20/20 score terms bit-exact on 1ubq, 1qys
  (full beta_nov2016_cart). See `docs/UPSTREAM_PARITY.md`.
* **Gradient:** matches central finite-difference of upstream's forward
  to ~1e-9 in float64 across all terms. (One known upstream bug at
  sp2 backbone-O acceptors documented in
  `docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md`; jtmol gives the
  mathematically correct gradient there.)
* **Multi-chain:** matches upstream on 1cgi, 7opb, 2oob; refuses
  1BL8 with the same error message as upstream.

## 7. Open knobs

| Knob | Where | Default | When to change |
|---|---|---|---|
| `term_weights` | `TmolInterfaceLoss(...)` | beta_nov2016 | always — this is the experimental axis |
| `scale` | `TmolInterfaceLoss(...)` | `1e-3` | larger if you want physics to dominate ML pressure |
| `binder_chain_id` | `TmolInterfaceLoss(...)` | `0` | for multi-binder designs |
| `use_neighbor_list` | (set on the score function) | True | set False for small systems where dense is faster |

## 8. References

* jtmol parity table: `docs/UPSTREAM_PARITY.md`
* Why JAX port: `docs/WHY_JAX.md`
* Architecture: `docs/ARCHITECTURE.md`
* Upstream bug we found: `docs/UPSTREAM_BUG_LK_BRIDGE_UNCPL.md`
* mosaic source: https://github.com/escalante-bio/mosaic
* Original hallucinator gist: https://gist.github.com/nboyd/7b30c4033f834fc532a106d902fbdffa
* Upstream tmol: https://github.com/uw-ipd/tmol
