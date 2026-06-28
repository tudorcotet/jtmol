# Upstream tmol bugs found while building jtmol

This page catalogues every bug we have identified in upstream
[uw-ipd/tmol](https://github.com/uw-ipd/tmol) while validating jtmol
parity. Each entry has a one-line summary, a minimal reproducer, the
location in the upstream source if we have it, and the workaround in
jtmol.

The bugs were found by running the upstream wheel `tmol-0.1.22+cu128torch2.10`
on A100-40GB inside the parity harness at
[`examples/modal_parity_run.py`](../examples/modal_parity_run.py) and
comparing forward + gradient values cell-by-cell against jtmol. We did
not modify upstream source; the workarounds live entirely on the
jtmol side or in the harness.

## 1. `lk_bridge_uncpl` gradient at sp2 backbone-O acceptors

Full reproducer and trace: [`UPSTREAM_BUG_LK_BRIDGE_UNCPL.md`](UPSTREAM_BUG_LK_BRIDGE_UNCPL.md).

**Summary.** Upstream's hand-written CUDA adjoint
`LKBallPoseScoreOp::backward` disagrees with central finite-difference
of its own forward by up to **0.16 kcal/mol/Å** at carbonyl-O atoms on
sp2 backbone acceptors. Forward energy is correct; only the gradient
is wrong.

**Impact in the parity harness.** 10 cells in
[`UPSTREAM_PARITY.md`](UPSTREAM_PARITY.md) (5 PDBs × `lk_bridge` +
`lk_bridge_uncpl`) show jtmol-vs-upstream gradient mismatches in the
0.04 – 0.16 kcal/mol/Å range. jtmol's `jax.grad` matches FD to ~1e-9
in float64; upstream's autograd does not.

**jtmol workaround.** None — jtmol returns the mathematically correct
derivative. The parity harness flags these cells as expected
mismatches in the per-term notes column.

## 2. `CartBondedEnergyTerm` silently returns 0 for `cart_lengths` and `cart_angles`

**Symptom.** Upstream's compiled `CartBondedEnergyTerm` kernel returns
exactly `0.0` for the 2-body (`cart_lengths`) and 3-body (`cart_angles`)
sub-channels of `beta_nov2016_cart` on every PDB we have tested, while
the 4-body sub-channels (`cart_torsions`, `cart_impropers`,
`cart_hxltorsions`) materialize correctly with the same setup pattern.

**Reproducer.** Inside a Modal container with the upstream wheel
installed (see `examples/modal_parity_run.py` for the full setup):

```python
import torch
from tmol import pose_stack_from_pdb, ParameterDatabase
from tmol.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm
from tmol.pose.pose_stack_builder import PoseStackBuilder

ps = pose_stack_from_pdb("1ubq.pdb", torch.device("cpu"))
pn = PoseStackBuilder.from_poses([ps] * 10, device=torch.device("cpu"))
term = CartBondedEnergyTerm(ParameterDatabase.get_default(), torch.device("cpu"))
for bt in pn.packed_block_types.active_block_types:
    term.setup_block_type(bt)
term.setup_packed_block_types(pn.packed_block_types)
term.setup_poses(pn)
scorer = term.render_whole_pose_scoring_module(pn)
coords = pn.coords.detach().clone()  # float32
per_channel = scorer(coords)         # shape [5, 10]
print(per_channel[:, 0])
# tensor([0.0, 0.0, 46.0236, 9.4305, 47.4197])
#        ^    ^
#        cart_lengths, cart_angles ← both zero, should be 37.7623 and 183.5697
```

The same `[0.0, 0.0, ...]` pattern reproduces under `coords.double()`,
across all 5 PDBs in our panel (1ubq, 1qys, 1r21, 3plc, 6DMZ_A), and
with every weight-setup pattern we have tried (single-target via
`ScoreFunction.set_weight`, all-5 set to 1.0, direct term-level scoring).
The setup pattern above is the exact one used by upstream's own
`EnergyTermTestBase.test_whole_pose_scoring_10` at
`tmol/tests/score/common/test_energy_term.py`, which generates the
bundled YAML goldens at
`tmol/tests/data/term_baselines/CartBondedEnergyTerm/test_whole_pose_scoring_10.yaml`.
Those goldens contain the correct values
(`term0` = +37.762332916259766, `term1` = +183.56971740722656 on 1ubq),
which proves the values are computable upstream — the v0.1.22 compiled
wheel just no longer emits them via the `render_whole_pose_scoring_module`
path.

**Likely cause.** Hash collision in the 2-body and 3-body subgraph
enumeration in upstream's compiled `cartbonded_pose_scores` op. The
4-body subgraphs (torsions / impropers / hxltorsions) use a wider hash
key with the same enumeration code, so they materialize. Upstream's
commit history shows a recent change `f4d0916` "Key on hash of
cart-bonded params in PBT caching (#383)" which is a candidate fix on
master but is not in `v0.1.22`.

**Impact in the parity harness.** 10 cells in
[`UPSTREAM_PARITY.md`](UPSTREAM_PARITY.md) (5 PDBs × `cart_lengths` +
`cart_angles`) showed `E(ref) = +0.0000` while jtmol emitted the
correct values. The harness now falls back to the bundled YAML goldens
for the 1ubq cells (`1d163b6`), turning 4 of those 10 forward-FAILs
into honest comparisons against a real upstream number. The other
6 cells stay flagged as known upstream-wheel limitation.

**jtmol workaround.** None at the implementation level — jtmol's
`cart_lengths` and `cart_angles` match the YAML goldens to ≤ 0.004%
(see [`PARITY.md`](PARITY.md)). The harness uses the YAML golden as
the reference for 1ubq and reports the other PDBs as "no reference
available." A patch upstream that re-enables the per-channel emission
from `render_whole_pose_scoring_module` would close this category in
one go.

## 3. `ScoreFunction.set_weight(st, 0.0)` raises `AttributeError`

**Symptom.** Calling `set_weight(score_type, 0.0)` on an
`UpScoreFunction` (after a prior non-zero weight was set on any term
sharing the same Term class) triggers a code path that calls
`remove_term_for_score_type`, which is unimplemented:

```
AttributeError: 'ScoreFunction' object has no attribute 'remove_term_for_score_type'
```

**Reproducer.**

```python
from tmol import ParameterDatabase
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
sfxn = ScoreFunction(ParameterDatabase.get_default(), torch.device("cpu"))
sfxn.set_weight(ScoreType.cart_lengths, 1.0)
sfxn.set_weight(ScoreType.cart_lengths, 0.0)  # AttributeError
```

**Likely cause.** The internal `_weights` tensor is set fine, but the
public API also tries to remove the now-unused term to free its
parameters. The removal hook is a `# TODO` comment in upstream's
source.

**Impact in the parity harness.** The harness must NEVER call
`set_weight(_, 0.0)` to "disable" a term. It uses a fresh
`ScoreFunction` per channel instead (weights start at 0 by
construction). This was the root cause of the original
"10538% diff on 1CGI" in `docs/MULTI_CHAIN_PARITY.md`, fixed in commit
`33138d4`.

**jtmol workaround.** Build a fresh ScoreFunction per term (the
"declare the one weight you want" pattern). See
`scripts/run_upstream_parity.py` and `examples/modal_parity_run.py`
for the worked example.

## 4. `CartBondedEnergyTerm` looks up `CYS` params for `CYD` residues but emits the wrong hash key

**Symptom.** For disulfide-bonded cysteines (block-type `CYD`),
upstream's cart_bonded improper-dihedral params get loaded from the
`CYS` entry in the parameter database, but the runtime hash key uses
the `CYD:<atom>` prefix. The lookup misses, so upstream silently emits
0 for those improper subgraphs.

**Reproducer.** Run upstream torch tmol on 6DMZ_A (5 SG-SG disulfides,
all CYDs) and inspect `cart_impropers` per-subgraph output. Upstream
returns +2.1039 kcal/mol; the YAML golden generator produces the same
number. The CYS-derived params for the CYD intramolecular improper
chains contribute 0 in upstream because the hash key never resolves.

**jtmol workaround.** Replicate the same hash-miss behaviour
deliberately in jtmol so we don't double-count what upstream silently
skips. Commit
[`ee8c8c2`](https://github.com/tudorcotet/jtmol/commit/ee8c8c2):
`cartbonded: drop CYD intra-residue params to match upstream hash-miss`.
Before this fix, jtmol returned the "correct" +2.5253 (CYS-params
applied to CYD); after, it returns +2.1039 (matches upstream's broken
behaviour). This is one of the few places we replicate an upstream
bug for parity rather than fix it — flagged as a `# TODO!` in the
upstream source.

## 5. Compiled cubic-Hermite kink adjoints use float32 internally even with float64 input

**Symptom.** On the five score channels whose gradient chains through
a cubic-Hermite kink (`omega`, `cart_torsions`, `cart_impropers`,
`cart_hxltorsions`, `dunbrack_rotdev`), upstream's compiled adjoint
delivers `max|dg|` floor of ~5e-5 to ~3e-2 against a pure-float64
reference, regardless of whether the input `coords` tensor is float32
or float64. Promoting jtmol's coords to float64 makes the diff worse
on `cart_torsions` (~3.7e-5 → ~3.0e-2 across 4 PDBs) — moving away
from upstream's precision-limited adjoint rather than toward it.

**Reproducer.** Load 1ubq, score `cart_torsions` only on both jtmol
(in `jax_enable_x64=True` mode) and upstream torch tmol (`coords.double()`),
take `jax.grad` and `torch.autograd.grad`, compare element-wise:

```
1ubq  cart_torsions  jax.grad vs torch float32 path: max|dg| = 3.7e-5
1ubq  cart_torsions  jax.grad vs torch float64 path: max|dg| = 3.0e-2  (worse)
```

The upstream adjoint's internal `dE/dx` computation drops to float32
at some intermediate step (the Blondel-Karplus BAH-frame Jacobian
product, most likely) and the float64-promotion of the input doesn't
flow through. We confirmed this by reverting our loader-side float64
promotion (`4260ed0` → `0db601e`) and observing that all 4 PDBs'
cart_torsions cells returned to the `~3e-5` baseline.

**jtmol workaround.** Leave `pose.coords` as float32 in the loader by
default. The parity harness sees the residual as a documented
"float32-internal-adjoint floor on kink terms" and reports those cells
as known-mismatch with notes.

**Bigger picture.** This is not a single bug we can patch — it is a
property of upstream's hand-tuned compiled kernels. The right path is
either (a) a per-term, per-kernel float64 promotion inside upstream's
C++/CUDA source, or (b) acceptance that upstream's gradients carry
this floor. Until upstream re-issues the kernels at higher precision,
the 9 cells in [`UPSTREAM_PARITY.md`](UPSTREAM_PARITY.md) with
`max|dg|` in the ~1e-4 range will stay flagged as documented residuals
rather than parity failures.

## 6. `3plc` dunbrack b-spline output diverges from the header source for synthesized phi/psi

**Symptom.** On 3plc, `dunbrack_rot` differs from upstream by 4.1%
and `dunbrack_rotdev` by 21% at the two residues bracketing the
missing-density gap in the PDB (residues 27–34 are absent; THR-26 and
ASP-35 end up as virtual peptide-bond neighbours after our
gap-bridging fix in `67ac14a`). The synthesized phi/psi values at
those two residues are bit-identical between jtmol and upstream:

```
THR-26  psi = +139.99°   (synthesized from N-CA-C-THR26 + N-ASP35)
ASP-35  phi = +14.18°    (synthesized from C-THR26 + N-CA-C-ASP35)
```

jtmol's bspline interpolation at these (phi, psi) points returns a
penalty of `(52.34°)² / (2·8.39°²) = 19.44`. Upstream returns a
different value, but we cannot get inside the compiled C++ kernel
locally to see why — the agent's audit (commit `43d9b48`) confirmed
that jtmol's `tmol_jax/numeric/bspline.py` is a line-for-line port of
`tmol/numeric/bspline_compiled/bspline.hh` (cubic prefilter pole
`√3 − 2`, normalisation λ, 4-cell periodic wrap), and the
prefilter-coefficient diff between float32 and float64 is at the
~1e-7 level — five orders of magnitude too small for 21%.

So either upstream's compiled bspline diverges from its own header
source at the synthesized phi/psi points, or upstream applies some
boundary-condition handling at the gap-bridging residues that our
header read did not reveal.

**Reproducer.** Run the parity harness on 3plc and inspect the
`dunbrack_rot` / `dunbrack_rotdev` cells in
[`UPSTREAM_PARITY.md`](UPSTREAM_PARITY.md). The full diagnostic is in
[`PARITY.md`](PARITY.md) under the "3plc dunbrack residual" section.

**jtmol workaround.** None. We cannot fix this without instrumenting
upstream's compiled bspline kernel to dump its intermediate
coefficients at the gap-bridging (phi, psi) points. Doing so requires
a Modal-side modification to the harness rather than a jtmol-side
change.

## 7. `gen_torsions` term not yet ported (divergence, not a bug)

**Summary.** Upstream PR
[#361 "Kdidi/ligand clean"](https://github.com/uw-ipd/tmol/pull/361)
(merged 2026-06-26) added a new `gen_torsions` score channel to
`beta2016.sfxn` with weight 1.0. The term scores proper and improper
dihedrals using generic atom-type parameters (rather than per-residue
atom-name parameters used by the existing `cart_torsions` and
`cart_impropers`). It was motivated by ligand support but fires on
protein backbone + sidechain torsions too.

**Impact.** Upstream's `beta_nov2016_cart` total now includes a
non-zero `gen_torsions` row that jtmol silently omits. On a 1ubq-class
protein the contribution is roughly cart_torsions-scale (tens of
kcal/mol). No bundled YAML golden ships yet at
`tmol/tests/data/term_baselines/GenBondedEnergyTerm/`, so the parity
harness cannot diff this row — but upstream's score-by-term will list
one extra channel that jtmol's score-by-term will not.

**This is not an upstream bug** — it is a real new term we have not
ported. Documented here so the divergence has a single home alongside
the other parity-affecting issues. Full port surface and decision
criteria in [`UPSTREAM_ALIGNMENT.md`](UPSTREAM_ALIGNMENT.md#1-gen_torsions-not-yet-ported).

**Workaround.** None. jtmol's `beta2016_score_function` returns a
sum that is lower than upstream's by `gen_torsions`. For consumers
who only use jtmol for design-loop gradients (mosaic affinity loss,
physics-regularized hallucination), the missing term is a fixed-scale
offset that does not affect the gradient direction on any sequence-
or coord-perturbation comparison done with jtmol-vs-jtmol. For
consumers who need bit-exact agreement with upstream's total, port
the term.

## How to add a new upstream-bug entry

When the next parity run surfaces a new upstream-side mismatch:

1. Add a numbered section here following the template above —
   summary, reproducer, likely cause, impact in the harness, jtmol
   workaround.
2. If the bug has a clean stand-alone reproducer, also add it as a
   dedicated `docs/UPSTREAM_BUG_<NAME>.md` file (see
   `UPSTREAM_BUG_LK_BRIDGE_UNCPL.md` for the template).
3. Update the relevant per-term notes column in
   [`UPSTREAM_PARITY.md`](UPSTREAM_PARITY.md) so readers of the
   parity table can immediately see why a cell fails.
4. Cross-link from [`PARITY.md`](PARITY.md) if the bug affects the
   canonical 1ubq table.
