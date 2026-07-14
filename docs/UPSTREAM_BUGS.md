# Upstream tmol bugs

This page catalogues bugs identified in upstream
[uw-ipd/tmol](https://github.com/uw-ipd/tmol) during jtmol parity
validation. Each entry has a one-line summary, a minimal reproducer, the
location in the upstream source where known, and the workaround in jtmol.

Each was identified by running the upstream wheel
`tmol-0.1.22+cu128torch2.10` on A100-40GB in the parity harness
([`benchmarks/exact_upstream_matrix.py`](../benchmarks/exact_upstream_matrix.py)) and
comparing forward and gradient values cell-by-cell against jtmol.
Upstream source is unmodified; the workarounds live entirely on the jtmol
side or in the harness.

## Fixed on upstream master (not yet in the pinned wheel)

| Bug | Fix | Wheel status |
|-----|-----|--------------|
| #2 cart_lengths / cart_angles silently 0 | `f4d0916` "Key on hash of cart-bonded params in PBT caching" (#383) | Not in `tmol-0.1.22+cu128torch2.10`; the harness still sees `0.0`. |

No other catalogued bug has an upstream fix on current master.

## 1. `lk_bridge_uncpl` gradient at sp2 backbone-O acceptors

Present in upstream 0.1.22 and current master; the adjoint math at sp2
backbone-O acceptors is unchanged by recent commits.

Full reproducer and trace:
[`UPSTREAM_BUG_LK_BRIDGE_UNCPL.md`](UPSTREAM_BUG_LK_BRIDGE_UNCPL.md).

**Summary.** Upstream's hand-written CUDA adjoint
`LKBallPoseScoreOp::backward` disagrees with central finite-difference
of its own forward by up to **0.16 kcal/mol/Å** at carbonyl-O atoms on
sp2 backbone acceptors. Forward energy is correct; only the gradient
is wrong.

**Impact in the parity harness.** 10 cells in
[`PARITY.md`](PARITY.md) (5 PDBs × `lk_bridge` +
`lk_bridge_uncpl`) show jtmol-vs-upstream gradient mismatches in the
0.04 – 0.16 kcal/mol/Å range. jtmol's `jax.grad` matches FD to ~1e-9
in float64; upstream's autograd does not.

**jtmol workaround.** None. jtmol returns the mathematically correct
derivative. The parity harness flags these cells as expected
mismatches in the per-term notes column.

## 2. `CartBondedEnergyTerm` silently returns 0 for `cart_lengths` and `cart_angles`

Candidate fix `f4d0916` ("Key on hash of cart-bonded params in PBT
caching", #383) is on upstream master but not in the pinned
`tmol-0.1.22+cu128torch2.10` wheel. Under that wheel these two channels
return `0.0`, so the YAML-golden fallback for 1ubq is load-bearing.

**Symptom.** Upstream's compiled `CartBondedEnergyTerm` kernel returns
exactly `0.0` for the 2-body (`cart_lengths`) and 3-body (`cart_angles`)
sub-channels of `beta_nov2016_cart` on every PDB tested, while the
4-body sub-channels (`cart_torsions`, `cart_impropers`,
`cart_hxltorsions`) materialize correctly with the same setup pattern.

**Reproducer.** Inside a Modal container with the upstream wheel
installed (see `benchmarks/exact_upstream_matrix.py` for the full setup):

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
across all 5 PDBs in the panel (1ubq, 1qys, 1r21, 3plc, 6DMZ_A), and
with every weight-setup pattern tested (single-target via
`ScoreFunction.set_weight`, all-5 set to 1.0, direct term-level scoring).
The setup pattern above is the exact one used by upstream's own
`EnergyTermTestBase.test_whole_pose_scoring_10` at
`tmol/tests/score/common/test_energy_term.py`, which generates the
bundled YAML goldens at
`tmol/tests/data/term_baselines/CartBondedEnergyTerm/test_whole_pose_scoring_10.yaml`.
Those goldens contain the correct values
(`term0` = +37.762332916259766, `term1` = +183.56971740722656 on 1ubq),
which proves the values are computable upstream; the v0.1.22 compiled
wheel just no longer emits them via the `render_whole_pose_scoring_module`
path.

**Likely cause.** Hash collision in the 2-body and 3-body subgraph
enumeration in upstream's compiled `cartbonded_pose_scores` op. The
4-body subgraphs (torsions / impropers / hxltorsions) use a wider hash
key with the same enumeration code, so they materialize. Upstream commit
`f4d0916` "Key on hash of cart-bonded params in PBT caching" (#383) is a
candidate fix on master but is not in v0.1.22.

**Impact in the parity harness.** 10 cells in
[`PARITY.md`](PARITY.md) (5 PDBs × `cart_lengths` +
`cart_angles`) showed `E(ref) = +0.0000` while jtmol emitted the
correct values. The harness falls back to the bundled YAML goldens
for the 1ubq cells, turning 4 of those 10 forward-FAILs into honest
comparisons against a real upstream number. The other 6 cells stay
flagged as a known upstream-wheel limitation.

**jtmol workaround.** None at the implementation level. jtmol's
`cart_lengths` and `cart_angles` match the YAML goldens to ≤ 0.004%
(see [`PARITY.md`](PARITY.md)). The harness uses the YAML golden as
the reference for 1ubq and reports the other PDBs as "no reference
available." An upstream patch that re-enables the per-channel emission
from `render_whole_pose_scoring_module` would close this category.

## 3. `ScoreFunction.set_weight(st, 0.0)` raises `AttributeError`

Present on upstream master: `set_weight` calls the unimplemented
`remove_term_for_score_type` (marked `# TO DO!`) when the last non-zero
weight on a term is zeroed out.

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

**Impact.** Never call `set_weight(_, 0.0)` to "disable" a term. Use a
fresh `ScoreFunction` per channel instead (weights start at 0 by
construction).

**jtmol workaround.** Build a fresh ScoreFunction per term (the
"declare the one weight you want" pattern). See
`scripts/run_upstream_parity.py` and `benchmarks/exact_upstream_matrix.py`
for the worked example.

## 4. `CartBondedEnergyTerm` looks up `CYS` params for `CYD` residues but emits the wrong hash key

Present on upstream master (unchanged by `f4d0916`). The Python-side
`temp_hack_base_name = "CYS" if … "CYD"` remap is in place, but the
compiled-side hash key uses `CYD:<atom>`, so CYD intramolecular
impropers silently zero.

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

**jtmol workaround.** Replicate the hash-miss deliberately: drop the
CYD intra-residue params so jtmol returns +2.1039 (matching upstream)
rather than the +2.5253 that CYS-params-applied-to-CYD would give. This
is one of the few places jtmol replicates an upstream bug for parity
rather than fixing it. The upstream source flags it as a `# TODO!`.

## 5. Compiled cubic-Hermite kink adjoints use float32 internally even with float64 input

Present on upstream; the cubic-Hermite kernels are only patchable inside
upstream's C++/CUDA, not from Python.

**Symptom.** On the five score channels whose gradient chains through
a cubic-Hermite kink (`omega`, `cart_torsions`, `cart_impropers`,
`cart_hxltorsions`, `dunbrack_rotdev`), upstream's compiled adjoint
delivers `max|dg|` floor of ~5e-5 to ~3e-2 against a pure-float64
reference, regardless of whether the input `coords` tensor is float32
or float64. Promoting jtmol's coords to float64 makes the diff worse
on `cart_torsions` (~3.7e-5 → ~3.0e-2 across 4 PDBs), moving away
from upstream's precision-limited adjoint rather than toward it.

**Reproducer.** Load 1ubq, score `cart_torsions` only on both jtmol
(in `jax_enable_x64=True` mode) and upstream torch tmol (`coords.double()`),
take `jax.grad` and `torch.autograd.grad`, compare element-wise:

```
1ubq  cart_torsions  jax.grad vs torch float32 path: max|dg| = 3.7e-5
1ubq  cart_torsions  jax.grad vs torch float64 path: max|dg| = 3.0e-2  (worse)
```

The upstream adjoint's internal `dE/dx` computation drops to float32
at some intermediate step (most likely the Blondel-Karplus BAH-frame
Jacobian product) and the float64 promotion of the input does not flow
through. Reverting loader-side float64 promotion returns all 4 PDBs'
cart_torsions cells to the ~3e-5 baseline.

**jtmol workaround.** Leave `pose.coords` as float32 in the loader by
default. The parity harness reports the residual as a documented
"float32-internal-adjoint floor on kink terms."

**Bigger picture.** This is a property of upstream's hand-tuned compiled
kernels, not a single patchable bug. Resolving it requires either a
per-term, per-kernel float64 promotion inside upstream's C++/CUDA
source, or acceptance that upstream's gradients carry this floor. Until
upstream re-issues the kernels at higher precision, the 9 cells in
[`PARITY.md`](PARITY.md) with `max|dg|` in the ~1e-4
range stay flagged as documented residuals rather than parity failures.

## 6. `3plc` dunbrack_rot / dunbrack_rotdev residual is upstream-side (stale golden)

The recorded 3plc `dunbrack_rot` / `dunbrack_rotdev` `E(ref)` goldens
are not reproducible from the actual pose+math; they are stale /
cross-run-inconsistent. jtmol's rotameric dunbrack is a verified
faithful port. No jtmol code change is warranted (any change regresses
the four other panel PDBs, which pass at ≤ 6e-06%).

**Symptom (recorded).** 3plc `dunbrack_rot` E(jax)=+52.9200 vs
E(ref)=+55.1774 (4.1%); `dunbrack_rotdev` E(jax)=+261.6107 vs
E(ref)=+331.0977 (21%); `dunbrack_semirot` E(jax)=E(ref)=+46.1987 (0%).
Localised to THR-26 / ASP-35, the residues bracketing the
missing-density gap (PDB residues 27–34 absent).

**What the gap is.** A real ~5 Å break: `C(THR26)–N(ASP35) = 5.162 Å`
(a peptide bond is ~1.33 Å). jtmol's `pose_stack_from_pdb` keeps a
polymer bond across it (gap-bridging behaviour), so it synthesizes
phi/psi across the fictional bond:

```
THR-26  psi = +139.99°   (N-CA-C(THR26)-N(ASP35), frac index ≈ 32.0 in a 37-wide grid)
ASP-35  phi = +14.18°    (C(THR26)-N-CA-C(ASP35),  frac index ≈ 19.4)
```

Both land in the grid **interior**, not near any boundary/periodicity
seam, and the spline is smooth there (sweeping THR-26 psi over
135°–150° moves its rotdev by < 1.4 units, from 18.4 to 19.7).

**Enumeration.** Recomputing jtmol's per-residue 3plc dunbrack directly
and enumerating every phi/psi assignment of the two gap residues
(bridged real angle vs upstream's undefined-dihedral sentinel φ=−60° /
ψ=+60° from `dunbrack_pose_score.impl.hh`):

```
                                 rot       rotdev    semirot
GOLDEN  E(ref)                   55.1774   331.0977  46.1987
THR26=bridge   ASP35=bridge      52.9200   261.6107  46.1987   <- current jtmol
THR26=bridge   ASP35=sentinel    52.9200   261.5082  46.3579
THR26=sentinel ASP35=bridge      52.0355   296.1244  46.1987
THR26=sentinel ASP35=sentinel    52.0355   296.0219  46.3579
```

Two conclusions:
1. **The golden pose == jtmol's pose.** `semirot` is bit-identical
   (46.1987) *only* when ASP-35 keeps its bridged phi (=+14.18°);
   the sentinel flips it to 46.3579. So the upstream run that produced
   the goldens used the same gap-bridged connectivity and the same
   coordinates as jtmol. Every non-gap residue therefore has identical
   phi/psi/chi on both sides (else `semirot` would not be 0%), so their
   rot/rotdev must be identical too.
2. **No gap-residue configuration reaches the golden.** The golden
   rot/rotdev sit *above* even the maximal (sentinel-THR26) case
   (52.04 / 296.12). The two gap residues cannot supply the missing
   +2.26 (rot) / +34.97 (rotdev) under any phi/psi, and every other
   residue is provably identical. The golden rot/rotdev are internally
   inconsistent with their own semirot cell.

**Verified identical between jtmol and upstream** (so the residual is
not a port bug): rotamer classification (`classify_rotamer`, ±120°/0
bins), dihedral wrap (`wrap ∈ [0, period]` vs jtmol `mod`, which differ
only exactly at the seam, and that cancels under the periodic 4-cell
gather), the bspline (pole √3−2, λ, weight polynomials, `interpolate`
index math, `bspline.hh` line-for-line), the mean pre-shift
(`means_deg[<−120]+=360`, `params.py:_calculate_rot_mean_coeffs` ==
`dunbrack_energy_term.py:478`), the sdev tables (upstream applies no
floor; jtmol's `where(sdev>1e-6, sdev, 1.0)` guard never fires,
0 of 942 192 cells are ≤ 1e-6 rad), and the chidev formula
(`chi<−120° ? chi+2π−mean : chi−mean`, `/2σ²`).

**Diagnosis.** Upstream-side stale / cross-run-inconsistent golden.
jtmol is a faithful port of upstream at these residues. The recorded
3plc rot/rotdev `E(ref)` column does not correspond to the pose that
produced its own semirot `E(ref)`.

**Regenerating the golden.** Rebuilding upstream's compiled C++/CUDA
extension and re-running the 3plc dunbrack live is expected to return
52.9200 / 261.6107, matching jtmol and closing the cells.

**Reproducer (local, no upstream build needed).** The per-residue +
combination sweep above runs entirely on jtmol against
`tmol/tests/data/pdb/3plc.pdb`; THR-26 is block 25, ASP-35 is
block 26 (0-indexed). See also
[`PARITY.md`](PARITY.md) §"Known residual: 3plc".

**jtmol workaround.** None needed; no jtmol bug. Do not "fix" the
gap-residue dunbrack: it would push jtmol away from its faithful-port
values and regress the four panel PDBs at ≤ 6e-06% parity.

## 7. `gen_torsions` term (ported)

Ported in `jtmol/score/genbonded/`, mirroring upstream PR
[#361](https://github.com/uw-ipd/tmol/pull/361): bare-cosine 4-period
proper torsions with negative-K lift + offset, plus harmonic impropers.

**On 1ubq the term evaluates to 0.0**, matching upstream's gold value
(`ScoreType.gen_torsions: n([0.0])` in
`tmol/tests/score/test_score_function.py`). Why zero: the upstream
`genbonded.yaml` `atoms:` hierarchy only defines fall-backs for
generic ligand chem types (`CS`, `CD`, `Nad`, ...). Canonical protein
residues are still typed with `Nbb`, `CAbb`, `CObb`, `Hapo`, ... in
both upstream and jtmol. Types absent from the hierarchy mean every
torsion is a hash-miss → zero contribution. The term is fully wired
for any future ligand workflow / YAML extension.

## How to add a new upstream-bug entry

When a parity run surfaces a new upstream-side mismatch:

1. Add a numbered section here following the template above:
   summary, reproducer, likely cause, impact in the harness, jtmol
   workaround.
2. If the bug has a clean stand-alone reproducer, also add it as a
   dedicated `docs/UPSTREAM_BUG_<NAME>.md` file (see
   `UPSTREAM_BUG_LK_BRIDGE_UNCPL.md` for the template).
3. Update the relevant per-term notes column in
   [`PARITY.md`](PARITY.md) so readers of the
   parity table can immediately see why a cell fails.
4. Cross-link from [`PARITY.md`](PARITY.md) if the bug affects the
   canonical 1ubq table.
