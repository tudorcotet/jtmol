# Parity against upstream tmol on 1ubq

Reference values come from upstream's per-term test goldens at
`tmol/tests/data/term_baselines/<Term>/test_whole_pose_scoring_10.yaml`.
All numbers below are unweighted per-pose energies on the single-pose
case (1UBQ, 76 residues).

## Final parity table (sorted by % error)

| Term | tmol_jax | upstream | abs err | % err |
|---|---:|---:|---:|---:|
| fa_ljatr | -417.9583 | -417.9583 | 0.0000 | 0.000% |
| fa_ljrep | 240.7147 | 240.7147 | 0.0000 | 0.000% |
| fa_lk | 298.2765 | 298.2765 | 0.0000 | 0.000% |
| fa_elec | -136.2925 | -136.2925 | 0.0000 | 0.000% |
| rama | -12.7434 | -12.7434 | 0.0000 | 0.000% |
| dunbrack_rot | 70.6497 | 70.6497 | 0.0000 | 0.000% |
| lk_ball_iso | 422.0395 | 422.0396 | 0.0001 | 0.000% |
| lk_ball | 172.1964 | 172.1964 | 0.0000 | 0.000% |
| lk_bridge_uncpl | 10.9946 | 10.9946 | 0.0000 | 0.000% |
| cart_angles | 183.5697 | 183.5697 | 0.0000 | 0.000% |
| cart_torsions | 46.0236 | 46.0236 | 0.0000 | 0.000% |
| cart_impropers | 9.4305 | 9.4305 | 0.0000 | 0.000% |
| cart_hxltorsions | 47.4197 | 47.4197 | 0.0000 | 0.000% |
| ref | -41.2750 | -41.2750 | 0.0000 | 0.000% |
| lk_bridge | 1.5786 | 1.5786 | 0.0000 | 0.001% |
| omega | 4.1002 | 4.1003 | 0.0001 | 0.002% |
| cart_lengths | 37.7637 | 37.7623 | 0.0014 | 0.004% |
| dunbrack_semirot | 99.9438 | 99.6609 | 0.2829 | 0.284% |
| dunbrack_rotdev | 239.5350 | 240.3101 | 0.7751 | 0.323% |
| hbond | -55.2215 | -55.6756 | 0.4541 | 0.816% |

**Result: 20/20 terms match within 1%. Mean absolute %err: 0.071%. Max %err: 0.816%.**

## What closed the gap

From the initial 30-agent port to current state:

| Fix | Before | After | Terms affected |
|---|---:|---:|---|
| Cart-bonded ideal-K + phantom-atom mask | 14000% | 0% | cart_lengths, cart_impropers |
| Rama bicubic B-spline | 35% | 0% | rama |
| Omega B-spline + prepro branch + cis fallback | 3.5% | 0% | omega |
| Dunbrack bilinear interp + semirot mask | 76% | 0% | dunbrack_rot |
| LK-ball sp2/sp3/ring water placement | 155-412% | 0% | lk_ball_* |
| N-term/C-term variant auto-detection | 5.4% | 0% | fa_elec, hbond, LJ/LK |
| Floyd-Warshall APSP on residue-connection graph for count-pair | 14000% → 3.4% | 0% | fa_ljatr, fa_ljrep, fa_lk |
| Rama prepro-table lookup | 0.7% | 0% | rama |
| Omega N-terminal cis-harmonic fallback | 0.7% | 0% | omega |

## Residual explanations (all < 1%)

### hbond 0.82%
- B/B0 acceptor-base selection uses bond-graph (first/second heavy
  neighbour) vs upstream's icoor-tree walk. Difference on ring
  acceptors (HIS NE2) and sp3 hydroxyls. Residual is 0.45 kcal/mol
  across 1UBQ's ~30 H-bonds.

### dunbrack_rotdev 0.32%
- Chi-mean / chi-sigma tables are bilinear-interpolated; upstream uses
  bicubic B-spline. Other two Dunbrack channels match exactly with
  bilinear.

### dunbrack_semirot 0.28%
- Last-chi neglnprob table is bilinear-interpolated; upstream uses
  bicubic on the full 3D (phi, psi, chi_last) table.

### cart_lengths 0.004%
- 0.0014 kcal/mol distributed across ~1230 bonds. float64 gives the
  same result, so it's algorithmic not precision — likely a subtle
  inter-residue bond pairing edge case.

## Reproducibility

```bash
cd /Users/tudorcotet/Documents/Adaptyv/tmol-jax
/opt/homebrew/bin/python3.13 -c "
import warnings; warnings.filterwarnings('ignore')
import jax.numpy as jnp
import tmol_jax
from tmol_jax.score.score_types import ScoreType

ps = tmol_jax.pose_stack_from_pdb('tmol_jax/tests/data/1ubq.pdb')
sfxn = tmol_jax.beta2016_score_function()
for st in ScoreType:
    if st.name == 'n_score_types': continue
    try: sfxn.set_weight(st, 1.0)
    except: pass
per = sfxn.score_by_term(ps)
for nm in ['fa_ljatr','fa_ljrep','fa_lk','fa_elec','hbond','rama','omega','dunbrack_rot','dunbrack_rotdev','dunbrack_semirot','cart_lengths','cart_angles','cart_torsions','cart_impropers','cart_hxltorsions','lk_ball_iso','lk_ball','lk_bridge','lk_bridge_uncpl','ref']:
    v = float(jnp.asarray(per.get(getattr(ScoreType, nm), jnp.zeros(1))).reshape(-1)[0])
    print(f'{nm:22s} {v:12.4f}')
"
```
