"""Guards added after an adversarial audit of proposed jtmol fixes.

Covers the three non-trivial behaviours:
  * the bonded-path-length setup caches are bounded LRU (memory-leak fix), not
    the old never-evicting plain dicts;
  * ScoreFunction(..., check_finite=True) RAISES on a non-finite score (a NaN
    coord poisons a term) rather than silently masking it;
  * run_cart_min refuses a neighbor-list score function (its NL goes stale as
    the minimizer moves atoms -> non-conservative field).
"""
from __future__ import annotations

import collections

import jax.numpy as jnp
import pytest

_PDB = "jtmol/tests/data/1ubq.pdb"


def test_bpl_cache_is_bounded_lru():
    from jtmol.score.common import count_pair as cp

    c = collections.OrderedDict()
    for i in range(3 * cp._CACHE_CAP):
        cp._cache_put(c, i, i)
    assert len(c) <= cp._CACHE_CAP, "cache is unbounded -> leaks [N,N] arrays"
    last = 3 * cp._CACHE_CAP - 1
    assert last in c and 0 not in c, "LRU should keep newest, evict oldest"


def test_check_finite_raises_not_masks():
    import jtmol

    sf = jtmol.beta2016_score_function()
    ps = jtmol.pose_stack_from_pdb(_PDB)
    bad = ps.coords.at[0, 10, 0].set(jnp.nan)
    # default: does NOT raise (zero host-sync cost), NaN propagates visibly
    r = sf(ps, coords_override=bad)
    assert not bool(jnp.all(jnp.isfinite(r)))
    # opt-in guard: RAISES at the source rather than masking to 0
    with pytest.raises(ValueError, match="non-finite"):
        sf(ps, coords_override=bad, check_finite=True)


def test_minimizer_refuses_neighbor_list_sfxn():
    import jtmol
    from jtmol.optimization.minimizers import run_cart_min

    sf = jtmol.beta2016_score_function(use_neighbor_list=True)
    ps = jtmol.pose_stack_from_pdb(_PDB)
    with pytest.raises(ValueError, match="neighbor"):
        run_cart_min(ps, sf, max_iter=1)


if __name__ == "__main__":  # pragma: no cover
    test_bpl_cache_is_bounded_lru()
    test_check_finite_raises_not_masks()
    test_minimizer_refuses_neighbor_list_sfxn()
    print("PASS")
