"""Hessian normal-mode selection (`jtmol.analysis.hessian`).

Locks the mode-selection fix: B-factors / entropy must be built from the
**soft** low-frequency internal modes, not the stiff end. The bug this
guards: `stiff_modes`' ``λ > 1e-3·λ_max`` cutoff sits *above* every soft
mode (λ_max is a stiff bond stretch), so it keeps the wrong end and the
B-factor correlation collapses to the null (r≈-0.23 vs +0.68 on 1ubq).

Uses a synthetic Hessian with a known spectrum so the selection is
checked deterministically in milliseconds. The full 1ubq Cα B-factor
correlation lives in ``analysis.demo_hessian.ca_bfactor_benchmark``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from jtmol.analysis.hessian import (  # noqa: E402
    bfactors,
    soft_modes,
    stiff_modes,
    vibrational_entropy,
)

# Known spectrum: 2 negative (saddle), 6 near-zero rigid-body, 4 soft
# physical, 3 stiff. n=15 (divisible by 3 for the B-factor reshape).
_NEG = [-1.0, -0.05]
_RIGID = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6]
_SOFT = [0.1, 0.3, 0.7, 1.5]
_STIFF = [50.0, 200.0, 1000.0]
_SPECTRUM = np.array(_NEG + _RIGID + _SOFT + _STIFF)


def _synthetic_hessian() -> np.ndarray:
    """H = Q diag(spectrum) Qᵀ with a fixed (deterministic) rotation Q."""
    n = _SPECTRUM.size
    a = np.cos(np.outer(np.arange(1, n + 1), np.arange(1, n + 1)))  # full rank
    q, _ = np.linalg.qr(a)
    return (q * _SPECTRUM) @ q.T


def test_soft_modes_drops_rigid_and_negative() -> None:
    e, v = soft_modes(_synthetic_hessian())
    e = np.asarray(e)
    # 15 modes - 2 negative - 6 rigid-body = 7 kept (soft + stiff), soft-first.
    assert e.size == len(_SOFT) + len(_STIFF)
    assert np.all(e > 1e-8), "kept a negative / rigid mode"
    assert np.all(np.diff(e) >= 0), "not softest-first"
    assert np.allclose(np.sort(e), _SOFT + _STIFF, rtol=1e-6, atol=1e-6)
    assert np.asarray(v).shape == (15, 7)


def test_soft_modes_k_takes_softest() -> None:
    e, _ = soft_modes(_synthetic_hessian(), k=3)
    assert np.allclose(np.sort(np.asarray(e)), _SOFT[:3], rtol=1e-6, atol=1e-6)


def test_stiff_modes_keep_wrong_end() -> None:
    # The stiff cutoff λ>1e-3·λ_max = 1.0 keeps only λ>=1.0 and EXCLUDES the
    # soft modes that carry the B-factor signal: the exact bug.
    e, _ = stiff_modes(_synthetic_hessian())
    e = np.asarray(e)
    assert e.min() >= 1.0
    assert not np.any(e < 1.0), "stiff_modes leaked a soft mode"
    # the softest physical mode (0.1) must be absent
    assert not np.any(np.isclose(e, 0.1))


def test_bfactors_finite_and_positive() -> None:
    b = np.asarray(bfactors(_synthetic_hessian()))
    assert b.shape == (5,)  # M = 15 // 3
    assert np.all(np.isfinite(b)) and np.all(b > 0)


def test_soft_modes_dominate_bfactors() -> None:
    # The stiff modes (λ>=50) contribute ~1/λ ≈ 0 next to the soft modes'
    # 1/0.1: dropping them barely moves the B-factors, i.e. the soft end
    # carries essentially all the fluctuation signal.
    h = _synthetic_hessian()
    b_soft = np.asarray(bfactors(h, k=len(_SOFT))).mean()  # 4 soft only
    b_all = np.asarray(bfactors(h)).mean()  # soft + stiff
    assert abs(b_soft - b_all) / b_all < 0.05


def test_vibrational_entropy_finite() -> None:
    s = float(vibrational_entropy(_synthetic_hessian()))
    assert np.isfinite(s)


if __name__ == "__main__":  # pragma: no cover
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = []
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {exc!r}")
            bad.append(fn.__name__)
    sys.exit(1 if bad else 0)
