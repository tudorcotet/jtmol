"""JAX-native port of :mod:`tmol.optimization`.

Re-exports the minimizer entry points and the lower-level L-BFGS /
line-search primitives. Downstream callers (e.g.
``tmol_jax.relax.fast_relax``) typically use :func:`run_min`,
:func:`run_cart_min`, or :func:`run_kin_min`; research / test code may
reach for :func:`lbfgs_minimize` and
:func:`strong_wolfe_line_search` directly.
"""

from .line_search import strong_wolfe_line_search
from .lbfgs import lbfgs_minimize
from .minimizers import (
    build_kinforest_network,
    run_cart_min,
    run_kin_min,
    run_min,
)


__all__ = [
    "strong_wolfe_line_search",
    "lbfgs_minimize",
    "build_kinforest_network",
    "run_cart_min",
    "run_kin_min",
    "run_min",
]
