"""Extra interface scores beyond Rosetta's ``beta_nov2016_cart``.

These are differentiable JAX implementations of published protein-
interface scoring functions that empirically correlate with
experimental binding kinetics / affinity but are not part of the
canonical Rosetta energy function. We use them as additional
``LossTerm`` channels in ML design loops (mosaic / Boltz-2
hallucination) where they're cheap to evaluate and add orthogonal
gradient signal.

See ``docs/EXTRA_SCORES.md`` for the literature references and the
implementation notes per score.

Currently included:
- ``ss99`` — Selzer-Schreiber 1999 (PARE) electrostatic ``k_on`` proxy
- ``buh`` — Stranges-Kuhlman 2013 buried-unsat polars (``k_off`` proxy)
- ``prodigy`` — PRODIGY contact-based ΔG (``K_d`` baseline)
"""

from tmol_jax.score.extra.ss99 import ss99_energy, ss99_dG_complex
from tmol_jax.score.extra.buh import buh_score, buh_per_kA2
from tmol_jax.score.extra.prodigy import prodigy_dG, prodigy_contact_counts


__all__ = [
    "ss99_energy",
    "ss99_dG_complex",
    "buh_score",
    "buh_per_kA2",
    "prodigy_dG",
    "prodigy_contact_counts",
]
