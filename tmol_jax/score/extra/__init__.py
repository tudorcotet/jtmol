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
- ``transcomp`` — Qin-Zhou 2011 transition-state ``k_on`` enhancement
- ``buh`` — Stranges-Kuhlman 2013 buried-unsat polars (``k_off`` proxy)
- ``prodigy`` — Vangone-Bonvin 2015 contact-based ΔG (``K_d`` baseline)
- ``dfire`` — Zhang-Zhou 2005 atom-pair statistical potential
  (placeholder coefficients only; the public Zhou-Yang table is not
  yet bundled)
- ``kinetics_tuning`` — composite three-channel loss exposing
  ``k_on``, ``k_off``, ``K_d`` as independent weight knobs
"""

from tmol_jax.score.extra.ss99 import ss99_energy, ss99_dG_complex
from tmol_jax.score.extra.buh import (
    buh_score, buh_per_kA2, hbond_satisfaction_from_coords,
)
from tmol_jax.score.extra.prodigy import prodigy_dG, prodigy_contact_counts
from tmol_jax.score.extra.transcomp import (
    transcomp_log_kon_enhancement, transcomp_kon_ratio,
)
from tmol_jax.score.extra.dfire import (
    dfire_energy, make_placeholder_dfire_table,
)


def _import_loss_term(name):
    """Lazy import for the LossTerm wrappers; avoids dragging
    examples/integration_mosaic_hallucinator into every import."""
    from tmol_jax.score.extra import losses
    return getattr(losses, name)


def __getattr__(name):
    if name in {"SS99Loss", "TransCompLoss", "BUHLoss",
                "PRODIGYLoss", "DFIRELoss"}:
        return _import_loss_term(name)
    raise AttributeError(name)


__all__ = [
    "ss99_energy",
    "ss99_dG_complex",
    "buh_score",
    "buh_per_kA2",
    "hbond_satisfaction_from_coords",
    "prodigy_dG",
    "prodigy_contact_counts",
    "transcomp_log_kon_enhancement",
    "transcomp_kon_ratio",
    "dfire_energy",
    "make_placeholder_dfire_table",
    # LossTerm wrappers (lazy):
    "SS99Loss",
    "TransCompLoss",
    "BUHLoss",
    "PRODIGYLoss",
    "DFIRELoss",
]
