"""mosaic-compatible ``LossTerm`` base class for the extra scores.

Lives inside ``tmol_jax.score.extra`` so installed wheels don't fall
through to a NotImplementedError stub when ``examples/`` is not on
``sys.path``. The class signature matches
``mosaic.common.LossTerm`` exactly: ``__call__(sequence, output, key)
→ (scalar, aux_dict)`` with ``+`` and ``*`` composition.

If mosaic is installed we subclass its ``LossTerm`` directly so the
composition algebra works across mosaic-side and jtmol-side losses
without conversion. Otherwise we use this in-package shim, which
provides the same ``+`` / ``*`` / ``-`` / ``*-`` algebra.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax


try:
    from mosaic.common import LossTerm as _MosaicLossTerm

    LossTerm = _MosaicLossTerm
except Exception:  # pragma: no cover

    class LossTerm(eqx.Module):
        """In-package shim that matches mosaic's LossTerm protocol.

        Subclass and implement ``__call__(self, sequence, output,
        key) → (scalar_loss, aux_dict)``. The class implements ``+``
        and ``*`` so ``0.5 * loss_a + 1.0 * loss_b`` composes.
        """

        def __call__(self, *args, key, **kwargs):
            raise NotImplementedError(
                "LossTerm subclass must implement __call__"
            )

        def __rmul__(self, scalar):
            if not isinstance(scalar, (int, float)):
                return NotImplemented
            return _ScaledLoss(loss=self, scale=float(scalar))

        def __add__(self, other):
            if isinstance(other, LossTerm):
                other = 1.0 * other
            return _SumLoss(losses=[1.0 * self, other])

        def __sub__(self, other):
            return self + (-1.0) * other

        def __neg__(self):
            return (-1.0) * self

    class _ScaledLoss(LossTerm):
        loss: Any
        scale: float

        def __call__(self, *args, key, **kwargs):
            v, aux = self.loss(*args, key=key, **kwargs)
            return self.scale * v, aux

        def __rmul__(self, scalar):
            if not isinstance(scalar, (int, float)):
                return NotImplemented
            return _ScaledLoss(loss=self.loss, scale=self.scale * float(scalar))

    class _SumLoss(LossTerm):
        losses: list

        def __call__(self, *args, key, **kwargs):
            total = 0.0
            auxes = []
            for L in self.losses:
                v, a = L(*args, key=key, **kwargs)
                key = jax.random.fold_in(key, 1)
                total = total + v
                auxes.append(a)
            return total, auxes

        def __add__(self, other):
            if isinstance(other, LossTerm):
                other = 1.0 * other
            if isinstance(other, _SumLoss):
                return _SumLoss(losses=self.losses + other.losses)
            return _SumLoss(losses=self.losses + [other])


__all__ = ["LossTerm"]
