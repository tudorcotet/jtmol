"""Ramachandran parameter database.

The upstream database ships a ``rama.zip`` that is a ``torch.save``
pickle of the full ``RamaDatabase`` (lookups + per-table energy grids).
To avoid a hard dependency on ``torch`` at runtime, this port prefers
a sibling ``rama.npz`` (produced by ``scripts/convert_binary_tables.py``)
and only falls back to torch if the ``.npz`` is missing:

* ``rama.yaml`` -- carries just the lookup rows (residue -> table name
  / prepro variant) and is always loaded via YAML into
  :class:`RamaMappingParams` tuples.
* ``rama.npz`` -- preferred on-disk format for the 2-D energy tables.
  Pure-numpy, no torch needed. Written by the conversion script.
* ``rama.zip`` -- legacy ``torch.save`` pickle. Read only if
  ``rama.npz`` is not present AND ``torch`` is importable; the zip
  pickle references the upstream classes under
  ``tmol.database.scoring.rama``, so we register lightweight shim
  classes in :data:`sys.modules` before calling :func:`torch.load`.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any, Optional, Tuple

import attr
import numpy as np
import yaml

from ._structure import structure


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaMappingParams:
    table_id: str
    res_middle: str
    res_upper: str = "_"
    invert_phi: bool = False
    invert_psi: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaTables:
    table_id: str
    table: np.ndarray  # 2-D energy grid over (phi, psi)
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _RamaLookupRaw:
    rama_lookup: Tuple[RamaMappingParams, ...]


def _tensor_to_numpy(t: Any) -> np.ndarray:
    """Best-effort conversion of a torch Tensor (or array-like) to numpy."""
    if hasattr(t, "detach"):
        return np.asarray(t.detach().cpu().numpy())
    return np.asarray(t)


# ------------------------------------------------------------------------- #
# Upstream-pickle shim                                                      #
# ------------------------------------------------------------------------- #
#
# The ``rama.zip`` payload was produced by ``torch.save`` on the upstream
# ``tmol.database.scoring.rama.{RamaDatabase, RamaTables, RamaMappingParams}``
# attrs classes. To unpickle them without importing the full upstream
# ``tmol`` package (which has its own set of dependencies) we register a
# synthetic module that provides field-compatible attrs shims. The pickle
# reconstructs these shim instances; we then translate them into the
# JAX-friendly dataclasses defined in this module (converting tensors to
# numpy arrays in the process).


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _UpstreamRamaMappingParams:
    table_id: str
    res_middle: str
    res_upper: str = "_"
    invert_phi: bool = False
    invert_psi: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _UpstreamRamaTables:
    table_id: str
    table: Any
    bbstep: Any
    bbstart: Any


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _UpstreamRamaDatabase:
    uniq_id: str
    rama_lookup: Any
    rama_tables: Any


def _install_upstream_shim() -> None:
    """Register shim ``tmol.database.scoring.rama`` in ``sys.modules``.

    Leaves any already-imported upstream modules untouched — we only
    fill in missing entries, so the shim never shadows a real install.
    """
    for mod_name in ("tmol", "tmol.database", "tmol.database.scoring"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    target = "tmol.database.scoring.rama"
    if target in sys.modules:
        existing = sys.modules[target]
        # Only patch the three classes if they're missing on the module.
        for cls_name, shim in (
            ("RamaDatabase", _UpstreamRamaDatabase),
            ("RamaTables", _UpstreamRamaTables),
            ("RamaMappingParams", _UpstreamRamaMappingParams),
        ):
            if not hasattr(existing, cls_name):
                setattr(existing, cls_name, shim)
        return

    shim_mod = types.ModuleType(target)
    shim_mod.RamaDatabase = _UpstreamRamaDatabase
    shim_mod.RamaTables = _UpstreamRamaTables
    shim_mod.RamaMappingParams = _UpstreamRamaMappingParams
    sys.modules[target] = shim_mod


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaDatabase:
    uniq_id: str  # unique id for memoization / cache keys
    rama_lookup: Tuple[RamaMappingParams, ...]
    rama_tables: Tuple[RamaTables, ...] = ()

    @classmethod
    def from_file(
        cls,
        path: str,
        tables_zip: Optional[str] = None,
    ) -> "RamaDatabase":
        """Load from a ``rama.yaml`` file.

        Resolution order for the binary energy tables:

        1. A sibling ``rama.npz`` (pure numpy; no torch required).
        2. A sibling ``rama.zip`` (or an explicit ``tables_zip`` path)
           loaded via ``torch`` -- only if torch is importable.

        If neither exists, ``rama_tables`` is left empty.
        """
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        lookup_raw = structure(raw, _RamaLookupRaw)

        tables: Tuple[RamaTables, ...] = ()
        stem = os.path.splitext(path)[0]
        npz_guess = stem + ".npz"
        zip_path = tables_zip
        if zip_path is None and os.path.exists(npz_guess):
            tables = cls.load_tables_from_npz(npz_guess)
        else:
            if zip_path is None:
                guess = stem + ".zip"
                if os.path.exists(guess):
                    zip_path = guess
            if zip_path is not None:
                tables = cls.load_tables_from_zip(zip_path)

        return cls(
            uniq_id=os.path.abspath(path),
            rama_lookup=lookup_raw.rama_lookup,
            rama_tables=tables,
        )

    @staticmethod
    def load_tables_from_npz(npz_path: str) -> Tuple[RamaTables, ...]:
        """Load rama energy tables from a numpy ``.npz`` archive.

        Uses the flat per-table layout written by
        ``scripts/convert_binary_tables.py``.
        """
        with np.load(npz_path, allow_pickle=False) as data:
            n = int(data["__n_tables"])
            out = []
            for i in range(n):
                out.append(
                    RamaTables(
                        table_id=str(data[f"t{i}__table_id"]),
                        table=np.asarray(data[f"t{i}__table"], dtype=np.float32),
                        bbstep=tuple(float(x) for x in data[f"t{i}__bbstep"]),
                        bbstart=tuple(float(x) for x in data[f"t{i}__bbstart"]),
                    )
                )
        return tuple(out)

    @staticmethod
    def load_tables_from_zip(zip_path: str) -> Tuple[RamaTables, ...]:
        """Load rama energy tables from a ``torch.save``-produced zip.

        This fallback requires ``torch``. Prefer the ``.npz`` written
        by ``scripts/convert_binary_tables.py`` -- that path needs only
        numpy.

        Raises
        ------
        ImportError
            If ``torch`` is not installed.
        ValueError
            If the pickle payload does not carry the expected
            ``rama_tables`` attribute.
        """
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Loading rama.zip requires 'torch' (the file is a "
                "torch.save pickle). Either install torch "
                "(`pip install torch`) or run "
                "`python scripts/convert_binary_tables.py` once to "
                "generate a rama.npz alongside it."
            ) from exc

        _install_upstream_shim()

        payload = torch.load(zip_path, map_location="cpu", weights_only=False)
        rama_tables = getattr(payload, "rama_tables", None)
        if rama_tables is None:
            raise ValueError(f"rama.zip payload missing 'rama_tables': {zip_path}")
        out = []
        for t in rama_tables:
            out.append(
                RamaTables(
                    table_id=t.table_id,
                    table=_tensor_to_numpy(t.table).astype(np.float32),
                    bbstep=tuple(float(x) for x in t.bbstep),
                    bbstart=tuple(float(x) for x in t.bbstart),
                )
            )
        return tuple(out)


__all__ = ["RamaMappingParams", "RamaTables", "RamaDatabase"]
