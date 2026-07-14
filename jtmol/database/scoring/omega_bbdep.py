"""Backbone-dependent omega parameter database.

Same dual-file pattern as :mod:`.rama`: ``omega_bbdep.yaml`` carries
the residue -> table lookups. The binary tables are loaded from one
of, in order:

* ``omega_bbdep.npz`` -- pure numpy archive produced by
  ``scripts/convert_binary_tables.py``. This is the default
  runtime path and requires **no** torch.
* ``omega_bbdep.zip`` -- legacy ``torch.save`` pickle. Only used if
  the ``.npz`` is missing and ``torch`` is importable; the pickle
  references the upstream classes under
  ``tmol.database.scoring.omega_bbdep``, so we register lightweight
  attrs shims in :data:`sys.modules` before calling
  :func:`torch.load`.
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
class OmegaBBDepMappingParams:
    table_id: str
    res_middle: str
    res_upper: str = "_"
    invert_phi: bool = False
    invert_psi: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepTables:
    table_id: str
    mu: np.ndarray
    sigma: np.ndarray
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _OmegaBBDepLookupRaw:
    omega_bbdep_lookup: Tuple[OmegaBBDepMappingParams, ...]


def _tensor_to_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        return np.asarray(t.detach().cpu().numpy())
    return np.asarray(t)


# ------------------------------------------------------------------------- #
# Upstream-pickle shim                                                      #
# ------------------------------------------------------------------------- #


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _UpstreamOmegaBBDepMappingParams:
    table_id: str
    res_middle: str
    res_upper: str = "_"
    invert_phi: bool = False
    invert_psi: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _UpstreamOmegaBBDepTables:
    table_id: str
    mu: Any
    sigma: Any
    bbstep: Any
    bbstart: Any


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _UpstreamOmegaBBDepDatabase:
    uniq_id: str
    bbdep_omega_lookup: Any
    bbdep_omega_tables: Any


def _install_upstream_shim() -> None:
    """Register shim ``tmol.database.scoring.omega_bbdep`` in ``sys.modules``."""
    for mod_name in ("tmol", "tmol.database", "tmol.database.scoring"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    target = "tmol.database.scoring.omega_bbdep"
    if target in sys.modules:
        existing = sys.modules[target]
        for cls_name, shim in (
            ("OmegaBBDepDatabase", _UpstreamOmegaBBDepDatabase),
            ("OmegaBBDepTables", _UpstreamOmegaBBDepTables),
            ("OmegaBBDepMappingParams", _UpstreamOmegaBBDepMappingParams),
        ):
            if not hasattr(existing, cls_name):
                setattr(existing, cls_name, shim)
        return

    shim_mod = types.ModuleType(target)
    shim_mod.OmegaBBDepDatabase = _UpstreamOmegaBBDepDatabase
    shim_mod.OmegaBBDepTables = _UpstreamOmegaBBDepTables
    shim_mod.OmegaBBDepMappingParams = _UpstreamOmegaBBDepMappingParams
    sys.modules[target] = shim_mod


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepDatabase:
    uniq_id: str
    bbdep_omega_lookup: Tuple[OmegaBBDepMappingParams, ...]
    bbdep_omega_tables: Tuple[OmegaBBDepTables, ...] = ()

    @classmethod
    def from_file(
        cls,
        path: str,
        tables_zip: Optional[str] = None,
    ) -> "OmegaBBDepDatabase":
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        lookup_raw = structure(raw, _OmegaBBDepLookupRaw)

        tables: Tuple[OmegaBBDepTables, ...] = ()
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
            bbdep_omega_lookup=lookup_raw.omega_bbdep_lookup,
            bbdep_omega_tables=tables,
        )

    @staticmethod
    def load_tables_from_npz(npz_path: str) -> Tuple[OmegaBBDepTables, ...]:
        """Load omega_bbdep tables from a numpy ``.npz`` archive."""
        with np.load(npz_path, allow_pickle=False) as data:
            n = int(data["__n_tables"])
            out = []
            for i in range(n):
                out.append(
                    OmegaBBDepTables(
                        table_id=str(data[f"t{i}__table_id"]),
                        mu=np.asarray(data[f"t{i}__mu"], dtype=np.float32),
                        sigma=np.asarray(data[f"t{i}__sigma"], dtype=np.float32),
                        bbstep=tuple(float(x) for x in data[f"t{i}__bbstep"]),
                        bbstart=tuple(float(x) for x in data[f"t{i}__bbstart"]),
                    )
                )
        return tuple(out)

    @staticmethod
    def load_tables_from_zip(zip_path: str) -> Tuple[OmegaBBDepTables, ...]:
        """Load omega_bbdep tables from a ``torch.save``-produced zip.

        Fallback for when no ``omega_bbdep.npz`` is available.

        Raises
        ------
        ImportError
            If ``torch`` is not installed.
        ValueError
            If the payload is missing the expected ``bbdep_omega_tables``
            attribute.
        """
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Loading omega_bbdep.zip requires 'torch' (the file is a "
                "torch.save pickle). Either install torch "
                "(`pip install torch`) or run "
                "`python scripts/convert_binary_tables.py` once to "
                "generate an omega_bbdep.npz alongside it."
            ) from exc

        _install_upstream_shim()

        payload = torch.load(zip_path, map_location="cpu", weights_only=False)
        tables = getattr(payload, "bbdep_omega_tables", None)
        if tables is None:
            raise ValueError(
                f"omega_bbdep.zip payload missing 'bbdep_omega_tables': {zip_path}"
            )
        out = []
        for t in tables:
            out.append(
                OmegaBBDepTables(
                    table_id=t.table_id,
                    mu=_tensor_to_numpy(t.mu).astype(np.float32),
                    sigma=_tensor_to_numpy(t.sigma).astype(np.float32),
                    bbstep=tuple(float(x) for x in t.bbstep),
                    bbstart=tuple(float(x) for x in t.bbstart),
                )
            )
        return tuple(out)


__all__ = [
    "OmegaBBDepMappingParams",
    "OmegaBBDepTables",
    "OmegaBBDepDatabase",
]
