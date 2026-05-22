"""Dunbrack rotamer-library parameter database.

Upstream :mod:`tmol.database.scoring.dunbrack_libraries` loads the
entire library from ``dunbrack.bin`` -- a ``torch.save`` pickle that
contains attrs-decorated classes referencing ``torch.Tensor`` fields.

This port avoids a runtime dependency on ``torch`` by preferring a
sibling ``dunbrack.npz`` (pure numpy, produced once via
``scripts/convert_binary_tables.py``) and only falling back to
``torch.load`` of ``dunbrack.bin`` if the ``.npz`` is missing:

* ``dunbrack.yaml`` -- carries the ``dunbrack_lookup`` rows (residue
  name -> library table name). Always loaded via YAML.
* ``dunbrack.npz`` -- preferred runtime format. Requires only numpy.
* ``dunbrack.bin`` -- legacy torch pickle. Loaded via ``torch.load``
  when torch is importable; the upstream pickle references the
  ``tmol.database.scoring.dunbrack_libraries`` module for the attrs
  classes, so we install a stub alias that redirects those names to
  this module's shim classes before unpickling.
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


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunMappingParams:
    dun_table_name: str
    residue_name: str


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericDataForAA:
    rotamers: np.ndarray  # nrotamers x nchi
    rotamer_probabilities: np.ndarray  # (1 + n-bb) dimensional
    rotamer_means: np.ndarray  # nrots x [nbb] x nchi
    rotamer_stdvs: np.ndarray  # nrots x [nbb] x nchi
    prob_sorted_rot_inds: np.ndarray
    backbone_dihedral_start: np.ndarray
    backbone_dihedral_step: np.ndarray
    rotamer_alias: np.ndarray

    def nrotamers(self) -> int:
        return int(self.rotamers.shape[0])

    def nchi(self) -> int:
        return int(self.rotamers.shape[1])


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA
    non_rot_chi_start: float
    non_rot_chi_step: float
    non_rot_chi_period: float
    rotameric_chi_rotamers: np.ndarray  # nrots x n-rotameric-chi
    nonrotameric_chi_probabilities: np.ndarray  # (1+nbb+1)-dim table
    rotamer_boundaries: np.ndarray  # (:, 2): 0=left, 1=right


@attr.s(auto_attribs=True, slots=True, frozen=True)
class _DunbrackLookupRaw:
    dunbrack_lookup: Tuple[DunMappingParams, ...]


def _tensor_to_numpy(t: Any) -> np.ndarray:
    """Convert a torch.Tensor (or anything array-like) to a plain numpy array.

    The tensor is detached, moved to CPU, and deep-copied so the
    resulting numpy array owns its storage (the raw storage from
    ``torch.load`` is otherwise shared with the unpickler buffer and
    would be freed when the torch objects are garbage-collected).
    """
    if hasattr(t, "detach"):
        return np.asarray(t.detach().cpu().numpy()).copy()
    return np.asarray(t).copy()


def _convert_rotameric_data(data: Any) -> RotamericDataForAA:
    return RotamericDataForAA(
        rotamers=_tensor_to_numpy(data.rotamers),
        rotamer_probabilities=_tensor_to_numpy(data.rotamer_probabilities),
        rotamer_means=_tensor_to_numpy(data.rotamer_means),
        rotamer_stdvs=_tensor_to_numpy(data.rotamer_stdvs),
        prob_sorted_rot_inds=_tensor_to_numpy(data.prob_sorted_rot_inds),
        backbone_dihedral_start=_tensor_to_numpy(data.backbone_dihedral_start),
        backbone_dihedral_step=_tensor_to_numpy(data.backbone_dihedral_step),
        rotamer_alias=_tensor_to_numpy(data.rotamer_alias),
    )


# --------------------------------------------------------------------- #
# Upstream-module stubbing                                              #
# --------------------------------------------------------------------- #
#
# The upstream pickle references ``tmol.database.scoring.dunbrack_libraries``
# (the upstream module path) for the attrs classes it instantiates. We
# redirect that import path to *our* classes so ``torch.load`` can
# reconstruct the hierarchy without needing the original ``tmol`` package
# on sys.path. Each attrs class below has an identical field layout to
# the upstream one, so the unpickled objects will populate correctly.


def _install_upstream_stub() -> None:
    """Register a ``tmol.database.scoring.dunbrack_libraries`` alias.

    Idempotent: will not clobber an existing entry (e.g. if the real
    upstream ``tmol`` package is also on sys.path) except to ensure the
    expected attrs class names resolve to this module's definitions,
    which keeps instance-building consistent.
    """
    tmol_mod = sys.modules.get("tmol")
    if tmol_mod is None:
        tmol_mod = types.ModuleType("tmol")
        sys.modules["tmol"] = tmol_mod
    db_mod = sys.modules.get("tmol.database")
    if db_mod is None:
        db_mod = types.ModuleType("tmol.database")
        sys.modules["tmol.database"] = db_mod
    scoring_mod = sys.modules.get("tmol.database.scoring")
    if scoring_mod is None:
        scoring_mod = types.ModuleType("tmol.database.scoring")
        sys.modules["tmol.database.scoring"] = scoring_mod
    dun_mod = sys.modules.get("tmol.database.scoring.dunbrack_libraries")
    if dun_mod is None:
        dun_mod = types.ModuleType("tmol.database.scoring.dunbrack_libraries")
        sys.modules["tmol.database.scoring.dunbrack_libraries"] = dun_mod
    # Always re-bind the attrs names: in case the real upstream module
    # is imported, we still want the unpickler to get classes with the
    # numpy-friendly field layout below (identical field names, so the
    # upstream pickle keyword-arg BUILD step works either way).
    for _name in (
        "DunbrackRotamerLibrary",
        "DunMappingParams",
        "RotamericAADunbrackLibrary",
        "RotamericDataForAA",
        "SemiRotamericAADunbrackLibrary",
    ):
        setattr(dun_mod, _name, globals()[f"_Upstream{_name}"])


# --------------------------------------------------------------------- #
# Upstream-shaped stand-ins                                             #
# --------------------------------------------------------------------- #
#
# The pickle calls ``cls()`` (EMPTY_TUPLE + NEWOBJ) and then BUILD with
# a dict of field values. Our public attrs classes declare required
# fields, which would fail under the argumentless constructor the
# upstream pickle expects. The stubs below mirror the upstream field
# layout but with *all-optional* fields so the BUILD step can populate
# them incrementally. After unpickling we hand these to
# ``_normalise_payload`` which copies them into the public attrs
# containers (with the tensor->numpy conversion folded in).


class _UpstreamBase:
    """Slots-less base for upstream pickle reconstruction.

    Exposes ``__setstate__`` (called by pickle's BUILD) so the
    unpickler can set every field from a plain dict. Using a non-attrs
    class here avoids the keyword-arg mismatch between upstream
    torch-tensor fields and our numpy-backed public attrs classes.
    """

    def __setstate__(self, state: dict) -> None:
        if isinstance(state, dict):
            for k, v in state.items():
                setattr(self, k, v)
        else:
            self.__dict__.update(state.__dict__)


class _UpstreamDunMappingParams(_UpstreamBase):
    pass


class _UpstreamRotamericDataForAA(_UpstreamBase):
    pass


class _UpstreamRotamericAADunbrackLibrary(_UpstreamBase):
    pass


class _UpstreamSemiRotamericAADunbrackLibrary(_UpstreamBase):
    pass


class _UpstreamDunbrackRotamerLibrary(_UpstreamBase):
    pass


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunbrackRotamerLibrary:
    dun_lookup: Tuple[DunMappingParams, ...]
    rotameric_libraries: Tuple[RotamericAADunbrackLibrary, ...] = ()
    semi_rotameric_libraries: Tuple[SemiRotamericAADunbrackLibrary, ...] = ()

    @classmethod
    def from_file(
        cls,
        path: str,
        tables_bin: Optional[str] = None,
    ) -> "DunbrackRotamerLibrary":
        """Load from a YAML lookup file.

        The YAML at ``path`` carries the residue -> library-name lookup
        and is always required. Resolution order for the numerical
        library tables:

        1. A sibling ``dunbrack.npz`` (pure numpy; no torch needed).
        2. A sibling ``dunbrack.bin`` (or an explicit ``tables_bin``
           path) read via ``torch.load`` -- only if torch is importable.

        If none is present, the rotameric / semi-rotameric library
        tuples are left empty.
        """
        with open(path, "r") as infile:
            raw = yaml.safe_load(infile)
        lookup_raw = structure(raw, _DunbrackLookupRaw)

        rot_libs: Tuple[RotamericAADunbrackLibrary, ...] = ()
        semi_libs: Tuple[SemiRotamericAADunbrackLibrary, ...] = ()

        stem = os.path.splitext(path)[0]
        npz_guess = stem + ".npz"
        bin_path = tables_bin
        if bin_path is None and os.path.exists(npz_guess):
            rot_libs, semi_libs = cls.load_libraries_from_npz(npz_guess)
        else:
            if bin_path is None:
                guess = stem + ".bin"
                if os.path.exists(guess):
                    bin_path = guess
            if bin_path is not None:
                rot_libs, semi_libs = cls.load_libraries_from_bin(bin_path)

        return cls(
            dun_lookup=lookup_raw.dunbrack_lookup,
            rotameric_libraries=rot_libs,
            semi_rotameric_libraries=semi_libs,
        )

    @staticmethod
    def load_libraries_from_npz(
        npz_path: str,
    ) -> Tuple[
        Tuple[RotamericAADunbrackLibrary, ...],
        Tuple[SemiRotamericAADunbrackLibrary, ...],
    ]:
        """Load the Dunbrack rotamer libraries from a numpy ``.npz``.

        The archive layout matches the one written by
        ``scripts/convert_binary_tables.py`` -- flat arrays keyed by
        ``rot{i}__...`` / ``semi{i}__...`` prefixes, plus ``__n_rot``
        and ``__n_semi`` counters.
        """
        rot_data_fields = (
            "rotamers",
            "rotamer_probabilities",
            "rotamer_means",
            "rotamer_stdvs",
            "prob_sorted_rot_inds",
            "backbone_dihedral_start",
            "backbone_dihedral_step",
            "rotamer_alias",
        )

        def _load_rot_data(data, prefix: str) -> RotamericDataForAA:
            return RotamericDataForAA(
                **{
                    field: np.asarray(data[prefix + "data__" + field]).copy()
                    for field in rot_data_fields
                }
            )

        with np.load(npz_path, allow_pickle=False) as data:
            n_rot = int(data["__n_rot"])
            n_semi = int(data["__n_semi"])

            rot_libs_out = []
            for i in range(n_rot):
                prefix = f"rot{i}__"
                rot_libs_out.append(
                    RotamericAADunbrackLibrary(
                        table_name=str(data[prefix + "table_name"]),
                        rotameric_data=_load_rot_data(data, prefix),
                    )
                )

            semi_libs_out = []
            for i in range(n_semi):
                prefix = f"semi{i}__"
                semi_libs_out.append(
                    SemiRotamericAADunbrackLibrary(
                        table_name=str(data[prefix + "table_name"]),
                        rotameric_data=_load_rot_data(data, prefix),
                        non_rot_chi_start=float(data[prefix + "non_rot_chi_start"]),
                        non_rot_chi_step=float(data[prefix + "non_rot_chi_step"]),
                        non_rot_chi_period=float(
                            data[prefix + "non_rot_chi_period"]
                        ),
                        rotameric_chi_rotamers=np.asarray(
                            data[prefix + "rotameric_chi_rotamers"]
                        ).copy(),
                        nonrotameric_chi_probabilities=np.asarray(
                            data[prefix + "nonrotameric_chi_probabilities"]
                        ).copy(),
                        rotamer_boundaries=np.asarray(
                            data[prefix + "rotamer_boundaries"]
                        ).copy(),
                    )
                )

        return tuple(rot_libs_out), tuple(semi_libs_out)

    @staticmethod
    def load_libraries_from_bin(
        bin_path: str,
    ) -> Tuple[
        Tuple[RotamericAADunbrackLibrary, ...],
        Tuple[SemiRotamericAADunbrackLibrary, ...],
    ]:
        """Load the binary Dunbrack rotamer libraries via ``torch``.

        Fallback for when no ``dunbrack.npz`` is available. Installs
        stub aliases at ``tmol.database.scoring.dunbrack_libraries``
        so the pickle rebuilds against our local classes, runs
        ``torch.load(path, map_location='cpu')``, then eagerly converts
        every tensor to numpy via ``.detach().cpu().numpy().copy()``.
        """
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Loading dunbrack.bin requires the 'torch' package. "
                "Either install torch (`pip install torch`), supply a "
                "different tables_bin path, or run "
                "`python scripts/convert_binary_tables.py` once to "
                "generate a dunbrack.npz alongside it."
            ) from exc

        _install_upstream_stub()

        # ``weights_only=False`` is required because the pickle contains
        # attrs dataclasses (not just tensor state). We re-pickle
        # through our stub classes so arbitrary-callable hazards are
        # restricted to the upstream-authored module-path above.
        payload = torch.load(bin_path, map_location="cpu", weights_only=False)

        rot_libs_out = []
        for lib in getattr(payload, "rotameric_libraries", ()) or ():
            rot_libs_out.append(
                RotamericAADunbrackLibrary(
                    table_name=str(lib.table_name),
                    rotameric_data=_convert_rotameric_data(lib.rotameric_data),
                )
            )
        semi_libs_out = []
        for lib in getattr(payload, "semi_rotameric_libraries", ()) or ():
            semi_libs_out.append(
                SemiRotamericAADunbrackLibrary(
                    table_name=str(lib.table_name),
                    rotameric_data=_convert_rotameric_data(lib.rotameric_data),
                    non_rot_chi_start=float(lib.non_rot_chi_start),
                    non_rot_chi_step=float(lib.non_rot_chi_step),
                    non_rot_chi_period=float(lib.non_rot_chi_period),
                    rotameric_chi_rotamers=_tensor_to_numpy(
                        lib.rotameric_chi_rotamers
                    ),
                    nonrotameric_chi_probabilities=_tensor_to_numpy(
                        lib.nonrotameric_chi_probabilities
                    ),
                    rotamer_boundaries=_tensor_to_numpy(lib.rotamer_boundaries),
                )
            )
        return tuple(rot_libs_out), tuple(semi_libs_out)


__all__ = [
    "DunMappingParams",
    "RotamericDataForAA",
    "RotamericAADunbrackLibrary",
    "SemiRotamericAADunbrackLibrary",
    "DunbrackRotamerLibrary",
]
