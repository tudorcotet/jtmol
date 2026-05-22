"""PDB file parsing utilities for tmol_jax.

Ported from ``tmol.io.pdb_parsing``. The upstream implementation returned
a ``pandas.DataFrame``; here we return a plain dictionary of ``numpy``
arrays. The dict has the same per-atom columns the rest of the io
pipeline needs: ``atom_name``, ``res_name``, ``chain_id``, ``res_num``,
``coords``, plus a few extras (``record_name``, ``alt_loc``, ``insert``,
``occupancy``, ``b_factor``, ``model``) that downstream consumers may
want but don't rely on.

Two parsing backends are supported:

1. **biotite** -- if ``biotite.structure.io.pdb`` is importable, it is
   used as the primary parser. Biotite is a hard dependency of tmol_jax
   (see ``pyproject.toml``), so this is the expected path.
2. **Direct line parser** -- a compact fallback that reads the
   fixed-column PDB ATOM/HETATM records directly. Useful if biotite is
   not installed at runtime for whatever reason.

The public entry point :func:`parse_pdb` accepts any of:
  * a list / tuple / iterable of lines,
  * a single string containing an entire PDB file,
  * a filesystem path (file-like string that resolves to an existing
    file).

``parse_pdb`` always returns a ``dict`` with plain numpy arrays. The
companion :func:`pose_stack_from_pdb` threads PDB -> canonical-form ->
:class:`PoseStack` (delegating to the other io submodules for the heavy
lifting).

Pure Python + numpy -- no jax, no torch.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, List, Union

import numpy as np


logger = logging.getLogger(__name__)


PDBInput = Union[str, Iterable[str], List[str]]


# --------------------------------------------------------------------- #
# Backend detection                                                      #
# --------------------------------------------------------------------- #


def _try_import_biotite_pdb():
    """Return the biotite PDB module, or ``None`` if unavailable."""
    try:  # pragma: no cover - import guard
        import biotite.structure.io.pdb as _biotite_pdb  # noqa: F401

        return _biotite_pdb
    except Exception:  # pragma: no cover
        return None


# --------------------------------------------------------------------- #
# Input normalisation                                                    #
# --------------------------------------------------------------------- #


def _normalize_pdb_input(pdb_input: PDBInput) -> str:
    """Normalise ``pdb_input`` into a single string of PDB text.

    Accepts:
      * a path to an existing file (str),
      * a string of PDB text with embedded newlines,
      * an iterable/list of lines.
    """
    if isinstance(pdb_input, (list, tuple)):
        return "".join(
            line if line.endswith("\n") else line + "\n" for line in pdb_input
        )
    if isinstance(pdb_input, str):
        if os.path.exists(pdb_input) and os.path.isfile(pdb_input):
            with open(pdb_input, "r") as f:
                return f.read()
        return pdb_input
    if hasattr(pdb_input, "__iter__"):
        return "".join(
            line if line.endswith("\n") else line + "\n" for line in pdb_input
        )
    raise TypeError(
        f"parse_pdb: cannot handle input of type {type(pdb_input).__name__}"
    )


# --------------------------------------------------------------------- #
# Direct line-by-line parser (fallback)                                  #
# --------------------------------------------------------------------- #


def _parse_pdb_direct(pdb_text: str) -> dict:
    """Parse ATOM/HETATM records from PDB text via fixed-column slicing.

    Columns are 1-based per PDB spec; Python slicing below is 0-based.

     1- 6   record name     "ATOM  " / "HETATM"
     7-11   atom serial
    13-16   atom name
    17      alt loc
    18-20   residue name
    22      chain id
    23-26   residue seq num
    27      insertion code
    31-38   x
    39-46   y
    47-54   z
    55-60   occupancy
    61-66   temperature factor (b)
    """
    record_names: List[str] = []
    atom_serials: List[int] = []
    atom_names: List[str] = []
    alt_locs: List[str] = []
    res_names: List[str] = []
    chain_ids: List[str] = []
    res_nums: List[int] = []
    insert_codes: List[str] = []
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    occupancies: List[float] = []
    b_factors: List[float] = []
    model_ids: List[int] = []

    current_model = 0

    for raw in pdb_text.splitlines():
        # Preserve column layout; pad so slicing never IndexErrors.
        line = raw
        if len(line) < 80:
            line = line + " " * (80 - len(line))

        tag = line[:6]
        if tag == "MODEL ":
            # Track the model index for readers that care.
            try:
                current_model = int(line[6:].strip() or "0")
            except ValueError:
                current_model += 1
            continue
        if tag == "ENDMDL":
            continue
        if tag.startswith("TER"):
            # TER records don't carry atom data.
            continue
        if tag != "ATOM  " and tag != "HETATM":
            continue

        record_names.append(tag.strip())
        try:
            atom_serials.append(int(line[6:11].strip() or "0"))
        except ValueError:
            atom_serials.append(0)
        atom_names.append(line[12:16].strip())
        alt_locs.append(line[16:17].strip())
        res_names.append(line[17:20].strip())
        chain_ids.append(line[21:22].strip())
        try:
            res_nums.append(int(line[22:26].strip() or "0"))
        except ValueError:
            res_nums.append(0)
        insert_codes.append(line[26:27].strip())
        try:
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
        except ValueError:
            xs.append(float("nan"))
            ys.append(float("nan"))
            zs.append(float("nan"))
        try:
            occupancies.append(float(line[54:60].strip() or "1.0"))
        except ValueError:
            occupancies.append(1.0)
        try:
            b_factors.append(float(line[60:66].strip() or "0.0"))
        except ValueError:
            b_factors.append(0.0)
        model_ids.append(current_model)

    n = len(atom_names)
    coords = np.zeros((n, 3), dtype=np.float32)
    if n > 0:
        coords[:, 0] = np.asarray(xs, dtype=np.float32)
        coords[:, 1] = np.asarray(ys, dtype=np.float32)
        coords[:, 2] = np.asarray(zs, dtype=np.float32)

    return {
        "record_name": np.asarray(record_names, dtype=object),
        "atom_serial": np.asarray(atom_serials, dtype=np.int32),
        "atom_name": np.asarray(atom_names, dtype=object),
        "alt_loc": np.asarray(alt_locs, dtype=object),
        "res_name": np.asarray(res_names, dtype=object),
        "chain_id": np.asarray(chain_ids, dtype=object),
        "res_num": np.asarray(res_nums, dtype=np.int32),
        "insert": np.asarray(insert_codes, dtype=object),
        "coords": coords,
        "occupancy": np.asarray(occupancies, dtype=np.float32),
        "b_factor": np.asarray(b_factors, dtype=np.float32),
        "model": np.asarray(model_ids, dtype=np.int32),
    }


# --------------------------------------------------------------------- #
# Biotite parser                                                         #
# --------------------------------------------------------------------- #


def _parse_pdb_biotite(pdb_text: str) -> dict:
    """Parse PDB text via :mod:`biotite.structure.io.pdb`.

    Falls back to the direct parser if biotite is not importable.
    """
    mod = _try_import_biotite_pdb()
    if mod is None:  # pragma: no cover
        return _parse_pdb_direct(pdb_text)

    import io as _io

    pdb_file = mod.PDBFile.read(_io.StringIO(pdb_text))
    # Use the *last* model. Upstream ``tmol.io.pdb_parsing.parse_pdb``
    # processes every ATOM record across all MODEL blocks into the
    # same residue-keyed canonical-form slots, so the last splat wins;
    # that is equivalent to reading the last model directly.
    try:
        atoms_all = pdb_file.get_structure(
            extra_fields=["occupancy", "b_factor"]
        )
    except Exception:
        atoms_all = pdb_file.get_structure()
    # biotite returns an AtomArrayStack when the file has multiple
    # models. Take the last model (matching upstream semantics); when
    # there is only one model, the return is a bare AtomArray already.
    import biotite.structure as _bio_struct

    if isinstance(atoms_all, _bio_struct.AtomArrayStack):
        atoms = atoms_all[-1]
    else:
        atoms = atoms_all

    # Biotite hands us numpy arrays already.
    atom_name = np.asarray(atoms.atom_name, dtype=object)
    res_name = np.asarray(atoms.res_name, dtype=object)
    chain_id = np.asarray(atoms.chain_id, dtype=object)
    res_num = np.asarray(atoms.res_id, dtype=np.int32)
    insert = np.asarray(
        getattr(atoms, "ins_code", np.array([""] * len(atoms))), dtype=object
    )
    coords = np.asarray(atoms.coord, dtype=np.float32)
    occ = np.asarray(
        getattr(atoms, "occupancy", np.ones(len(atoms), dtype=np.float32)),
        dtype=np.float32,
    )
    bfac = np.asarray(
        getattr(atoms, "b_factor", np.zeros(len(atoms), dtype=np.float32)),
        dtype=np.float32,
    )

    # biotite stores hetero flag as a bool ``hetero`` annotation array.
    hetero = getattr(atoms, "hetero", np.zeros(len(atoms), dtype=bool))
    record_name = np.where(
        np.asarray(hetero, dtype=bool), "HETATM", "ATOM"
    ).astype(object)

    return {
        "record_name": record_name,
        "atom_serial": np.arange(len(atoms), dtype=np.int32),
        "atom_name": atom_name,
        "alt_loc": np.asarray(
            getattr(atoms, "altloc_id", np.array([""] * len(atoms))), dtype=object
        ),
        "res_name": res_name,
        "chain_id": chain_id,
        "res_num": res_num,
        "insert": insert,
        "coords": coords,
        "occupancy": occ,
        "b_factor": bfac,
        "model": np.zeros(len(atoms), dtype=np.int32),
    }


# --------------------------------------------------------------------- #
# Public API                                                             #
# --------------------------------------------------------------------- #


def parse_pdb(pdb_input: PDBInput, *, backend: str = "auto") -> dict:
    """Parse ATOM/HETATM records from a PDB file, string, or iterable of lines.

    Parameters
    ----------
    pdb_input:
        One of:
          * a path to an existing PDB file,
          * a string of PDB text with embedded newlines,
          * an iterable of lines.
    backend:
        ``"auto"`` (default) picks biotite if available, else the direct
        line parser. ``"biotite"`` forces biotite (raising if it's not
        installed); ``"direct"`` forces the pure-Python parser.

    Returns
    -------
    dict
        A dictionary of per-atom numpy arrays with keys
        ``"atom_name"``, ``"res_name"``, ``"chain_id"``, ``"res_num"``,
        ``"coords"`` (shape ``[n_atoms, 3]``), ``"insert"``, ``"alt_loc"``,
        ``"occupancy"``, ``"b_factor"``, ``"record_name"``,
        ``"atom_serial"``, and ``"model"``.
    """
    text = _normalize_pdb_input(pdb_input)

    if backend == "biotite":
        mod = _try_import_biotite_pdb()
        if mod is None:
            raise ImportError(
                "parse_pdb(backend='biotite') requested but biotite is not installed"
            )
        return _parse_pdb_biotite(text)
    if backend == "direct":
        return _parse_pdb_direct(text)
    if backend != "auto":
        raise ValueError(f"parse_pdb: unknown backend {backend!r}")

    if _try_import_biotite_pdb() is not None:
        try:
            return _parse_pdb_biotite(text)
        except Exception as exc:
            logger.warning(
                "biotite PDB parse failed (%s); falling back to direct parser", exc
            )
    return _parse_pdb_direct(text)


# --------------------------------------------------------------------- #
# PDB -> PoseStack convenience                                           #
# --------------------------------------------------------------------- #


def pose_stack_from_pdb(
    pdb_input: PDBInput,
    *,
    canonical_ordering: Any = None,
    packed_block_types: Any = None,
    **kwargs: Any,
):
    """Top-level helper: parse a PDB and build a :class:`PoseStack`.

    This threads together :func:`parse_pdb`,
    :func:`tmol_jax.io.canonical_form.canonical_form_from_pdb`, the
    default ``PackedBlockTypes``, and the companion
    :func:`tmol_jax.io.pose_stack_construction.pose_stack_from_canonical_form`
    (authored by a parallel agent).

    Parameters
    ----------
    pdb_input:
        Path / text / line-iterable for a PDB file; see :func:`parse_pdb`.
    canonical_ordering:
        Optional :class:`CanonicalOrdering`. Defaults to
        :func:`default_canonical_ordering`.
    packed_block_types:
        Optional :class:`PackedBlockTypes`. Defaults to
        :func:`default_packed_block_types`.
    **kwargs:
        Forwarded to :func:`pose_stack_from_canonical_form`.

    Returns
    -------
    PoseStack
        The reconstructed pose stack. When the sibling pose_stack
        construction module is not yet available, a clear
        :class:`NotImplementedError` is raised so callers see the gap.
    """
    # Lazy imports to avoid circular loads and tolerate partial ports.
    from tmol_jax.io.canonical_ordering import (
        default_canonical_ordering,
        default_packed_block_types,
    )
    from tmol_jax.io.canonical_form import canonical_form_from_pdb

    co = canonical_ordering if canonical_ordering is not None else (
        default_canonical_ordering()
    )
    pbt = packed_block_types if packed_block_types is not None else (
        default_packed_block_types()
    )

    cf = canonical_form_from_pdb(pdb_input, co)

    try:
        from tmol_jax.io.pose_stack_construction import (  # type: ignore
            pose_stack_from_canonical_form,
        )
    except Exception as exc:  # pragma: no cover - parallel-port gap
        raise NotImplementedError(
            "pose_stack_from_pdb: tmol_jax.io.pose_stack_construction is "
            "not yet available; parsed the PDB into a canonical-form dict "
            "but cannot build a PoseStack. Call canonical_form_from_pdb "
            "directly if you only need the canonical-form arrays."
        ) from exc

    return pose_stack_from_canonical_form(cf, pbt, **kwargs)


__all__ = [
    "parse_pdb",
    "pose_stack_from_pdb",
]
