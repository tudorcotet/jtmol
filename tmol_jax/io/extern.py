"""External-library integration utilities for ``tmol_jax.io``.

Ported from ``tmol.io.extern``. Upstream only exposed a single
:func:`fetch_pdb` that pulled a structure from the RCSB by ID; we keep
that and add a handful of small helpers that the openfold / rosettafold2
/ biotite bridges all want:

* :func:`fetch_pdb` -- download a PDB file from the RCSB by 4-letter code
  (returns the raw text).
* :func:`fetch_pdb_to_file` -- same, but write the bytes to a given path.
* :func:`pose_stack_from_pdb_id` -- convenience wrapper that fetches the
  PDB and routes it through :func:`tmol_jax.io.pose_stack_from_pdb`.

``requests`` is used where available; if it isn't importable we fall back
to :mod:`urllib.request` so the function still works on a minimal install.
No torch, no jax.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional


logger = logging.getLogger(__name__)


RCSB_PDB_URL = "https://files.rcsb.org/download/%s.pdb"


# --------------------------------------------------------------------- #
# HTTP helpers                                                           #
# --------------------------------------------------------------------- #


def _http_get(url: str, timeout: float = 30.0) -> str:
    """Fetch ``url`` and return the response body as text.

    Tries :mod:`requests` first, falling back to :mod:`urllib.request` if
    requests is not installed. Raises ``RuntimeError`` on non-2xx
    statuses so callers can tell a failed download from an empty file.
    """
    try:  # pragma: no cover - import guard
        import requests  # type: ignore

        resp = requests.get(url, timeout=timeout)
        if not resp.ok:
            raise RuntimeError(
                f"HTTP GET {url} failed with status {resp.status_code}"
            )
        return resp.text
    except ImportError:  # pragma: no cover - fallback path
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as fh:
            status = getattr(fh, "status", 200)
            if status and int(status) >= 400:
                raise RuntimeError(f"HTTP GET {url} failed with status {status}")
            data = fh.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return data


# --------------------------------------------------------------------- #
# Public API                                                             #
# --------------------------------------------------------------------- #


def fetch_pdb(pdbid: str, timeout: float = 30.0) -> str:
    """Download a PDB file from the RCSB by 4-letter code.

    Parameters
    ----------
    pdbid:
        PDB identifier (case-insensitive), e.g. ``"1ubq"``.
    timeout:
        HTTP timeout in seconds. Default 30.

    Returns
    -------
    str
        The full PDB file as text.
    """
    pdbid = str(pdbid).strip()
    if len(pdbid) != 4:
        logger.warning(
            "fetch_pdb: pdbid %r is not 4 characters; RCSB expects 4-letter codes",
            pdbid,
        )
    url = RCSB_PDB_URL % pdbid.upper()
    return _http_get(url, timeout=timeout)


def fetch_pdb_to_file(pdbid: str, out_path: str, timeout: float = 30.0) -> str:
    """Fetch ``pdbid`` from the RCSB and write it to ``out_path``.

    Creates the parent directory if necessary. Returns the path that
    was written.
    """
    text = fetch_pdb(pdbid, timeout=timeout)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    return out_path


def pose_stack_from_pdb_id(
    pdbid: str,
    *,
    canonical_ordering: Optional[Any] = None,
    packed_block_types: Optional[Any] = None,
    timeout: float = 30.0,
    **kwargs: Any,
):
    """Fetch a PDB by ID from the RCSB and build a :class:`PoseStack`.

    Thin wrapper: calls :func:`fetch_pdb` and forwards the result to
    :func:`tmol_jax.io.pose_stack_from_pdb`. ``**kwargs`` are forwarded to
    the underlying pose-stack builder.
    """
    from tmol_jax.io.pdb_parsing import pose_stack_from_pdb

    text = fetch_pdb(pdbid, timeout=timeout)
    return pose_stack_from_pdb(
        text,
        canonical_ordering=canonical_ordering,
        packed_block_types=packed_block_types,
        **kwargs,
    )


__all__ = [
    "fetch_pdb",
    "fetch_pdb_to_file",
    "pose_stack_from_pdb_id",
    "RCSB_PDB_URL",
]
