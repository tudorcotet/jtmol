#!/usr/bin/env python3
"""Fail-closed jtmol versus official-upstream-tmol benchmark matrix.

This harness deliberately runs in three separate, subprocess-friendly stages::

    python benchmarks/exact_upstream_matrix.py upstream \
        --manifest benchmark_manifest.json --out-dir out/upstream
    python benchmarks/exact_upstream_matrix.py jtmol \
        --manifest benchmark_manifest.json \
        --upstream-dir out/upstream --out-dir out/jtmol
    python benchmarks/exact_upstream_matrix.py compare \
        --upstream-dir out/upstream --jtmol-dir out/jtmol \
        --output out/comparison.json

The manifest schema is intentionally small::

    {
      "pdb_dir": "/absolute/or/manifest-relative/pdb/directory",
      "s1_id": "1ubq",
      "s3_ids": ["id_000", "id_001", "... exactly 128 unique ids ..."]
    }

An id may include ``.pdb``; otherwise that suffix is added.  S3 ids and their
semantic topology hashes must both be unique.  This prevents four structures
repeated 32 times from being described as 128 different structures.
The versioned ``benchmarks/upstream_exact_panel_v1.json`` rich schema is also
accepted; its PDB sizes/hashes and exact upstream wheel provenance are checked.

The upstream stage is the sole owner of coordinate generation.  It writes the
exact float32 tensors used for all three scenarios to ``data.npz``:

* S1: the native upstream coordinates for one structure;
* S2: 128 deterministic, low-frequency smooth deformations of S1;
* S3: the native padded coordinates of one upstream 128-pose stack.

The jtmol stage reads those bytes and is not permitted to regenerate or perturb
them.  It uses the dense score path for the strict streaming comparison.  S3
is attempted as one native heterogeneous PoseStack; an OOM is recorded as a
failure and is never converted into a silently chunked result.

Both backends emit weighted per-channel energies for the official 22-channel
``beta2016.sfxn`` contract, the coordinate gradient of each weighted channel
sum, total per-pose energies, and ``d(sum(E_per_pose))/d(coords)``.  Channel
gradients are an untimed accuracy oracle.  Forward and value-plus-gradient
timings are warm, synchronized wall-clock measurements.  Setup/render/JIT time
is recorded separately and is not included in steady-state timings.

Known upstream exception
------------------------
``lk_bridge_uncpl`` has a documented upstream adjoint bug.  This harness does
*not* silently exempt it.  A mismatch still fails comparison and is annotated
in the report.  Accepting a deliberate divergence requires a separate
high-precision finite-difference oracle; it cannot be converted into a pass by
this benchmark.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
UPSTREAM_VERSION = "0.1.36"
UPSTREAM_RELEASE_COMMIT = "ea3d46295ab142c9f6f57a171638d420af41e034"
UPSTREAM_WHEEL_FILENAME = (
    "tmol-0.1.36+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
)
UPSTREAM_WHEEL_URL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.36/"
    + UPSTREAM_WHEEL_FILENAME
)
UPSTREAM_WHEEL_SHA256 = (
    "2093723b74eb4fff4a7ccf2ccb2f08f3bcee5171fc092044a38871ccecbd89dc"
)
JAX_VERSION = "0.10.2"
JAXLIB_VERSION = "0.10.2"
JAX_CUDA_PLUGIN_VERSION = "0.10.2"
ARRAYS_FILENAME = "data.npz"
METADATA_FILENAME = "contract.json"
SCENARIOS = ("S1", "S2", "S3")

# ScoreType enum order, not YAML insertion order.  Upstream's rendered
# per-channel tensor is reordered by name into this order before it is saved.
OFFICIAL_WEIGHTS: dict[str, float] = {
    "fa_ljatr": 1.0,
    "fa_ljrep": 0.55,
    "fa_lk": 1.0,
    "fa_elec": 1.0,
    "hbond": 1.0,
    "cart_lengths": 0.5,
    "cart_angles": 0.5,
    "cart_torsions": 0.5,
    "cart_impropers": 0.5,
    "cart_hxltorsions": 0.5,
    "disulfide": 1.25,
    "omega": 0.48,
    "rama": 0.50,
    "dunbrack_rot": 0.76,
    "dunbrack_rotdev": 0.69,
    "dunbrack_semirot": 0.78,
    "lk_ball_iso": -0.38,
    "lk_ball": 0.92,
    "lk_bridge": -0.33,
    "lk_bridge_uncpl": -0.33,
    "ref": 1.0,
    "gen_torsions": 1.0,
}
OFFICIAL_NAMES = tuple(OFFICIAL_WEIGHTS)

KNOWN_LK_BRIDGE_UNCPL = (
    "Upstream tmol has a documented lk_bridge_uncpl adjoint bug. No automatic "
    "exception is applied: any energy or full-gradient mismatch remains a "
    "failure until a separate finite-difference oracle adjudicates it."
)
NATIVE_COORD_ATOL = 1e-4
CERTIFICATION_TOLERANCES = {
    "energy_atol": 1e-4,
    "energy_rtol": 2e-6,
    "total_atol": 5e-4,
    "total_rtol": 2e-6,
    "grad_atol": 5e-4,
    "grad_rtol": 2e-5,
    "grad_rel_l2": 2e-5,
}


class ContractError(RuntimeError):
    """Raised when two artifacts do not describe exactly the same workload."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _directory_hash(path: Path) -> str:
    h = hashlib.sha256()
    for child in sorted(x for x in path.rglob("*") if x.is_file()):
        relative = child.relative_to(path)
        if "__pycache__" in relative.parts or child.suffix == ".pyc":
            continue
        h.update(_canonical_json(str(relative)))
        h.update(bytes.fromhex(_file_hash(child)))
    return h.hexdigest()


def _array_hash(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(_canonical_json({"dtype": arr.dtype.str, "shape": list(arr.shape)}))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


def _update_hash_with_array(h: "hashlib._Hash", value: Any) -> None:
    arr = np.ascontiguousarray(np.asarray(value))
    h.update(_canonical_json({"dtype": arr.dtype.str, "shape": list(arr.shape)}))
    h.update(arr.view(np.uint8))


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _failure(stage: str, exc: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc)[:2000],
        "traceback_tail": traceback.format_exc()[-4000:],
        "oom": "out of memory" in str(exc).lower()
        or "resource_exhausted" in str(exc).lower(),
    }


def _timing_summary(samples_ms: Sequence[float], n_poses: int) -> dict[str, Any]:
    a = np.asarray(samples_ms, dtype=np.float64)
    median = float(np.median(a))
    return {
        "median_ms_batch": median,
        "median_ms_per_pose": median / n_poses,
        "min_ms_batch": float(np.min(a)),
        "p10_ms_batch": float(np.percentile(a, 10)),
        "p90_ms_batch": float(np.percentile(a, 90)),
        "max_ms_batch": float(np.max(a)),
        "samples_ms_batch": [float(x) for x in a],
    }


def _resolve_pdb(pdb_dir: Path, identifier: str) -> Path:
    rel = Path(identifier)
    if rel.suffix.lower() != ".pdb":
        rel = rel.with_suffix(".pdb")
    path = rel if rel.is_absolute() else pdb_dir / rel
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"PDB not found for id {identifier!r}: {path}")
    return path


def _resolve_manifest_file(manifest_path: Path, value: str) -> Path:
    """Resolve a manifest path against its directory or the repository root."""

    requested = Path(value)
    candidates = (
        (requested,)
        if requested.is_absolute()
        else (manifest_path.parent / requested, manifest_path.parent.parent / requested)
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    rendered = ", ".join(str(x.resolve()) for x in candidates)
    raise ContractError(f"manifest file {value!r} not found; tried {rendered}")


def _validate_pinned_file(
    path: Path, *, sha256: str | None, size_bytes: int | None, label: str
) -> None:
    if not path.is_file():
        raise ContractError(f"{label} is not a file: {path}")
    if sha256 is not None and _file_hash(path) != sha256:
        raise ContractError(f"{label} SHA256 differs from the manifest: {path}")
    if size_bytes is not None and path.stat().st_size != size_bytes:
        raise ContractError(f"{label} size differs from the manifest: {path}")


def _load_manifest(path: Path, pdb_dir_override: str | None = None) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ContractError("manifest must contain a JSON object")

    # The repository's versioned panel carries the exact wheel provenance and
    # per-file hashes.  Continue to accept the deliberately small standalone
    # schema documented above for ad-hoc runs.
    if "scenarios" in raw:
        scenarios = raw.get("scenarios")
        if not isinstance(scenarios, dict):
            raise ContractError("manifest.scenarios must be an object")
        try:
            s1_scenario = scenarios["s1_k1"]
            s2_scenario = scenarios["s2_k128_same_topology"]
            s3_scenario = scenarios["s3_k128_different_structures"]
            s1_entry = s1_scenario["structure"]
            s2_entry = s2_scenario["structure"]
            s3_entries = s3_scenario["structures"]
        except (KeyError, TypeError) as exc:
            raise ContractError("rich manifest is missing a benchmark scenario") from exc
        if s1_scenario.get("k") != 1 or s2_scenario.get("k") != 128:
            raise ContractError("rich manifest S1/S2 K values must be 1 and 128")
        if s3_scenario.get("k") != 128 or not isinstance(s3_entries, list):
            raise ContractError("rich manifest S3 must contain K=128 structures")
        if len(s3_entries) != 128:
            raise ContractError("rich manifest S3 must contain exactly 128 structures")
        if s1_entry.get("pdb_id") != s2_entry.get("pdb_id"):
            raise ContractError("rich manifest S1 and S2 must use the same topology")
        if s1_entry.get("sha256") != s2_entry.get("sha256"):
            raise ContractError("rich manifest S1 and S2 PDB hashes differ")

        s1_id = s1_entry.get("pdb_id")
        s1_relpath = s1_entry.get("path")
        if not isinstance(s1_id, str) or not isinstance(s1_relpath, str):
            raise ContractError("rich manifest S1 structure is malformed")
        s1_path = _resolve_manifest_file(path, s1_relpath)
        _validate_pinned_file(
            s1_path,
            sha256=s1_entry.get("sha256"),
            size_bytes=s1_entry.get("size_bytes"),
            label="S1 PDB",
        )

        s3_ids: list[str] = []
        filenames: list[str] = []
        for pose_ind, entry in enumerate(s3_entries):
            if not isinstance(entry, dict):
                raise ContractError(f"rich manifest S3 entry {pose_ind} is malformed")
            pdb_id, filename = entry.get("pdb_id"), entry.get("filename")
            if not isinstance(pdb_id, str) or not pdb_id:
                raise ContractError(f"rich manifest S3 id {pose_ind} is malformed")
            if not isinstance(filename, str) or not filename:
                raise ContractError(f"rich manifest S3 filename {pose_ind} is malformed")
            s3_ids.append(pdb_id)
            filenames.append(filename)
        if len(set(s3_ids)) != 128 or len(set(filenames)) != 128:
            raise ContractError("rich manifest S3 ids and filenames must be unique")

        if pdb_dir_override is not None:
            pdb_dir = Path(pdb_dir_override)
        else:
            cache_hint = raw.get("selection", {}).get("local_cache_hint")
            if not isinstance(cache_hint, str) or not cache_hint:
                raise ContractError("rich manifest needs --pdb-dir or local_cache_hint")
            pdb_dir = Path(cache_hint)
        if not pdb_dir.is_absolute():
            pdb_dir = path.parent / pdb_dir
        pdb_dir = pdb_dir.resolve()
        if not pdb_dir.is_dir():
            raise ContractError(f"S3 PDB directory is not a directory: {pdb_dir}")

        s3_paths = [_resolve_pdb(pdb_dir, name) for name in filenames]
        for pose_ind, (entry, pdb_path) in enumerate(zip(s3_entries, s3_paths)):
            _validate_pinned_file(
                pdb_path,
                sha256=entry.get("sha256"),
                size_bytes=entry.get("size_bytes"),
                label=f"S3[{pose_ind}]/{s3_ids[pose_ind]} PDB",
            )
        return {
            "raw": raw,
            "manifest_sha256": _json_hash(raw),
            "manifest_path": str(path.resolve()),
            "pdb_dir": str(pdb_dir),
            "s1_id": s1_id,
            "s1_path": s1_path,
            "s3_ids": s3_ids,
            "s3_paths": s3_paths,
            "upstream_pin": raw.get("upstream_tmol"),
        }

    if not isinstance(raw.get("s1_id"), str) or not raw["s1_id"]:
        raise ContractError("manifest.s1_id must be a non-empty string")
    s3_ids = raw.get("s3_ids")
    if not isinstance(s3_ids, list) or len(s3_ids) != 128:
        raise ContractError("manifest.s3_ids must contain exactly 128 ids")
    if not all(isinstance(x, str) and x for x in s3_ids):
        raise ContractError("every manifest.s3_ids entry must be a non-empty string")
    if len(set(s3_ids)) != 128:
        raise ContractError("manifest.s3_ids must be 128 unique ids")
    if not isinstance(raw.get("pdb_dir"), str) or not raw["pdb_dir"]:
        raise ContractError("manifest.pdb_dir must be a non-empty string")

    pdb_dir = Path(pdb_dir_override or raw["pdb_dir"])
    if not pdb_dir.is_absolute():
        pdb_dir = path.parent / pdb_dir
    pdb_dir = pdb_dir.resolve()
    if not pdb_dir.is_dir():
        raise ContractError(f"manifest.pdb_dir is not a directory: {pdb_dir}")

    s1_path = _resolve_pdb(pdb_dir, raw["s1_id"])
    s3_paths = [_resolve_pdb(pdb_dir, x) for x in s3_ids]
    return {
        "raw": raw,
        "manifest_sha256": _json_hash(raw),
        "manifest_path": str(path.resolve()),
        "pdb_dir": str(pdb_dir),
        "s1_id": raw["s1_id"],
        "s1_path": s1_path,
        "s3_ids": list(s3_ids),
        "s3_paths": s3_paths,
        "upstream_pin": raw.get("upstream_tmol"),
    }


def _atom_name(atom: Any) -> str:
    return str(getattr(atom, "name", atom))


def _connection_name(conn: Any) -> str:
    return str(getattr(conn, "name", ""))


def _connection_atom(conn: Any) -> str:
    return str(getattr(conn, "atom", ""))


def _normalized_bond_type(value: Any) -> str:
    if value is None:
        return "SINGLE"
    if hasattr(value, "name"):
        return str(value.name).upper()
    rendered = str(value).upper()
    return rendered.rsplit(".", 1)[-1]


def _connection_type(conn: Any) -> str:
    return _normalized_bond_type(getattr(conn, "type", "SINGLE"))


def _pose_contract(pose_stack: Any, *, pdb_path: Path, identifier: str) -> dict[str, Any]:
    """Return backend-neutral hashes for a one-pose PoseStack.

    Palette indices are intentionally excluded.  Residue type names, ordered
    atoms, intra-residue bonds, named connections, direct inter-residue
    connections, offsets, and the count-pair separation tensor are included.
    """

    if int(_as_numpy(pose_stack.block_type_ind).shape[0]) != 1:
        raise ContractError("_pose_contract expects a one-pose PoseStack")

    pbt = pose_stack.packed_block_types
    bti_all = _as_numpy(pose_stack.block_type_ind).astype(np.int64, copy=False)
    bti = bti_all[0]
    real_b = np.flatnonzero(bti >= 0)
    if len(real_b) == 0:
        raise ContractError(f"{identifier}: pose contains no real blocks")
    n_blocks = int(real_b[-1]) + 1
    if not np.array_equal(real_b, np.arange(n_blocks)):
        raise ContractError(f"{identifier}: real blocks are not left-justified")

    n_atoms_by_type = _as_numpy(pbt.n_atoms).astype(np.int64, copy=False)
    atom_keys: list[list[Any]] = []
    block_descriptors: list[dict[str, Any]] = []

    for block_ind in range(n_blocks):
        bt_ind = int(bti[block_ind])
        bt = pbt.active_block_types[bt_ind]
        atoms = [_atom_name(a) for a in bt.atoms]
        expected_n = int(n_atoms_by_type[bt_ind])
        if len(atoms) != expected_n:
            raise ContractError(
                f"{identifier}: {getattr(bt, 'name', bt_ind)} atom count "
                f"{len(atoms)} != packed count {expected_n}"
            )
        atom_keys.extend([[block_ind, str(bt.name), name] for name in atoms])

        bond_records: set[tuple[str, str, str, bool]] = set()
        for bond in getattr(bt, "bonds", ()):
            try:
                a_name, b_name = str(bond[0]), str(bond[1])
                bond_type = _normalized_bond_type(
                    bond[2] if len(bond) > 2 else "SINGLE"
                )
                is_ring = bool(bond[3]) if len(bond) > 3 else False
            except Exception as exc:  # pragma: no cover - defensive schema guard
                raise ContractError(f"{identifier}: unsupported bond record {bond!r}") from exc
            first, second = sorted((a_name, b_name))
            bond_records.add((first, second, bond_type, is_ring))

        connections = [
            [_connection_name(c), _connection_atom(c), _connection_type(c)]
            for c in getattr(bt, "connections", ())
        ]
        block_descriptors.append(
            {
                "name": str(bt.name),
                "atoms": atoms,
                "bonds": [list(x) for x in sorted(bond_records)],
                "connections": connections,
            }
        )

    n_atoms = len(atom_keys)
    coords_all = _as_numpy(pose_stack.coords)
    if coords_all.ndim != 3 or coords_all.shape[0] != 1 or coords_all.shape[2] != 3:
        raise ContractError(f"{identifier}: invalid coords shape {coords_all.shape}")
    if coords_all.shape[1] < n_atoms:
        raise ContractError(
            f"{identifier}: coords hold {coords_all.shape[1]} atoms, need {n_atoms}"
        )
    native_coords = np.ascontiguousarray(coords_all[0, :n_atoms], dtype=np.float32)

    irc = _as_numpy(pose_stack.inter_residue_connections)[0]
    inter_edges: list[list[Any]] = []
    for block_ind in range(n_blocks):
        bt = pbt.active_block_types[int(bti[block_ind])]
        connections = list(getattr(bt, "connections", ()))
        for conn_ind in range(min(len(connections), irc.shape[1])):
            other_block = int(irc[block_ind, conn_ind, 0])
            other_conn = int(irc[block_ind, conn_ind, 1])
            if other_block < 0:
                continue
            if other_block >= n_blocks:
                raise ContractError(f"{identifier}: connection points to padding block")
            other_bt = pbt.active_block_types[int(bti[other_block])]
            other_connections = list(getattr(other_bt, "connections", ()))
            if other_conn < 0 or other_conn >= len(other_connections):
                raise ContractError(f"{identifier}: invalid partner connection index")
            inter_edges.append(
                [
                    block_ind,
                    _connection_name(connections[conn_ind]),
                    other_block,
                    _connection_name(other_connections[other_conn]),
                ]
            )
    inter_edges.sort()

    offsets = np.ascontiguousarray(
        _as_numpy(pose_stack.block_coord_offset)[0, :n_blocks], dtype=np.int32
    )
    bondsep = np.ascontiguousarray(
        _as_numpy(pose_stack.inter_block_bondsep)[0, :n_blocks, :n_blocks],
        dtype=np.int32,
    )

    # ``execution_layout`` is the backend-neutral tensor contract consumed by
    # the score kernels: ordered residue/atom identities plus the exact
    # coordinate offsets, connection-index tensor, and count-pair tensor.  It
    # deliberately does not pretend that matching integer connection indices
    # proves preservation of richer chemical annotations; those are recorded
    # separately below and remain part of the exact-port verdict.
    execution_layout_descriptor = {
        "blocks": [
            {"name": block["name"], "atoms": block["atoms"]}
            for block in block_descriptors
        ]
    }
    execution_layout_hasher = hashlib.sha256()
    execution_layout_hasher.update(_canonical_json(execution_layout_descriptor))
    _update_hash_with_array(execution_layout_hasher, offsets)
    _update_hash_with_array(
        execution_layout_hasher,
        np.ascontiguousarray(irc[:n_blocks], dtype=np.int32),
    )
    _update_hash_with_array(execution_layout_hasher, bondsep)

    topology_descriptor = {
        "blocks": block_descriptors,
        "inter_residue_connections": inter_edges,
    }
    connectivity_descriptor = {
        "blocks": [
            {
                "name": block["name"],
                "atoms": block["atoms"],
                # Project to an *untyped set* before hashing.  Upstream may
                # retain two typed records for one atom pair (for example a
                # C-O SINGLE and C-O DOUBLE record); projecting first and then
                # de-duplicating avoids turning that richer metadata into a
                # false untyped-connectivity difference.
                "bonds": [
                    list(bond)
                    for bond in sorted(
                        {tuple(typed_bond[:2]) for typed_bond in block["bonds"]}
                    )
                ],
                "connections": [connection[:2] for connection in block["connections"]],
            }
            for block in block_descriptors
        ],
        "inter_residue_connections": inter_edges,
    }
    chemical_metadata_descriptor = {
        "blocks": [
            {
                "name": block["name"],
                "bond_types_and_ringness": block["bonds"],
                "connection_types": block["connections"],
            }
            for block in block_descriptors
        ]
    }
    topology_hasher = hashlib.sha256()
    topology_hasher.update(_canonical_json(topology_descriptor))
    _update_hash_with_array(topology_hasher, offsets)
    _update_hash_with_array(topology_hasher, bondsep)
    connectivity_hasher = hashlib.sha256()
    connectivity_hasher.update(_canonical_json(connectivity_descriptor))
    _update_hash_with_array(connectivity_hasher, offsets)
    _update_hash_with_array(connectivity_hasher, bondsep)

    return {
        "id": identifier,
        "pdb_sha256": _file_hash(pdb_path),
        "n_blocks": n_blocks,
        "n_atoms": n_atoms,
        "native_coord_shape": list(native_coords.shape),
        "native_coord_sha256": _array_hash(native_coords),
        "atom_order_sha256": _json_hash(atom_keys),
        "execution_layout_sha256": execution_layout_hasher.hexdigest(),
        "connectivity_topology_sha256": connectivity_hasher.hexdigest(),
        "chemical_metadata_sha256": _json_hash(chemical_metadata_descriptor),
        "topology_sha256": topology_hasher.hexdigest(),
    }


def _assert_contract_equal(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    """Require the same score-kernel workload, independent of palette indices.

    Rich bond/connection metadata is checked separately by
    ``_representation_contract_differences``.  It must not be silently treated
    as equal, but it also must not prevent us from measuring whether the two
    implementations produce the same 22 outputs on byte-identical fixtures.
    """

    keys = (
        "id",
        "pdb_sha256",
        "n_blocks",
        "n_atoms",
        "native_coord_shape",
        "atom_order_sha256",
        "execution_layout_sha256",
    )
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in keys
        if expected.get(key) != actual.get(key)
    }
    if mismatches:
        raise ContractError(f"native pose contract mismatch: {mismatches}")


def _representation_contract_differences(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return richer topology differences that invalidate an exact-port claim."""

    keys = (
        "connectivity_topology_sha256",
        "chemical_metadata_sha256",
        "topology_sha256",
    )
    return {
        key: {"upstream": expected.get(key), "jtmol": actual.get(key)}
        for key in keys
        if expected.get(key) != actual.get(key)
    }


def _native_coord_max_abs(
    upstream_coords: Any, jtmol_coords: Any, *, identifier: str
) -> float:
    """Validate native-coordinate shape/finiteness and return max difference."""

    upstream = np.asarray(upstream_coords, dtype=np.float32)
    candidate = np.asarray(jtmol_coords, dtype=np.float32)
    if upstream.shape != candidate.shape:
        raise ContractError(
            f"{identifier}: native coordinate shape mismatch "
            f"{candidate.shape} != {upstream.shape}"
        )
    if not np.all(np.isfinite(upstream)) or not np.all(np.isfinite(candidate)):
        raise ContractError(f"{identifier}: native coordinates contain NaN/Inf")
    return float(np.max(np.abs(candidate.astype(np.float64) - upstream)))


def _assert_native_coords_close(
    upstream_coords: Any, jtmol_coords: Any, *, identifier: str
) -> float:
    """Permit only the documented tiny PDB-reconstruction roundoff envelope.

    Fixture bytes still come exclusively from upstream.  This check concerns
    each backend's independently reconstructed *native* coordinates, where
    missing-atom placement can differ by a few float32 ulps.
    """

    max_abs = _native_coord_max_abs(
        upstream_coords, jtmol_coords, identifier=identifier
    )
    if max_abs > NATIVE_COORD_ATOL:
        raise ContractError(
            f"{identifier}: native coordinate max |delta| {max_abs:.3e} exceeds "
            f"{NATIVE_COORD_ATOL:.1e}"
        )
    return max_abs


def _smooth_s2_fixture(base: np.ndarray, n_poses: int = 128) -> np.ndarray:
    """Create deterministic, spatially smooth deformations of one pose.

    The displacement is a low-frequency vector field over molecular space,
    plus a small rigid rotation.  Neighboring atoms therefore move similarly;
    this avoids the severe bond/angle damage caused by independent atom noise.
    The first member is exactly the native structure.
    """

    base32 = np.ascontiguousarray(base, dtype=np.float32)
    if base32.ndim != 2 or base32.shape[1] != 3:
        raise ContractError(f"S2 base must have shape [N,3], got {base32.shape}")
    x = base32.astype(np.float64)
    center = x.mean(axis=0)
    xc = x - center
    spatial_scale = max(float(np.linalg.norm(np.ptp(xc, axis=0))), 1.0)
    q = xc / spatial_scale
    out = np.empty((n_poses,) + base32.shape, dtype=np.float32)
    out[0] = base32

    for i in range(1, n_poses):
        phase = 2.0 * math.pi * i / n_poses
        axis = np.array(
            [
                0.7 + 0.2 * math.sin(phase),
                0.5 + 0.2 * math.cos(2.0 * phase),
                0.4 + 0.1 * math.sin(3.0 * phase),
            ],
            dtype=np.float64,
        )
        axis /= np.linalg.norm(axis)
        angle = 0.025 * math.sin(phase)
        cross = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        rotation = (
            np.eye(3)
            + math.sin(angle) * cross
            + (1.0 - math.cos(angle)) * (cross @ cross)
        )
        rotated = xc @ rotation.T

        # A quarter-Angstrom low-frequency deformation is still gentle at the
        # bond scale, but materially exercises nonlinear score regions and
        # moves some nonbonded pairs across cutoff/fade boundaries.  The old
        # 0.035 A amplitude was too close to 128 rigid copies to expose crude
        # frozen-neighbor approximations.
        amplitude = 0.25 * math.sin(phase * 0.5 + 0.3)
        field = np.stack(
            [
                np.sin(2.0 * math.pi * (0.37 * q[:, 0] + 0.19 * q[:, 1]) + phase),
                np.sin(2.0 * math.pi * (0.31 * q[:, 1] - 0.17 * q[:, 2]) + 1.3 * phase),
                np.sin(2.0 * math.pi * (0.29 * q[:, 2] + 0.13 * q[:, 0]) - 0.7 * phase),
            ],
            axis=1,
        )
        field -= field.mean(axis=0, keepdims=True)
        translation = 0.05 * np.array(
            [math.sin(phase), math.cos(phase), math.sin(2.0 * phase)],
            dtype=np.float64,
        )
        out[i] = (rotated + amplitude * field + center + translation).astype(np.float32)

    if not np.array_equal(out[0], base32):
        raise AssertionError("S2 fixture member zero must be byte-identical to S1")
    if not np.all(np.isfinite(out)):
        raise ContractError("S2 fixture contains non-finite coordinates")
    if len({_array_hash(member) for member in out}) != n_poses:
        raise ContractError(f"S2 fixture must contain {n_poses} distinct conformations")
    return out


def _fixture_metadata(
    key: str,
    coords: np.ndarray,
    ids: Sequence[str],
    per_pose: Sequence[Mapping[str, Any]],
    *,
    generator: str,
) -> dict[str, Any]:
    return {
        "array_key": key,
        "coord_shape": list(coords.shape),
        "coord_dtype": np.asarray(coords).dtype.str,
        "coord_sha256": _array_hash(coords),
        "ids": list(ids),
        "per_pose": [dict(x) for x in per_pose],
        "generator": generator,
    }


def _validate_saved_fixture(meta: Mapping[str, Any], arrays: Mapping[str, Any]) -> np.ndarray:
    key = str(meta["array_key"])
    if key not in arrays:
        raise ContractError(f"fixture array missing: {key}")
    coords = np.asarray(arrays[key])
    if list(coords.shape) != list(meta["coord_shape"]):
        raise ContractError(f"{key}: shape mismatch {coords.shape} != {meta['coord_shape']}")
    if coords.dtype.str != meta["coord_dtype"]:
        raise ContractError(f"{key}: dtype mismatch {coords.dtype.str} != {meta['coord_dtype']}")
    if coords.dtype != np.float32:
        raise ContractError(f"{key}: fixtures must be float32, got {coords.dtype}")
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ContractError(f"{key}: coordinates must have shape [K,N,3]")
    ids = meta.get("ids")
    per_pose = meta.get("per_pose")
    if not isinstance(ids, list) or len(ids) != coords.shape[0]:
        raise ContractError(f"{key}: ids do not match coordinate batch size")
    if not isinstance(per_pose, list) or len(per_pose) != coords.shape[0]:
        raise ContractError(f"{key}: per-pose contracts do not match batch size")
    for pose_ind, contract in enumerate(per_pose):
        n_atoms = int(contract.get("n_atoms", -1))
        if n_atoms < 1 or n_atoms > coords.shape[1]:
            raise ContractError(f"{key}: invalid n_atoms for pose {pose_ind}: {n_atoms}")
    if _array_hash(coords) != meta["coord_sha256"]:
        raise ContractError(f"{key}: coordinate hash mismatch")
    if not np.all(np.isfinite(coords)):
        raise ContractError(f"{key}: non-finite coordinates")
    return np.ascontiguousarray(coords)


def _verify_official_weights(names: Iterable[str], weight_for: Callable[[str], float]) -> None:
    names_set = set(names)
    expected = set(OFFICIAL_NAMES)
    if names_set != expected:
        raise ContractError(
            f"score channel set mismatch; missing={sorted(expected - names_set)}, "
            f"extra={sorted(names_set - expected)}"
        )
    mismatched: dict[str, Any] = {}
    for name, expected_weight in OFFICIAL_WEIGHTS.items():
        actual = float(weight_for(name))
        # Upstream stores weights in a float32 torch tensor; jtmol keeps the
        # Python literal.  Require the exact same float32 bit pattern instead
        # of incorrectly rejecting values such as 0.55 after tensor storage.
        expected_bits = np.asarray(expected_weight, dtype=np.float32).tobytes()
        actual_bits = np.asarray(actual, dtype=np.float32).tobytes()
        if not math.isfinite(actual) or actual_bits != expected_bits:
            mismatched[name] = {"expected": expected_weight, "actual": actual}
    if mismatched:
        raise ContractError(f"score weights differ from official beta2016: {mismatched}")


def _torch_timing(
    fn: Callable[[], Any], *, warmup: int, reps: int, n_poses: int, torch: Any
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return _timing_summary(samples, n_poses)


def _jax_block(value: Any, jax: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _jax_timing(
    fn: Callable[[], Any], *, warmup: int, reps: int, n_poses: int, jax: Any
) -> dict[str, Any]:
    for _ in range(warmup):
        _jax_block(fn(), jax)
    samples: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        result = fn()
        _jax_block(result, jax)
        samples.append((time.perf_counter() - start) * 1000.0)
    return _timing_summary(samples, n_poses)


def _physical_gpu_context() -> dict[str, Any]:
    """Capture enough host/UUID context to reject cross-allocation ratios."""

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        inventory = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    except Exception as exc:
        raise ContractError("nvidia-smi GPU identity query failed") from exc
    if not inventory:
        raise ContractError("nvidia-smi returned no physical GPU identities")
    return {
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_inventory": inventory,
        "logical_device_index": 0,
    }


def _upstream_environment(
    torch: Any, tmol_module: Any, pin: Mapping[str, Any] | None
) -> dict[str, Any]:
    from importlib.metadata import distribution
    from packaging.version import Version

    dist = distribution("tmol")
    installed = dist.version
    # The release filename uses a PEP 440 local tag to encode its CUDA/Torch
    # build (``+cu128torch2.10``), while the wheel's METADATA intentionally
    # declares the public project version ``0.1.36``.  The exact build is
    # authenticated below by its direct URL and SHA-256, so compare the
    # installed metadata to the official public version here.
    expected_version = Version(UPSTREAM_VERSION)
    if Version(installed) != expected_version:
        raise ContractError(
            f"exact official tmol wheel version {expected_version} required, "
            f"found {installed}"
        )
    module_path = str(Path(tmol_module.__file__).resolve())
    if "site-packages" not in module_path:
        raise ContractError(
            "upstream must be the installed official release wheel, not the "
            f"gitignored source clone; imported {module_path}"
        )
    if not torch.cuda.is_available():
        raise ContractError("CUDA is required for the performance matrix")
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != "3.12":
        raise ContractError(f"pinned Python 3.12 required, found {actual_python}")
    torch_public = str(torch.__version__).split("+", 1)[0]
    if not torch_public.startswith("2.10."):
        raise ContractError(f"pinned torch 2.10.* required, found {torch.__version__}")
    if str(torch.version.cuda) != "12.8":
        raise ContractError(
            f"pinned torch CUDA 12.8 required, found {torch.version.cuda}"
        )

    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text is None:
        raise ContractError(
            "tmol installation has no direct_url.json; exact official wheel "
            "provenance cannot be verified"
        )
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise ContractError("tmol direct_url.json is invalid") from exc
    if direct_url.get("url") != UPSTREAM_WHEEL_URL:
        raise ContractError("installed tmol wheel URL is not the official pinned URL")
    archive_info = direct_url.get("archive_info", {})
    installed_sha = archive_info.get("hashes", {}).get("sha256")
    if installed_sha is None:
        legacy_hash = archive_info.get("hash", "")
        if isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
            installed_sha = legacy_hash.removeprefix("sha256=")
    if installed_sha != UPSTREAM_WHEEL_SHA256:
        raise ContractError("installed tmol wheel SHA256 is not the pinned official SHA")

    provenance: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tmol_version": installed,
        "tmol_module": module_path,
        "tmol_release_commit": UPSTREAM_RELEASE_COMMIT,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "physical_gpu_context": _physical_gpu_context(),
        "verified_wheel": {
            "filename": UPSTREAM_WHEEL_FILENAME,
            "url": UPSTREAM_WHEEL_URL,
            "sha256": UPSTREAM_WHEEL_SHA256,
        },
    }

    if pin is not None:
        if not isinstance(pin, Mapping):
            raise ContractError("manifest.upstream_tmol must be an object")
        if pin.get("version") != UPSTREAM_VERSION:
            raise ContractError("manifest pins a different upstream tmol version")
        if pin.get("git_commit") != UPSTREAM_RELEASE_COMMIT:
            raise ContractError("manifest pins a different upstream release commit")
        wheel = pin.get("wheel")
        if not isinstance(wheel, Mapping):
            raise ContractError("rich manifest is missing upstream_tmol.wheel")
        expected_pin = {
            "filename": UPSTREAM_WHEEL_FILENAME,
            "url": UPSTREAM_WHEEL_URL,
            "sha256": UPSTREAM_WHEEL_SHA256,
            "python": "3.12",
            "torch": "2.10.*",
            "cuda": "12.8",
        }
        mismatched_pin = {
            key: {"expected": expected, "actual": wheel.get(key)}
            for key, expected in expected_pin.items()
            if wheel.get(key) != expected
        }
        if mismatched_pin:
            raise ContractError(f"manifest upstream wheel pin differs: {mismatched_pin}")

    return provenance


def _upstream_score(
    pose_stack: Any,
    coords_np: np.ndarray,
    *,
    warmup: int,
    reps: int,
    device: Any,
    torch: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from tmol.database import ParameterDatabase
    from tmol.score import beta2016_score_function
    from tmol.score.score_types import ScoreType

    if tuple(coords_np.shape) != tuple(pose_stack.coords.shape):
        raise ContractError(
            f"upstream native stack shape {tuple(pose_stack.coords.shape)} != "
            f"fixture shape {coords_np.shape}"
        )
    if coords_np.dtype != np.float32:
        raise ContractError(f"upstream coordinates must be float32, got {coords_np.dtype}")

    setup_start = time.perf_counter()
    score_function = beta2016_score_function(
        device, param_db=ParameterDatabase.get_default()
    )
    scorer = score_function.render_whole_pose_scoring_module(pose_stack)
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    native_names = [st.name for st in score_function.all_score_types()]
    _verify_official_weights(
        native_names,
        lambda name: float(score_function.get_weight(ScoreType[name]).item()),
    )
    row_for = {name: i for i, name in enumerate(native_names)}
    reorder = [row_for[name] for name in OFFICIAL_NAMES]

    coords_fwd = torch.as_tensor(coords_np, dtype=torch.float32, device=device)
    coords_grad = coords_fwd.detach().clone().requires_grad_(True)

    def forward() -> Any:
        with torch.inference_mode():
            return scorer(coords_fwd)

    def value_and_grad() -> tuple[Any, Any]:
        energies = scorer(coords_grad)
        (gradient,) = torch.autograd.grad(energies.sum(), coords_grad)
        return energies, gradient

    with torch.inference_mode():
        weighted_native = scorer(
            coords_fwd, sum_terms=False, apply_weights=True
        )
        weighted = weighted_native[reorder]
        total = scorer(coords_fwd)
    energies_for_grad = scorer(coords_grad)
    (gradient,) = torch.autograd.grad(energies_for_grad.sum(), coords_grad)
    torch.cuda.synchronize()
    delta = torch.abs(total - energies_for_grad.detach())
    allowed = 5e-4 + 2e-6 * torch.abs(total)
    normalized = delta / allowed
    worst_abs = int(torch.argmax(delta).item())
    worst_normalized = int(torch.argmax(normalized).item())
    self_consistency = {
        "pass": bool(torch.all(delta <= allowed).item()),
        "max_abs": float(delta[worst_abs].item()),
        "max_abs_pose": worst_abs,
        "max_normalized_error": float(normalized[worst_normalized].item()),
        "worst_normalized_pose": worst_normalized,
        "worst_normalized_forward": float(total[worst_normalized].item()),
        "worst_normalized_grad_forward": float(
            energies_for_grad.detach()[worst_normalized].item()
        ),
        "atol": 5e-4,
        "rtol": 2e-6,
    }
    weighted_host = np.ascontiguousarray(weighted.detach().cpu().numpy())
    total_host = np.ascontiguousarray(total.detach().cpu().numpy())
    value_and_grad_total_host = np.ascontiguousarray(
        energies_for_grad.detach().cpu().numpy()
    )
    gradient_host = np.ascontiguousarray(gradient.detach().cpu().numpy())
    del weighted_native, weighted, total, energies_for_grad, gradient
    torch.cuda.empty_cache()

    # Time first, in a warmed but accuracy-oracle-neutral state.  Running 22
    # full reverse passes before the measured loop caused large, backend-
    # dependent thermal/cache preconditioning.
    n_poses = int(coords_np.shape[0])
    timing = {
        "setup_render_ms": setup_ms,
        "forward_value_grad_self_consistency": self_consistency,
        "forward": _torch_timing(
            forward, warmup=warmup, reps=reps, n_poses=n_poses, torch=torch
        ),
        "value_and_grad": _torch_timing(
            value_and_grad,
            warmup=warmup,
            reps=reps,
            n_poses=n_poses,
            torch=torch,
        ),
    }

    # Untimed accuracy oracle.  Recompute the graph for one weighted channel
    # at a time instead of retaining 22 reverse graphs for the heterogeneous
    # batch simultaneously.
    channel_grad_start = time.perf_counter()
    weighted_term_gradients = np.empty(
        (len(OFFICIAL_NAMES),) + tuple(coords_np.shape), dtype=np.float32
    )
    for output_row, native_row in enumerate(reorder):
        weighted_for_grad = scorer(
            coords_grad, sum_terms=False, apply_weights=True
        )
        (channel_gradient,) = torch.autograd.grad(
            weighted_for_grad[native_row].sum(), coords_grad
        )
        weighted_term_gradients[output_row] = (
            channel_gradient.detach().cpu().numpy()
        )
        del weighted_for_grad, channel_gradient
    torch.cuda.synchronize()
    channel_grad_ms = (time.perf_counter() - channel_grad_start) * 1000.0
    timing["untimed_weighted_term_gradients_ms"] = channel_grad_ms

    arrays = {
        "weighted_terms": weighted_host,
        "weighted_term_gradients": np.ascontiguousarray(weighted_term_gradients),
        "total": total_host,
        "value_and_grad_total": value_and_grad_total_host,
        "grad": gradient_host,
    }
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ContractError(f"upstream {name} contains non-finite values")

    return arrays, timing


def run_upstream(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path, args.pdb_dir)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    active_scenarios = (
        ("S1",) if args.s1_only else (("S1", "S2") if args.skip_s3 else SCENARIOS)
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backend": "upstream",
        "status": "running",
        "scenarios_requested": list(active_scenarios),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_path": manifest["manifest_path"],
        "pdb_dir": manifest["pdb_dir"],
        "benchmark_scope": manifest["raw"].get("benchmark_scope"),
        "official_weights": OFFICIAL_WEIGHTS,
        "official_channel_order": list(OFFICIAL_NAMES),
        "known_exceptions": {"lk_bridge_uncpl": KNOWN_LK_BRIDGE_UNCPL},
        "timing_contract": {
            "forward": "coords -> weighted E[n_poses]",
            "value_and_grad": "coords -> (E[n_poses], d(sum(E))/dcoords)",
            "backend_execution": "official rendered tmol scorer; eager dispatch",
            "warmup": args.warmup,
            "reps": args.reps,
            "synchronized": True,
            "setup_and_compilation_excluded": True,
        },
        "fixtures": {},
        "scenarios": {},
    }
    arrays: dict[str, np.ndarray] = {}

    try:
        import torch
        import tmol
        from tmol.io import pose_stack_from_pdb
        from tmol.pose.pose_stack_builder import PoseStackBuilder

        if args.device != "cuda":
            raise ContractError("strict performance runs require --device cuda")
        device = torch.device(args.device, 0)
        torch.cuda.set_device(0)
        metadata["environment"] = _upstream_environment(
            torch, tmol, manifest["upstream_pin"]
        )

        # S1 and S2 share one topology.  Upstream owns both fixture tensors.
        s1 = pose_stack_from_pdb(str(manifest["s1_path"]), device)
        s1_contract = _pose_contract(
            s1, pdb_path=manifest["s1_path"], identifier=manifest["s1_id"]
        )
        s1_coords = np.ascontiguousarray(
            s1.coords.detach().cpu().numpy(), dtype=np.float32
        )
        arrays["S1_coords"] = s1_coords
        metadata["fixtures"]["S1"] = _fixture_metadata(
            "S1_coords",
            s1_coords,
            [manifest["s1_id"]],
            [s1_contract],
            generator="official tmol native PoseStack.coords",
        )
        scenario_inputs: list[tuple[str, Any, np.ndarray]] = [
            ("S1", s1, s1_coords)
        ]
        if "S2" in active_scenarios:
            s2_coords = _smooth_s2_fixture(s1_coords[0], 128)
            arrays["S2_coords"] = s2_coords
            metadata["fixtures"]["S2"] = _fixture_metadata(
                "S2_coords",
                s2_coords,
                [manifest["s1_id"]] * 128,
                [s1_contract] * 128,
                generator=(
                    "smooth-spatial-v2; 0.25A low-frequency deformation; member 0 "
                    "is native S1"
                ),
            )
            scenario_inputs.append((
                "S2",
                PoseStackBuilder.from_poses([s1] * 128, device),
                s2_coords,
            ))

        for scenario, pose_stack, coords in scenario_inputs:
            try:
                result, timing = _upstream_score(
                    pose_stack,
                    coords,
                    warmup=args.warmup,
                    reps=args.reps,
                    device=device,
                    torch=torch,
                )
                for name, value in result.items():
                    arrays[f"{scenario}_{name}"] = value
                metadata["scenarios"][scenario] = {
                    "status": "ok",
                    "n_poses": int(coords.shape[0]),
                    "padded_n_atoms": int(coords.shape[1]),
                    "timing": timing,
                    "array_keys": {
                        name: f"{scenario}_{name}" for name in result
                    },
                }
            except BaseException as exc:
                metadata["scenarios"][scenario] = _failure("score_and_time", exc)

        # Release every S1/S2 PoseStack reference before constructing the much
        # larger S3 stack.  Deleting only the loop variable leaves the stacks
        # alive through ``scenario_inputs`` and can create a false S3 OOM.
        del pose_stack
        scenario_inputs.clear()
        del scenario_inputs, s1
        gc.collect()
        torch.cuda.empty_cache()

        if "S3" not in active_scenarios:
            metadata["status"] = (
                "ok"
                if all(
                    metadata["scenarios"].get(s, {}).get("status") == "ok"
                    for s in active_scenarios
                )
                else "failed"
            )
            _save_npz(out_dir / ARRAYS_FILENAME, arrays)
            _write_json(out_dir / METADATA_FILENAME, metadata)
            print(json.dumps({"status": metadata["status"], "out_dir": str(out_dir)}))
            return 0 if metadata["status"] == "ok" else 1

        try:
            s3_singles = [
                pose_stack_from_pdb(str(path), device) for path in manifest["s3_paths"]
            ]
            s3_contracts = [
                _pose_contract(ps, pdb_path=path, identifier=identifier)
                for ps, path, identifier in zip(
                    s3_singles, manifest["s3_paths"], manifest["s3_ids"]
                )
            ]
            topology_hashes = [x["topology_sha256"] for x in s3_contracts]
            if len(set(topology_hashes)) != 128:
                raise ContractError(
                    "S3 must contain 128 unique semantic topologies; found "
                    f"{len(set(topology_hashes))}"
                )
            s3 = PoseStackBuilder.from_poses(s3_singles, device)
            s3_coords = np.ascontiguousarray(
                s3.coords.detach().cpu().numpy(), dtype=np.float32
            )
            arrays["S3_coords"] = s3_coords
            metadata["fixtures"]["S3"] = _fixture_metadata(
                "S3_coords",
                s3_coords,
                manifest["s3_ids"],
                s3_contracts,
                generator="official tmol native 128-pose PoseStack.coords",
            )
            del s3_singles
            gc.collect()
            torch.cuda.empty_cache()
            try:
                result, timing = _upstream_score(
                    s3,
                    s3_coords,
                    warmup=args.warmup,
                    reps=args.reps,
                    device=device,
                    torch=torch,
                )
                for name, value in result.items():
                    arrays[f"S3_{name}"] = value
                metadata["scenarios"]["S3"] = {
                    "status": "ok",
                    "n_poses": 128,
                    "padded_n_atoms": int(s3_coords.shape[1]),
                    "native_single_batch": True,
                    "timing": timing,
                    "array_keys": {name: f"S3_{name}" for name in result},
                }
            except BaseException as exc:
                failure = _failure("native_128_score_and_time", exc)
                failure["native_single_batch"] = True
                failure["chunk_fallback_attempted"] = False
                metadata["scenarios"]["S3"] = failure
        except BaseException as exc:
            failure = _failure("native_128_fixture_or_stack", exc)
            failure["native_single_batch"] = True
            failure["chunk_fallback_attempted"] = False
            metadata["scenarios"]["S3"] = failure

        metadata["status"] = (
            "ok"
            if all(
                metadata["scenarios"].get(s, {}).get("status") == "ok"
                for s in active_scenarios
            )
            else "failed"
        )
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["fatal_error"] = _failure("upstream_contract_or_setup", exc)

    _save_npz(out_dir / ARRAYS_FILENAME, arrays)
    _write_json(out_dir / METADATA_FILENAME, metadata)
    print(json.dumps({"status": metadata["status"], "out_dir": str(out_dir)}))
    return 0 if metadata["status"] == "ok" else 1


def _jtmol_environment(jax: Any, jtmol_module: Any) -> dict[str, Any]:
    if jax.default_backend() != "gpu":
        raise ContractError(
            f"CUDA/GPU JAX backend required, found {jax.default_backend()}"
        )
    if not bool(jax.config.jax_enable_x64):
        raise ContractError(
            "JAX x64 must remain enabled: jtmol hbond uses double-precision "
            "polynomials to match upstream even for float32 coordinates"
        )
    try:
        from importlib.metadata import version

        jtmol_version = version("jtmol")
    except Exception:
        jtmol_version = "source-tree"
    import jaxlib

    if jax.__version__ != JAX_VERSION or jaxlib.__version__ != JAXLIB_VERSION:
        raise ContractError(
            f"pinned jax/jaxlib {JAX_VERSION}/{JAXLIB_VERSION} required, found "
            f"{jax.__version__}/{jaxlib.__version__}"
        )
    if f"{sys.version_info.major}.{sys.version_info.minor}" != "3.12":
        raise ContractError("strict jtmol benchmark requires pinned Python 3.12")
    try:
        cuda_plugin_version = version("jax-cuda12-plugin")
        cuda_pjrt_version = version("jax-cuda12-pjrt")
    except Exception as exc:
        raise ContractError("pinned JAX CUDA 12 plugin and PJRT packages are required") from exc
    if (
        cuda_plugin_version != JAX_CUDA_PLUGIN_VERSION
        or cuda_pjrt_version != JAX_CUDA_PLUGIN_VERSION
    ):
        raise ContractError(
            f"pinned JAX CUDA plugins {JAX_CUDA_PLUGIN_VERSION} required, found "
            f"plugin={cuda_plugin_version}, pjrt={cuda_pjrt_version}"
        )
    devices = jax.devices()
    package_root = Path(jtmol_module.__file__).resolve().parent
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jtmol_version": jtmol_version,
        "jtmol_module": str(Path(jtmol_module.__file__).resolve()),
        "jtmol_package_sha256": _directory_hash(package_root),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "jax_cuda12_plugin_version": cuda_plugin_version,
        "jax_cuda12_pjrt_version": cuda_pjrt_version,
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "gpu": str(getattr(devices[0], "device_kind", devices[0])),
        "physical_gpu_context": _physical_gpu_context(),
        "devices": [str(x) for x in devices],
    }


def _jtmol_score_function(jtmol_module: Any) -> Any:
    from jtmol.score.score_types import ScoreType

    score_function = jtmol_module.beta2016_score_function(use_neighbor_list=False)
    # These are jtmol extensions, not part of upstream beta2016.
    score_function.set_weight(ScoreType.gen_bond_length, 0.0)
    score_function.set_weight(ScoreType.gen_bond_angle, 0.0)
    _verify_official_weights(
        [st.name for st in score_function.active_score_types()],
        lambda name: score_function.get_weight(ScoreType[name]),
    )
    if score_function.get_weight(ScoreType.gen_bond_length) != 0.0:
        raise ContractError("gen_bond_length must be disabled")
    if score_function.get_weight(ScoreType.gen_bond_angle) != 0.0:
        raise ContractError("gen_bond_angle must be disabled")
    for term in score_function.terms:
        if getattr(term, "use_neighbor_list", False):
            raise ContractError("strict streaming comparison must use the dense path")
    return score_function


def _jax_weighted_term_gradients(
    weighted_terms: Callable[[Any], Any],
    coords: Any,
    *,
    jax: Any,
    jnp: Any,
) -> tuple[np.ndarray, float]:
    """Return gradients of each weighted channel sum, one reverse pass at a time."""

    def one_channel_gradient(c: Any, row: Any) -> Any:
        def loss(x: Any) -> Any:
            terms = weighted_terms(x)
            selected = jax.lax.dynamic_index_in_dim(
                terms, row, axis=0, keepdims=False
            )
            return jnp.sum(selected)

        return jax.grad(loss)(c)

    one_channel_jit = jax.jit(one_channel_gradient)
    start = time.perf_counter()
    host_gradients: list[np.ndarray] = []
    for row in range(len(OFFICIAL_NAMES)):
        gradient = one_channel_jit(coords, jnp.asarray(row, dtype=jnp.int32))
        _jax_block(gradient, jax)
        host_gradients.append(np.ascontiguousarray(np.asarray(gradient)))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return np.ascontiguousarray(np.stack(host_gradients, axis=0)), elapsed_ms


def _jtmol_score_direct(
    score_function: Any,
    pose_stack: Any,
    coords_np: np.ndarray,
    *,
    warmup: int,
    reps: int,
    jax: Any,
    jnp: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from jtmol.score.score_types import ScoreType

    if tuple(coords_np.shape) != tuple(pose_stack.coords.shape):
        raise ContractError(
            f"jtmol native stack shape {tuple(pose_stack.coords.shape)} != "
            f"fixture shape {coords_np.shape}"
        )
    coords = jax.device_put(np.ascontiguousarray(coords_np, dtype=np.float32))

    setup_start = time.perf_counter()
    state = score_function.setup_for_pose_stack(pose_stack)
    _jax_block(state, jax)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    def energy(c: Any) -> Any:
        # Upstream's public scorer returns float32 even though its hbond
        # polynomials evaluate internally in double precision.  Preserve that
        # same boundary contract without disabling jtmol's internal x64 math.
        return score_function(
            pose_stack, state=state, coords_override=c
        ).astype(jnp.float32)

    def weighted_terms(c: Any) -> Any:
        values = score_function.score_by_term(
            pose_stack, state=state, coords_override=c
        )
        return jnp.stack(
            [values[ScoreType[name]] for name in OFFICIAL_NAMES]
        ).astype(jnp.float32)

    def vg_impl(c: Any) -> tuple[Any, Any]:
        def loss_aux(x: Any) -> tuple[Any, Any]:
            per_pose = energy(x)
            return jnp.sum(per_pose), per_pose

        (_, per_pose), gradient = jax.value_and_grad(loss_aux, has_aux=True)(c)
        return per_pose, gradient

    forward_jit = jax.jit(energy)
    terms_jit = jax.jit(weighted_terms)
    vg_jit = jax.jit(vg_impl)

    compile_start = time.perf_counter()
    total = forward_jit(coords)
    terms = terms_jit(coords)
    total_vg, gradient = vg_jit(coords)
    _jax_block((total, terms, total_vg, gradient), jax)
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    total_host_for_consistency = np.asarray(total)
    total_vg_host_for_consistency = np.asarray(total_vg)
    consistency_delta = np.abs(
        total_host_for_consistency - total_vg_host_for_consistency
    )
    consistency_allowed = 5e-4 + 2e-6 * np.abs(total_host_for_consistency)
    consistency_normalized = consistency_delta / consistency_allowed
    worst_abs = int(np.argmax(consistency_delta))
    worst_normalized = int(np.argmax(consistency_normalized))
    self_consistency = {
        "pass": bool(np.all(consistency_delta <= consistency_allowed)),
        "max_abs": float(consistency_delta[worst_abs]),
        "max_abs_pose": worst_abs,
        "max_normalized_error": float(consistency_normalized[worst_normalized]),
        "worst_normalized_pose": worst_normalized,
        "atol": 5e-4,
        "rtol": 2e-6,
    }
    n_poses = int(coords_np.shape[0])
    timing = {
        "setup_ms": setup_ms,
        "compile_first_forward_terms_value_grad_ms": compile_ms,
        "forward_value_grad_self_consistency": self_consistency,
        "forward": _jax_timing(
            lambda: forward_jit(coords),
            warmup=warmup,
            reps=reps,
            n_poses=n_poses,
            jax=jax,
        ),
        "value_and_grad": _jax_timing(
            lambda: vg_jit(coords),
            warmup=warmup,
            reps=reps,
            n_poses=n_poses,
            jax=jax,
        ),
    }
    weighted_term_gradients, channel_grad_ms = _jax_weighted_term_gradients(
        weighted_terms, coords, jax=jax, jnp=jnp
    )
    timing["untimed_weighted_term_gradients_compile_and_run_ms"] = channel_grad_ms
    arrays = {
        "weighted_terms": np.ascontiguousarray(np.asarray(terms)),
        "weighted_term_gradients": weighted_term_gradients,
        "total": np.ascontiguousarray(np.asarray(total)),
        "value_and_grad_total": np.ascontiguousarray(np.asarray(total_vg)),
        "grad": np.ascontiguousarray(np.asarray(gradient)),
    }
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ContractError(f"jtmol {name} contains non-finite values")

    return arrays, timing


def _jtmol_score_same_topology(
    score_function: Any,
    pose_stack_one: Any,
    coords_np: np.ndarray,
    *,
    batch_size: int,
    warmup: int,
    reps: int,
    jax: Any,
    jnp: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Score K conformers with one topology/state and dense per-pose scoring."""

    from jtmol.score.score_types import ScoreType

    if pose_stack_one.coords.shape[0] != 1:
        raise ContractError("same-topology path expects a one-pose topology")
    if coords_np.ndim != 3 or coords_np.shape[1:] != tuple(pose_stack_one.coords.shape[1:]):
        raise ContractError(
            f"S2 fixture shape {coords_np.shape} incompatible with native "
            f"shape {tuple(pose_stack_one.coords.shape)}"
        )
    if coords_np.shape[0] % batch_size != 0:
        raise ContractError(
            f"S2 K={coords_np.shape[0]} must be divisible by batch size {batch_size}"
        )
    coords = jax.device_put(np.ascontiguousarray(coords_np, dtype=np.float32))

    setup_start = time.perf_counter()
    state = score_function.setup_for_pose_stack(pose_stack_one)
    _jax_block(state, jax)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    def one_energy(c: Any) -> Any:
        return score_function(
            pose_stack_one, state=state, coords_override=c[None]
        )[0].astype(jnp.float32)

    def one_terms(c: Any) -> Any:
        values = score_function.score_by_term(
            pose_stack_one, state=state, coords_override=c[None]
        )
        return jnp.stack(
            [values[ScoreType[name]][0] for name in OFFICIAL_NAMES]
        ).astype(jnp.float32)

    def energy(c: Any) -> Any:
        return jax.lax.map(one_energy, c, batch_size=batch_size)

    def weighted_terms(c: Any) -> Any:
        return jax.lax.map(one_terms, c, batch_size=batch_size).T

    def vg_impl(c: Any) -> tuple[Any, Any]:
        def loss_aux(x: Any) -> tuple[Any, Any]:
            per_pose = energy(x)
            return jnp.sum(per_pose), per_pose

        (_, per_pose), gradient = jax.value_and_grad(loss_aux, has_aux=True)(c)
        return per_pose, gradient

    forward_jit = jax.jit(energy)
    terms_jit = jax.jit(weighted_terms)
    vg_jit = jax.jit(vg_impl)

    compile_start = time.perf_counter()
    total = forward_jit(coords)
    terms = terms_jit(coords)
    total_vg, gradient = vg_jit(coords)
    _jax_block((total, terms, total_vg, gradient), jax)
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    total_host_for_consistency = np.asarray(total)
    total_vg_host_for_consistency = np.asarray(total_vg)
    consistency_delta = np.abs(
        total_host_for_consistency - total_vg_host_for_consistency
    )
    consistency_allowed = 5e-4 + 2e-6 * np.abs(total_host_for_consistency)
    consistency_normalized = consistency_delta / consistency_allowed
    worst_abs = int(np.argmax(consistency_delta))
    worst_normalized = int(np.argmax(consistency_normalized))
    self_consistency = {
        "pass": bool(np.all(consistency_delta <= consistency_allowed)),
        "max_abs": float(consistency_delta[worst_abs]),
        "max_abs_pose": worst_abs,
        "max_normalized_error": float(consistency_normalized[worst_normalized]),
        "worst_normalized_pose": worst_normalized,
        "atol": 5e-4,
        "rtol": 2e-6,
    }
    n_poses = int(coords_np.shape[0])
    timing = {
        "setup_ms": setup_ms,
        "compile_first_forward_terms_value_grad_ms": compile_ms,
        "forward_value_grad_self_consistency": self_consistency,
        "dense_same_topology_batch_size": batch_size,
        "forward": _jax_timing(
            lambda: forward_jit(coords),
            warmup=warmup,
            reps=reps,
            n_poses=n_poses,
            jax=jax,
        ),
        "value_and_grad": _jax_timing(
            lambda: vg_jit(coords),
            warmup=warmup,
            reps=reps,
            n_poses=n_poses,
            jax=jax,
        ),
    }
    weighted_term_gradients, channel_grad_ms = _jax_weighted_term_gradients(
        weighted_terms, coords, jax=jax, jnp=jnp
    )
    timing["untimed_weighted_term_gradients_compile_and_run_ms"] = channel_grad_ms
    arrays = {
        "weighted_terms": np.ascontiguousarray(np.asarray(terms)),
        "weighted_term_gradients": weighted_term_gradients,
        "total": np.ascontiguousarray(np.asarray(total)),
        "value_and_grad_total": np.ascontiguousarray(np.asarray(total_vg)),
        "grad": np.ascontiguousarray(np.asarray(gradient)),
    }
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ContractError(f"jtmol S2 {name} contains non-finite values")

    return arrays, timing


def _jtmol_score_same_topology_host_stream(
    score_function: Any,
    pose_stack_one: Any,
    coords_np: np.ndarray,
    *,
    warmup: int,
    reps: int,
    jax: Any,
    jnp: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Exact K-pose workload via K precompiled single-pose dispatches.

    This is an explicit diagnostic execution strategy, not a native batched
    kernel.  It exists because reverse-mode through the dense K=128 ``lax.map``
    materializes a >37 GB argument surface even at map batch size one on 1UBQ.
    We time the whole 128-dispatch workload and label it as such; it is never
    presented as a fused/batched implementation.
    """

    from jtmol.score.score_types import ScoreType

    if pose_stack_one.coords.shape[0] != 1:
        raise ContractError("host-stream path expects a one-pose topology")
    if coords_np.ndim != 3 or coords_np.shape[1:] != tuple(
        pose_stack_one.coords.shape[1:]
    ):
        raise ContractError(
            f"S2 fixture shape {coords_np.shape} incompatible with native "
            f"shape {tuple(pose_stack_one.coords.shape)}"
        )

    pose_coords = tuple(
        jax.device_put(np.ascontiguousarray(coords_np[i : i + 1], dtype=np.float32))
        for i in range(coords_np.shape[0])
    )

    setup_start = time.perf_counter()
    state = score_function.setup_for_pose_stack(pose_stack_one)
    _jax_block(state, jax)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    def energy_one(c: Any) -> Any:
        return score_function(
            pose_stack_one, state=state, coords_override=c
        ).astype(jnp.float32)

    def weighted_terms_one(c: Any) -> Any:
        values = score_function.score_by_term(
            pose_stack_one, state=state, coords_override=c
        )
        return jnp.stack(
            [values[ScoreType[name]] for name in OFFICIAL_NAMES], axis=0
        ).astype(jnp.float32)

    def vg_one(c: Any) -> tuple[Any, Any]:
        def loss_aux(x: Any) -> tuple[Any, Any]:
            per_pose = energy_one(x)
            return jnp.sum(per_pose), per_pose

        (_, per_pose), gradient = jax.value_and_grad(loss_aux, has_aux=True)(c)
        return per_pose, gradient

    def channel_grad_one(c: Any, row: Any) -> Any:
        def loss(x: Any) -> Any:
            terms = weighted_terms_one(x)
            selected = jax.lax.dynamic_index_in_dim(
                terms, row, axis=0, keepdims=False
            )
            return jnp.sum(selected)

        return jax.grad(loss)(c)

    forward_jit = jax.jit(energy_one)
    terms_jit = jax.jit(weighted_terms_one)
    vg_jit = jax.jit(vg_one)
    channel_grad_jit = jax.jit(channel_grad_one)

    compile_start = time.perf_counter()
    compiled_values = (
        forward_jit(pose_coords[0]),
        terms_jit(pose_coords[0]),
        vg_jit(pose_coords[0]),
        channel_grad_jit(pose_coords[0], jnp.asarray(0, dtype=jnp.int32)),
    )
    _jax_block(compiled_values, jax)
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    def forward_batch() -> list[Any]:
        return [forward_jit(c) for c in pose_coords]

    def vg_batch() -> list[Any]:
        return [vg_jit(c) for c in pose_coords]

    timing = {
        "setup_ms": setup_ms,
        "compile_single_pose_forward_terms_value_grad_channel_grad_ms": compile_ms,
        "execution_mode": "exact_host_stream_of_single_pose_jits",
        "dispatches_per_batch": len(pose_coords),
        "forward": _jax_timing(
            forward_batch,
            warmup=warmup,
            reps=reps,
            n_poses=len(pose_coords),
            jax=jax,
        ),
        "value_and_grad": _jax_timing(
            vg_batch,
            warmup=warmup,
            reps=reps,
            n_poses=len(pose_coords),
            jax=jax,
        ),
    }

    totals_device = forward_batch()
    terms_device = [terms_jit(c) for c in pose_coords]
    vg_device = vg_batch()
    _jax_block((totals_device, terms_device, vg_device), jax)
    total = np.ascontiguousarray(
        np.concatenate([np.asarray(x) for x in totals_device], axis=0)
    )
    terms = np.ascontiguousarray(
        np.concatenate([np.asarray(x) for x in terms_device], axis=1)
    )
    total_vg = np.ascontiguousarray(
        np.concatenate([np.asarray(x[0]) for x in vg_device], axis=0)
    )
    gradient = np.ascontiguousarray(
        np.concatenate([np.asarray(x[1]) for x in vg_device], axis=0)
    )

    consistency_delta = np.abs(total - total_vg)
    consistency_allowed = 5e-4 + 2e-6 * np.abs(total)
    consistency_normalized = consistency_delta / consistency_allowed
    worst_abs = int(np.argmax(consistency_delta))
    worst_normalized = int(np.argmax(consistency_normalized))
    timing["forward_value_grad_self_consistency"] = {
        "pass": bool(np.all(consistency_delta <= consistency_allowed)),
        "max_abs": float(consistency_delta[worst_abs]),
        "max_abs_pose": worst_abs,
        "max_normalized_error": float(consistency_normalized[worst_normalized]),
        "worst_normalized_pose": worst_normalized,
        "atol": 5e-4,
        "rtol": 2e-6,
    }

    channel_start = time.perf_counter()
    channel_rows: list[np.ndarray] = []
    for row in range(len(OFFICIAL_NAMES)):
        row_device = [
            channel_grad_jit(c, jnp.asarray(row, dtype=jnp.int32))
            for c in pose_coords
        ]
        _jax_block(row_device, jax)
        channel_rows.append(
            np.ascontiguousarray(
                np.concatenate([np.asarray(x) for x in row_device], axis=0)
            )
        )
    timing["untimed_weighted_term_gradients_ms"] = (
        time.perf_counter() - channel_start
    ) * 1000.0
    weighted_term_gradients = np.ascontiguousarray(np.stack(channel_rows, axis=0))

    arrays = {
        "weighted_terms": terms,
        "weighted_term_gradients": weighted_term_gradients,
        "total": total,
        "value_and_grad_total": total_vg,
        "grad": gradient,
    }
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ContractError(f"jtmol S2 host-stream {name} contains non-finite values")
    return arrays, timing


def _build_jtmol_native_stack(
    singles: Sequence[Any], coords_np: np.ndarray, *, jnp: Any
) -> Any:
    """Pad native jtmol PoseStack fields without changing their topology."""

    from jtmol.chemical.constants import MAX_SIG_BOND_SEPARATION
    from jtmol.pose.pose_stack import PoseStack

    if len(singles) != coords_np.shape[0]:
        raise ContractError("number of S3 singletons does not match fixture K")
    pbt = singles[0].packed_block_types
    if any(ps.packed_block_types is not pbt for ps in singles):
        raise ContractError("all jtmol S3 poses must share one PackedBlockTypes")

    n_poses = len(singles)
    max_n_atoms = int(coords_np.shape[1])
    max_n_blocks = max(int(ps.block_type_ind.shape[1]) for ps in singles)
    max_n_conn = max(int(ps.inter_residue_connections.shape[2]) for ps in singles)

    block_type_ind = np.full((n_poses, max_n_blocks), -1, dtype=np.int32)
    block_coord_offset = np.zeros((n_poses, max_n_blocks), dtype=np.int32)
    inter_residue_connections = np.full(
        (n_poses, max_n_blocks, max_n_conn, 2), -1, dtype=np.int32
    )
    inter_block_bondsep = np.full(
        (n_poses, max_n_blocks, max_n_blocks, max_n_conn, max_n_conn),
        int(MAX_SIG_BOND_SEPARATION),
        dtype=np.int32,
    )

    for pose_ind, ps in enumerate(singles):
        bti = _as_numpy(ps.block_type_ind)[0].astype(np.int32, copy=False)
        bco = _as_numpy(ps.block_coord_offset)[0].astype(np.int32, copy=False)
        irc = _as_numpy(ps.inter_residue_connections)[0].astype(np.int32, copy=False)
        ibs = _as_numpy(ps.inter_block_bondsep)[0].astype(np.int32, copy=False)
        n_blocks = bti.shape[0]
        n_conn = irc.shape[1]
        block_type_ind[pose_ind, :n_blocks] = bti
        block_coord_offset[pose_ind, :n_blocks] = bco
        inter_residue_connections[pose_ind, :n_blocks, :n_conn] = irc
        inter_block_bondsep[
            pose_ind, :n_blocks, :n_blocks, :n_conn, :n_conn
        ] = ibs

    return PoseStack(
        packed_block_types=pbt,
        coords=jnp.asarray(coords_np, dtype=jnp.float32),
        block_coord_offset=jnp.asarray(block_coord_offset, dtype=jnp.int32),
        block_coord_offset64=jnp.asarray(block_coord_offset, dtype=jnp.int64),
        inter_residue_connections=jnp.asarray(
            inter_residue_connections, dtype=jnp.int32
        ),
        inter_block_bondsep=jnp.asarray(inter_block_bondsep, dtype=jnp.int32),
        block_type_ind=jnp.asarray(block_type_ind, dtype=jnp.int32),
        pdb_info=None,
    )


def run_jtmol(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path, args.pdb_dir)
    upstream_dir = Path(args.upstream_dir).resolve()
    upstream_meta_path = upstream_dir / METADATA_FILENAME
    upstream_arrays_path = upstream_dir / ARRAYS_FILENAME
    if not upstream_meta_path.is_file() or not upstream_arrays_path.is_file():
        raise ContractError(f"upstream artifacts missing from {upstream_dir}")
    upstream_meta = json.loads(upstream_meta_path.read_text())
    upstream_arrays = np.load(upstream_arrays_path, allow_pickle=False)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    active_scenarios = (
        ("S1",) if args.s1_only else (("S1", "S2") if args.skip_s3 else SCENARIOS)
    )
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backend": "jtmol",
        "status": "running",
        "scenarios_requested": list(active_scenarios),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_path": manifest["manifest_path"],
        "pdb_dir": manifest["pdb_dir"],
        "benchmark_scope": manifest["raw"].get("benchmark_scope"),
        "source_upstream_contract_sha256": _file_hash(upstream_meta_path),
        "source_upstream_arrays_sha256": _file_hash(upstream_arrays_path),
        "official_weights": OFFICIAL_WEIGHTS,
        "official_channel_order": list(OFFICIAL_NAMES),
        "known_exceptions": {"lk_bridge_uncpl": KNOWN_LK_BRIDGE_UNCPL},
        "timing_contract": {
            "forward": "coords -> weighted E[n_poses]",
            "value_and_grad": "coords -> (E[n_poses], d(sum(E))/dcoords)",
            "backend_execution": (
                "S1 jax.jit dense scorer; S2 exact host stream of 128 "
                "single-pose jits (diagnostic, not native batch)"
                if "S2" in active_scenarios and args.s2_execution == "host_stream"
                else "jax.jit compiled dense scorer"
            ),
            "warmup": args.warmup,
            "reps": args.reps,
            "synchronized": True,
            "setup_and_compilation_excluded": True,
            "strict_dense_streaming_path": True,
        },
        "fixtures": {},
        "scenarios": {},
        "representation_contract": {
            "execution_layout_match_required": True,
            "exact_representation_pass": True,
            "per_pose_differences": [],
        },
    }
    arrays: dict[str, np.ndarray] = {}

    try:
        if upstream_meta.get("schema_version") != SCHEMA_VERSION:
            raise ContractError("upstream artifact schema version mismatch")
        if upstream_meta.get("backend") != "upstream":
            raise ContractError("source artifact is not an upstream run")
        if upstream_meta.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ContractError("manifest hash differs from upstream fixture manifest")
        if upstream_meta.get("official_weights") != OFFICIAL_WEIGHTS:
            raise ContractError("upstream artifact does not use official 22 weights")

        for scenario in active_scenarios:
            if scenario not in upstream_meta.get("fixtures", {}):
                raise ContractError(f"upstream fixture metadata missing for {scenario}")
            fixture_meta = upstream_meta["fixtures"][scenario]
            fixture = _validate_saved_fixture(fixture_meta, upstream_arrays)
            arrays[f"{scenario}_coords"] = fixture
            metadata["fixtures"][scenario] = fixture_meta

        # These must be set before importing JAX/jtmol.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["JAX_ENABLE_X64"] = "1"
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
        import jax
        import jax.numpy as jnp
        try:
            import jtmol
        except ImportError:
            # Running ``python benchmarks/exact_upstream_matrix.py`` sets
            # sys.path[0] to benchmarks/, not the repository root.  Add the
            # root only in this jtmol-only subprocess; doing so in upstream
            # mode could shadow the official wheel with the gitignored clone.
            repo_root = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(repo_root))
            import jtmol

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        metadata["environment"] = _jtmol_environment(jax, jtmol)

        s1 = jtmol.pose_stack_from_pdb(str(manifest["s1_path"]))
        s1_actual = _pose_contract(
            s1, pdb_path=manifest["s1_path"], identifier=manifest["s1_id"]
        )
        s1_expected = upstream_meta["fixtures"]["S1"]["per_pose"][0]
        _assert_contract_equal(s1_expected, s1_actual)
        s1_representation_differences = _representation_contract_differences(
            s1_expected, s1_actual
        )
        if s1_representation_differences:
            metadata["representation_contract"]["exact_representation_pass"] = False
            metadata["representation_contract"]["per_pose_differences"].append(
                {
                    "scenario": "S1/S2",
                    "id": s1_actual["id"],
                    "differences": s1_representation_differences,
                }
            )
        s1_coords = arrays["S1_coords"]
        s2_coords = arrays.get("S2_coords")
        if tuple(s1.coords.shape) != tuple(s1_coords.shape):
            raise ContractError("jtmol S1 native shape differs from upstream fixture")
        s1_native_max_abs = _assert_native_coords_close(
            s1_coords[0, : int(s1_actual["n_atoms"])],
            _as_numpy(s1.coords)[0, : int(s1_actual["n_atoms"])],
            identifier=manifest["s1_id"],
        )
        metadata["native_coordinate_validation"] = {
            "atol": NATIVE_COORD_ATOL,
            "pass": True,
            "S1_max_abs": s1_native_max_abs,
            "failures": [],
        }

        for scenario in tuple(x for x in ("S1", "S2") if x in active_scenarios):
            try:
                score_function = _jtmol_score_function(jtmol)
                if scenario == "S1":
                    result, timing = _jtmol_score_direct(
                        score_function,
                        s1,
                        s1_coords,
                        warmup=args.warmup,
                        reps=args.reps,
                        jax=jax,
                        jnp=jnp,
                    )
                    n_poses = 1
                else:
                    assert s2_coords is not None
                    if args.s2_execution == "host_stream":
                        result, timing = _jtmol_score_same_topology_host_stream(
                            score_function,
                            s1,
                            s2_coords,
                            warmup=args.warmup,
                            reps=args.reps,
                            jax=jax,
                            jnp=jnp,
                        )
                    else:
                        result, timing = _jtmol_score_same_topology(
                            score_function,
                            s1,
                            s2_coords,
                            batch_size=args.s2_batch_size,
                            warmup=args.warmup,
                            reps=args.reps,
                            jax=jax,
                            jnp=jnp,
                        )
                    n_poses = 128
                for name, value in result.items():
                    arrays[f"{scenario}_{name}"] = value
                metadata["scenarios"][scenario] = {
                    "status": "ok",
                    "n_poses": n_poses,
                    "padded_n_atoms": int(s1_coords.shape[1]),
                    "dense_path": True,
                    "timing": timing,
                    "array_keys": {
                        name: f"{scenario}_{name}" for name in result
                    },
                }
            except BaseException as exc:
                metadata["scenarios"][scenario] = _failure("score_and_time", exc)

        # Do not let cached S1/S2 executables turn an otherwise valid native
        # S3 attempt into an allocator-artifact OOM.
        del s1
        gc.collect()
        jax.clear_caches()

        if "S3" not in active_scenarios:
            metadata["status"] = (
                "ok"
                if all(
                    metadata["scenarios"].get(s, {}).get("status") == "ok"
                    for s in active_scenarios
                )
                else "failed"
            )
            _save_npz(out_dir / ARRAYS_FILENAME, arrays)
            _write_json(out_dir / METADATA_FILENAME, metadata)
            print(json.dumps({"status": metadata["status"], "out_dir": str(out_dir)}))
            return 0 if metadata["status"] == "ok" else 1

        # S3 must be one native padded heterogeneous PoseStack.  There is no
        # chunk fallback in this strict matrix.
        try:
            s3_singles = [
                jtmol.pose_stack_from_pdb(str(path)) for path in manifest["s3_paths"]
            ]
            s3_expected = upstream_meta["fixtures"]["S3"]["per_pose"]
            s3_actual = [
                _pose_contract(ps, pdb_path=path, identifier=identifier)
                for ps, path, identifier in zip(
                    s3_singles, manifest["s3_paths"], manifest["s3_ids"]
                )
            ]
            for expected, actual in zip(s3_expected, s3_actual):
                _assert_contract_equal(expected, actual)
                representation_differences = _representation_contract_differences(
                    expected, actual
                )
                if representation_differences:
                    metadata["representation_contract"][
                        "exact_representation_pass"
                    ] = False
                    metadata["representation_contract"][
                        "per_pose_differences"
                    ].append(
                        {
                            "scenario": "S3",
                            "id": actual["id"],
                            "differences": representation_differences,
                        }
                    )
            topology_hashes = [x["topology_sha256"] for x in s3_actual]
            if len(set(topology_hashes)) != 128:
                raise ContractError("jtmol S3 does not contain 128 unique topologies")

            s3_coords = arrays["S3_coords"]
            for pose_ind, contract in enumerate(s3_actual):
                native = _as_numpy(s3_singles[pose_ind].coords)[
                    0, : int(contract["n_atoms"])
                ].astype(np.float32, copy=False)
                fixture_native = s3_coords[pose_ind, : int(contract["n_atoms"])]
                max_abs = _native_coord_max_abs(
                    fixture_native,
                    native,
                    identifier=f"S3[{pose_ind}]/{contract['id']}",
                )
                metadata["native_coordinate_validation"].setdefault(
                    "S3_max_abs_by_pose", []
                ).append(max_abs)
                if max_abs > NATIVE_COORD_ATOL:
                    metadata["native_coordinate_validation"]["pass"] = False
                    metadata["native_coordinate_validation"]["failures"].append(
                        {
                            "pose_index": pose_ind,
                            "id": contract["id"],
                            "max_abs": max_abs,
                        }
                    )

            s3 = _build_jtmol_native_stack(s3_singles, s3_coords, jnp=jnp)
            del s3_singles
            gc.collect()
            score_function = _jtmol_score_function(jtmol)
            result, timing = _jtmol_score_direct(
                score_function,
                s3,
                s3_coords,
                warmup=args.warmup,
                reps=args.reps,
                jax=jax,
                jnp=jnp,
            )
            for name, value in result.items():
                arrays[f"S3_{name}"] = value
            metadata["scenarios"]["S3"] = {
                "status": "ok",
                "n_poses": 128,
                "padded_n_atoms": int(s3_coords.shape[1]),
                "dense_path": True,
                "native_single_batch": True,
                "chunk_fallback_attempted": False,
                "timing": timing,
                "array_keys": {name: f"S3_{name}" for name in result},
            }
        except BaseException as exc:
            failure = _failure("native_128_dense_score_and_time", exc)
            failure["dense_path"] = True
            failure["native_single_batch"] = True
            failure["chunk_fallback_attempted"] = False
            metadata["scenarios"]["S3"] = failure

        metadata["status"] = (
            "ok"
            if all(
                metadata["scenarios"].get(s, {}).get("status") == "ok"
                for s in active_scenarios
            )
            else "failed"
        )
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["fatal_error"] = _failure("jtmol_contract_or_setup", exc)

    _save_npz(out_dir / ARRAYS_FILENAME, arrays)
    _write_json(out_dir / METADATA_FILENAME, metadata)
    print(json.dumps({"status": metadata["status"], "out_dir": str(out_dir)}))
    return 0 if metadata["status"] == "ok" else 1


def _array_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        return {
            "pass": False,
            "shape_mismatch": {"upstream": list(ref.shape), "jtmol": list(cand.shape)},
        }
    finite = bool(np.all(np.isfinite(ref)) and np.all(np.isfinite(cand)))
    diff = np.abs(cand - ref)
    allowed = atol + rtol * np.abs(ref)
    violations = diff > allowed
    rel_l2 = float(np.linalg.norm(cand - ref) / max(np.linalg.norm(ref), 1e-30))
    return {
        "pass": finite and not bool(np.any(violations)),
        "finite": finite,
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs": float(np.mean(diff)) if diff.size else 0.0,
        "relative_l2": rel_l2,
        "n_violations": int(np.count_nonzero(violations)),
        "n_values": int(diff.size),
        "atol": atol,
        "rtol": rtol,
    }


def _per_pose_gradient_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    pose_contracts: Sequence[Mapping[str, Any]],
    *,
    atol: float,
    rtol: float,
    relative_l2_limit: float,
) -> dict[str, Any]:
    """Prevent large poses/channels from diluting a bad per-pose gradient."""

    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    if ref.shape != cand.shape or ref.ndim != 3:
        return {
            "pass": False,
            "shape_mismatch": {
                "upstream": list(ref.shape),
                "jtmol": list(cand.shape),
            },
        }
    if len(pose_contracts) != ref.shape[0]:
        return {
            "pass": False,
            "pose_count_mismatch": {
                "contracts": len(pose_contracts),
                "gradient": ref.shape[0],
            },
        }

    failures: list[dict[str, Any]] = []
    n_failed = 0
    worst: dict[str, Any] | None = None
    for pose_ind, contract in enumerate(pose_contracts):
        n_atoms = int(contract["n_atoms"])
        metric = _array_metrics(
            ref[pose_ind, :n_atoms],
            cand[pose_ind, :n_atoms],
            atol=atol,
            rtol=rtol,
        )
        metric["relative_l2_limit"] = relative_l2_limit
        metric["pass"] = bool(
            metric.get("pass", False)
            and metric.get("relative_l2", math.inf) <= relative_l2_limit
        )
        compact = {
            "pose_index": pose_ind,
            "id": contract.get("id"),
            "n_atoms": n_atoms,
            "pass": metric["pass"],
            "max_abs": metric.get("max_abs"),
            "relative_l2": metric.get("relative_l2"),
            "n_violations": metric.get("n_violations"),
        }
        if worst is None or float(compact["relative_l2"] or 0.0) > float(
            worst["relative_l2"] or 0.0
        ):
            worst = compact
        if not metric["pass"]:
            n_failed += 1
            if len(failures) < 32:
                failures.append(compact)

    return {
        "pass": n_failed == 0,
        "relative_l2_limit": relative_l2_limit,
        "n_failed_poses": n_failed,
        "worst_pose": worst,
        "failed_poses": failures,
        "failed_poses_truncated": n_failed > len(failures),
    }


def _normalized_gpu_name(value: Any) -> str:
    name = " ".join(str(value).casefold().split())
    return name.removeprefix("nvidia ")


def _validated_timing_median(
    summary: Mapping[str, Any], *, reps: int, n_poses: int, label: str
) -> float:
    samples = np.asarray(summary.get("samples_ms_batch", []), dtype=np.float64)
    if samples.shape != (reps,):
        raise ContractError(f"{label}: expected {reps} timing samples")
    if not np.all(np.isfinite(samples)) or np.any(samples <= 0.0):
        raise ContractError(f"{label}: timing samples must be finite and positive")
    median = float(np.median(samples))
    stored_median = float(summary.get("median_ms_batch", math.nan))
    stored_per_pose = float(summary.get("median_ms_per_pose", math.nan))
    if not math.isclose(median, stored_median, rel_tol=1e-12, abs_tol=1e-12):
        raise ContractError(f"{label}: stored timing median is inconsistent")
    if not math.isclose(
        median / n_poses, stored_per_pose, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ContractError(f"{label}: stored per-pose timing is inconsistent")
    return median


def run_compare(args: argparse.Namespace) -> int:
    upstream_dir = Path(args.upstream_dir).resolve()
    jtmol_dir = Path(args.jtmol_dir).resolve()
    output = Path(args.output).resolve()
    up_meta_path = upstream_dir / METADATA_FILENAME
    jt_meta_path = jtmol_dir / METADATA_FILENAME
    up_arrays_path = upstream_dir / ARRAYS_FILENAME
    jt_arrays_path = jtmol_dir / ARRAYS_FILENAME

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "official_weights": OFFICIAL_WEIGHTS,
        "official_channel_order": list(OFFICIAL_NAMES),
        "known_exceptions": {"lk_bridge_uncpl": KNOWN_LK_BRIDGE_UNCPL},
        "certification_tolerance_ceilings": CERTIFICATION_TOLERANCES,
        "speed_ratio_definition": "upstream median / jtmol median; >1 means jtmol faster",
        "timing_certification": {
            "outer_trials": 1,
            "backend_order": ["upstream", "jtmol"],
            "confidence_interval": None,
            "decisive_ten_x_claim_certified": False,
            "reason": "A decisive claim requires fresh-process ABBA/randomized outer trials and a lower 95% confidence bound above 10x.",
        },
        "contract_errors": [],
        "representation_contract": None,
        "native_coordinate_validation": None,
        "scenarios": {},
    }

    try:
        requested_tolerances = {
            key: float(getattr(args, key)) for key in CERTIFICATION_TOLERANCES
        }
        invalid_tolerances = {
            key: {
                "requested": requested,
                "certification_ceiling": CERTIFICATION_TOLERANCES[key],
            }
            for key, requested in requested_tolerances.items()
            if not math.isfinite(requested)
            or requested < 0.0
            or requested > CERTIFICATION_TOLERANCES[key]
        }
        if invalid_tolerances:
            raise ContractError(
                "comparison tolerances are invalid or looser than the pinned "
                f"certification ceilings: {invalid_tolerances}"
            )
        report["requested_tolerances"] = requested_tolerances
        for path in (up_meta_path, jt_meta_path, up_arrays_path, jt_arrays_path):
            if not path.is_file():
                raise ContractError(f"artifact missing: {path}")
        up_meta = json.loads(up_meta_path.read_text())
        jt_meta = json.loads(jt_meta_path.read_text())
        up_arrays = np.load(up_arrays_path, allow_pickle=False)
        jt_arrays = np.load(jt_arrays_path, allow_pickle=False)

        if up_meta.get("backend") != "upstream" or jt_meta.get("backend") != "jtmol":
            raise ContractError("backend metadata labels are invalid")
        active_scenarios = tuple(up_meta.get("scenarios_requested", ()))
        if active_scenarios not in (SCENARIOS, ("S1", "S2"), ("S1",)):
            raise ContractError(
                f"invalid upstream scenario request: {active_scenarios}"
            )
        if tuple(jt_meta.get("scenarios_requested", ())) != active_scenarios:
            raise ContractError("upstream and jtmol scenario requests differ")
        report["scenarios_requested"] = list(active_scenarios)
        complete_matrix = active_scenarios == SCENARIOS
        report["complete_three_scenario_matrix"] = complete_matrix
        if (
            up_meta.get("schema_version") != SCHEMA_VERSION
            or jt_meta.get("schema_version") != SCHEMA_VERSION
        ):
            raise ContractError("artifact schema mismatch")
        if up_meta.get("manifest_sha256") != jt_meta.get("manifest_sha256"):
            raise ContractError("upstream and jtmol manifests differ")
        if up_meta.get("official_weights") != OFFICIAL_WEIGHTS:
            raise ContractError("upstream artifact weights are not official beta2016")
        if jt_meta.get("official_weights") != OFFICIAL_WEIGHTS:
            raise ContractError("jtmol artifact weights are not official beta2016")
        if jt_meta.get("source_upstream_contract_sha256") != _file_hash(up_meta_path):
            raise ContractError("jtmol was produced from a different upstream contract.json")
        if jt_meta.get("source_upstream_arrays_sha256") != _file_hash(up_arrays_path):
            raise ContractError("jtmol was produced from a different upstream data.npz")
        if up_meta.get("fixtures") != jt_meta.get("fixtures"):
            raise ContractError("fixture metadata differs between backends")
        if up_meta.get("benchmark_scope") != jt_meta.get("benchmark_scope"):
            raise ContractError("benchmark-scope metadata differs between backends")

        benchmark_scope = jt_meta.get("benchmark_scope")
        if not isinstance(benchmark_scope, dict):
            benchmark_scope = {
                "panel_role": "unspecified",
                "conditioned_on_jtmol_compatibility": True,
                "general_no_compromise_certification": False,
            }
        report["benchmark_scope"] = benchmark_scope
        # A manifest assertion is not evidence of general force-field parity.
        # Until parameter-table hashes plus ligand/noncanonical and displaced-
        # coordinate coverage are implemented, this harness may certify only
        # the exact measured panel, never a general no-compromise claim.
        general_panel_pass = False
        report["general_panel_certification"] = {
            "pass": False,
            "reason": (
                "parameter-table, ligand/noncanonical, and displaced-coordinate "
                "coverage gates are not yet implemented"
            ),
        }

        representation_contract = jt_meta.get("representation_contract")
        if not isinstance(representation_contract, dict):
            raise ContractError("jtmol representation-contract metadata is missing")
        report["representation_contract"] = representation_contract
        representation_contract["scope"] = (
            "bond order, ringness, and connection-type metadata; scoring parameter "
            "tables require separate coverage"
        )
        exact_representation_pass = bool(
            representation_contract.get("exact_representation_pass")
        )
        native_coordinate_validation = jt_meta.get("native_coordinate_validation")
        if not isinstance(native_coordinate_validation, dict):
            raise ContractError("jtmol native-coordinate validation metadata is missing")
        report["native_coordinate_validation"] = native_coordinate_validation
        native_coordinate_pass = bool(native_coordinate_validation.get("pass"))

        timing_keys = (
            "forward",
            "value_and_grad",
            "warmup",
            "reps",
            "synchronized",
            "setup_and_compilation_excluded",
        )
        up_timing_contract = up_meta.get("timing_contract", {})
        jt_timing_contract = jt_meta.get("timing_contract", {})
        timing_mismatches = {
            key: {
                "upstream": up_timing_contract.get(key),
                "jtmol": jt_timing_contract.get(key),
            }
            for key in timing_keys
            if up_timing_contract.get(key) != jt_timing_contract.get(key)
        }
        if timing_mismatches:
            raise ContractError(f"timing API/settings differ: {timing_mismatches}")

        up_environment = up_meta.get("environment", {})
        jt_environment = jt_meta.get("environment", {})
        up_gpu = _normalized_gpu_name(up_environment.get("gpu", ""))
        jt_gpu = _normalized_gpu_name(jt_environment.get("gpu", ""))
        if not up_gpu or not jt_gpu or up_gpu != jt_gpu:
            raise ContractError(
                "performance artifacts were not measured on the same GPU: "
                f"upstream={up_environment.get('gpu')!r}, "
                f"jtmol={jt_environment.get('gpu')!r}"
            )
        up_gpu_context = up_environment.get("physical_gpu_context")
        jt_gpu_context = jt_environment.get("physical_gpu_context")
        if not isinstance(up_gpu_context, dict) or not isinstance(
            jt_gpu_context, dict
        ):
            raise ContractError("physical GPU identity metadata is missing")
        if up_gpu_context != jt_gpu_context:
            raise ContractError(
                "performance artifacts were not measured in the same physical "
                "GPU allocation"
            )

        # Fixture arrays must be byte-identical, not merely numerically close.
        expected_k = {"S1": 1, "S2": 128, "S3": 128}
        for scenario in active_scenarios:
            key = f"{scenario}_coords"
            fixture_meta = up_meta["fixtures"][scenario]
            if fixture_meta.get("array_key") != key:
                raise ContractError(f"{scenario} fixture has unexpected array key")
            up_coords = _validate_saved_fixture(fixture_meta, up_arrays)
            jt_coords = _validate_saved_fixture(fixture_meta, jt_arrays)
            if up_coords.shape[0] != expected_k[scenario]:
                raise ContractError(
                    f"{scenario} must have K={expected_k[scenario]}, "
                    f"found K={up_coords.shape[0]}"
                )
            if _array_hash(up_coords) != _array_hash(jt_coords):
                raise ContractError(f"{scenario} coordinate bytes differ")

        report["manifest_sha256"] = up_meta["manifest_sha256"]
        report["upstream_environment"] = up_meta.get("environment")
        report["jtmol_environment"] = jt_meta.get("environment")

        # Scenario output/timing results remain useful on this exact canonical
        # panel, but the overall no-compromise certification cannot pass while
        # richer upstream chemical metadata is absent.
        all_pass = (
            exact_representation_pass
            and native_coordinate_pass
            and general_panel_pass
            and complete_matrix
        )
        channel_coverage: dict[str, dict[str, Any]] = {
            name: {
                "upstream_energy_nonzero_values": 0,
                "upstream_gradient_nonzero_values": 0,
            }
            for name in OFFICIAL_NAMES
        }
        for scenario in active_scenarios:
            scenario_report: dict[str, Any] = {}
            up_status = up_meta.get("scenarios", {}).get(scenario, {})
            jt_status = jt_meta.get("scenarios", {}).get(scenario, {})
            if up_status.get("status") != "ok" or jt_status.get("status") != "ok":
                scenario_report.update(
                    {
                        "status": "failed",
                        "upstream_run": up_status,
                        "jtmol_run": jt_status,
                        "speed_ratios": None,
                    }
                )
                report["scenarios"][scenario] = scenario_report
                all_pass = False
                continue

            backend_self_consistency = {
                "upstream": up_status.get("timing", {}).get(
                    "forward_value_grad_self_consistency"
                ),
                "jtmol": jt_status.get("timing", {}).get(
                    "forward_value_grad_self_consistency"
                ),
            }
            self_consistency_pass = all(
                isinstance(metric, dict) and metric.get("pass") is True
                for metric in backend_self_consistency.values()
            )
            scenario_report["forward_value_grad_self_consistency"] = (
                backend_self_consistency
            )

            execution_mode = jt_status.get("timing", {}).get(
                "execution_mode", "native_single_batch"
            )
            native_batch_timing_pass = execution_mode != (
                "exact_host_stream_of_single_pose_jits"
            )
            scenario_report["jtmol_execution_mode"] = execution_mode
            scenario_report["native_batch_timing_pass"] = native_batch_timing_pass

            expected_n_poses = {"S1": 1, "S2": 128, "S3": 128}[scenario]
            expected_n_atoms = int(
                np.asarray(up_arrays[f"{scenario}_coords"]).shape[1]
            )
            status_errors: dict[str, Any] = {}
            for backend, status in (("upstream", up_status), ("jtmol", jt_status)):
                if status.get("n_poses") != expected_n_poses:
                    status_errors[f"{backend}_n_poses"] = status.get("n_poses")
                if status.get("padded_n_atoms") != expected_n_atoms:
                    status_errors[f"{backend}_padded_n_atoms"] = status.get(
                        "padded_n_atoms"
                    )
            if not jt_status.get("dense_path"):
                status_errors["jtmol_dense_path"] = jt_status.get("dense_path")
            if scenario == "S3":
                for backend, status in (("upstream", up_status), ("jtmol", jt_status)):
                    if not status.get("native_single_batch"):
                        status_errors[f"{backend}_native_single_batch"] = status.get(
                            "native_single_batch"
                        )
                if jt_status.get("chunk_fallback_attempted") is not False:
                    status_errors["jtmol_chunk_fallback_attempted"] = jt_status.get(
                        "chunk_fallback_attempted"
                    )
            if status_errors:
                scenario_report.update(
                    {
                        "status": "failed",
                        "scenario_contract_errors": status_errors,
                        "speed_ratios": None,
                    }
                )
                report["scenarios"][scenario] = scenario_report
                all_pass = False
                continue

            required = (
                "weighted_terms",
                "weighted_term_gradients",
                "total",
                "value_and_grad_total",
                "grad",
            )
            missing = [
                f"{scenario}_{name}"
                for name in required
                if f"{scenario}_{name}" not in up_arrays
                or f"{scenario}_{name}" not in jt_arrays
            ]
            if missing:
                scenario_report.update(
                    {"status": "failed", "missing_arrays": missing, "speed_ratios": None}
                )
                report["scenarios"][scenario] = scenario_report
                all_pass = False
                continue

            coords_shape = tuple(np.asarray(up_arrays[f"{scenario}_coords"]).shape)
            if len(coords_shape) != 3 or coords_shape[2] != 3:
                raise ContractError(f"{scenario} coordinate shape is invalid")
            n_poses = coords_shape[0]
            expected_shapes = {
                "weighted_terms": (22, n_poses),
                "weighted_term_gradients": (22,) + coords_shape,
                "total": (n_poses,),
                "value_and_grad_total": (n_poses,),
                "grad": coords_shape,
            }
            result_contract_errors: dict[str, Any] = {}
            for name, expected_shape in expected_shapes.items():
                up_value = np.asarray(up_arrays[f"{scenario}_{name}"])
                jt_value = np.asarray(jt_arrays[f"{scenario}_{name}"])
                if tuple(up_value.shape) != expected_shape:
                    result_contract_errors.setdefault(name, {})["upstream_shape"] = {
                        "expected": list(expected_shape),
                        "actual": list(up_value.shape),
                    }
                if tuple(jt_value.shape) != expected_shape:
                    result_contract_errors.setdefault(name, {})["jtmol_shape"] = {
                        "expected": list(expected_shape),
                        "actual": list(jt_value.shape),
                    }
                if up_value.dtype != np.float32:
                    result_contract_errors.setdefault(name, {})["upstream_dtype"] = str(
                        up_value.dtype
                    )
                if jt_value.dtype != np.float32:
                    result_contract_errors.setdefault(name, {})["jtmol_dtype"] = str(
                        jt_value.dtype
                    )
            if result_contract_errors:
                scenario_report.update(
                    {
                        "status": "failed",
                        "result_contract_errors": result_contract_errors,
                        "speed_ratios": None,
                    }
                )
                report["scenarios"][scenario] = scenario_report
                all_pass = False
                continue

            channel_metrics: dict[str, Any] = {}
            up_terms = np.asarray(up_arrays[f"{scenario}_weighted_terms"])
            jt_terms = np.asarray(jt_arrays[f"{scenario}_weighted_terms"])
            if up_terms.shape != jt_terms.shape or up_terms.shape[0] != 22:
                terms_pass = False
                channel_metrics["shape"] = {
                    "pass": False,
                    "upstream": list(up_terms.shape),
                    "jtmol": list(jt_terms.shape),
                }
            else:
                for row, name in enumerate(OFFICIAL_NAMES):
                    channel_coverage[name]["upstream_energy_nonzero_values"] += int(
                        np.count_nonzero(up_terms[row])
                    )
                    metric = _array_metrics(
                        up_terms[row],
                        jt_terms[row],
                        atol=args.energy_atol,
                        rtol=args.energy_rtol,
                    )
                    if name == "lk_bridge_uncpl" and not metric["pass"]:
                        metric["known_upstream_exception_note"] = KNOWN_LK_BRIDGE_UNCPL
                    channel_metrics[name] = metric
                terms_pass = all(x.get("pass", False) for x in channel_metrics.values())

            pose_contracts = up_meta["fixtures"][scenario]["per_pose"]
            up_term_gradients = np.asarray(
                up_arrays[f"{scenario}_weighted_term_gradients"]
            )
            jt_term_gradients = np.asarray(
                jt_arrays[f"{scenario}_weighted_term_gradients"]
            )
            channel_gradient_metrics: dict[str, Any] = {}
            for row, name in enumerate(OFFICIAL_NAMES):
                channel_coverage[name]["upstream_gradient_nonzero_values"] += int(
                    np.count_nonzero(up_term_gradients[row])
                )
                metric = _array_metrics(
                    up_term_gradients[row],
                    jt_term_gradients[row],
                    atol=args.grad_atol,
                    rtol=args.grad_rtol,
                )
                per_pose_metric = _per_pose_gradient_metrics(
                    up_term_gradients[row],
                    jt_term_gradients[row],
                    pose_contracts,
                    atol=args.grad_atol,
                    rtol=args.grad_rtol,
                    relative_l2_limit=args.grad_rel_l2,
                )
                metric["per_pose"] = per_pose_metric
                padding_violations_up = 0
                padding_violations_jt = 0
                for pose_ind, pose_contract in enumerate(pose_contracts):
                    n_atoms = int(pose_contract["n_atoms"])
                    padding_violations_up += int(
                        np.count_nonzero(up_term_gradients[row, pose_ind, n_atoms:])
                    )
                    padding_violations_jt += int(
                        np.count_nonzero(jt_term_gradients[row, pose_ind, n_atoms:])
                    )
                padding_pass = (
                    padding_violations_up == 0 and padding_violations_jt == 0
                )
                metric["padding_exact_zero"] = {
                    "pass": padding_pass,
                    "upstream_nonzero_values": padding_violations_up,
                    "jtmol_nonzero_values": padding_violations_jt,
                }
                metric["relative_l2_limit"] = args.grad_rel_l2
                metric["pass"] = bool(
                    metric.get("pass", False)
                    and metric.get("relative_l2", math.inf) <= args.grad_rel_l2
                    and per_pose_metric.get("pass", False)
                    and padding_pass
                )
                if name == "lk_bridge_uncpl" and not metric["pass"]:
                    metric["known_upstream_exception_note"] = KNOWN_LK_BRIDGE_UNCPL
                channel_gradient_metrics[name] = metric
            channel_gradients_pass = all(
                metric.get("pass", False)
                for metric in channel_gradient_metrics.values()
            )

            total_metrics = _array_metrics(
                up_arrays[f"{scenario}_total"],
                jt_arrays[f"{scenario}_total"],
                atol=args.total_atol,
                rtol=args.total_rtol,
            )
            vg_total_metrics = _array_metrics(
                up_arrays[f"{scenario}_value_and_grad_total"],
                jt_arrays[f"{scenario}_value_and_grad_total"],
                atol=args.total_atol,
                rtol=args.total_rtol,
            )
            grad_metrics = _array_metrics(
                up_arrays[f"{scenario}_grad"],
                jt_arrays[f"{scenario}_grad"],
                atol=args.grad_atol,
                rtol=args.grad_rtol,
            )
            up_grad = np.asarray(up_arrays[f"{scenario}_grad"])
            jt_grad = np.asarray(jt_arrays[f"{scenario}_grad"])
            per_pose_grad_metrics = _per_pose_gradient_metrics(
                up_grad,
                jt_grad,
                pose_contracts,
                atol=args.grad_atol,
                rtol=args.grad_rtol,
                relative_l2_limit=args.grad_rel_l2,
            )
            grad_metrics["per_pose"] = per_pose_grad_metrics
            padding_violations_up = 0
            padding_violations_jt = 0
            if up_grad.shape == jt_grad.shape and up_grad.ndim == 3:
                if len(pose_contracts) != up_grad.shape[0]:
                    grad_metrics["padding_contract_error"] = (
                        f"{len(pose_contracts)} per-pose contracts for "
                        f"gradient batch {up_grad.shape[0]}"
                    )
                    padding_violations_up = padding_violations_jt = 1
                else:
                    for pose_ind, pose_contract in enumerate(pose_contracts):
                        n_atoms = int(pose_contract["n_atoms"])
                        padding_violations_up += int(
                            np.count_nonzero(up_grad[pose_ind, n_atoms:])
                        )
                        padding_violations_jt += int(
                            np.count_nonzero(jt_grad[pose_ind, n_atoms:])
                        )
            else:
                padding_violations_up = padding_violations_jt = 1
            padding_pass = padding_violations_up == 0 and padding_violations_jt == 0
            grad_metrics["padding_exact_zero"] = {
                "pass": padding_pass,
                "upstream_nonzero_values": padding_violations_up,
                "jtmol_nonzero_values": padding_violations_jt,
            }
            grad_metrics["relative_l2_limit"] = args.grad_rel_l2
            grad_metrics["pass"] = bool(
                grad_metrics.get("pass", False)
                and grad_metrics.get("relative_l2", math.inf) <= args.grad_rel_l2
                and per_pose_grad_metrics.get("pass", False)
                and padding_pass
            )

            parity_pass = bool(
                terms_pass
                and channel_gradients_pass
                and total_metrics["pass"]
                and vg_total_metrics["pass"]
                and grad_metrics["pass"]
                and self_consistency_pass
            )
            scenario_report.update(
                {
                    "status": "ok" if parity_pass else "failed",
                    "parity_pass": parity_pass,
                    "weighted_channels": channel_metrics,
                    "weighted_channel_gradients": channel_gradient_metrics,
                    "total": total_metrics,
                    "value_and_grad_total": vg_total_metrics,
                    "gradient": grad_metrics,
                    "speed_ratios": None,
                }
            )

            # Ratios are deliberately absent until every parity gate for this
            # scenario succeeds.
            if parity_pass:
                up_timing = up_status["timing"]
                jt_timing = jt_status["timing"]
                reps = int(up_timing_contract["reps"])
                up_forward_ms = _validated_timing_median(
                    up_timing["forward"],
                    reps=reps,
                    n_poses=n_poses,
                    label=f"upstream {scenario} forward",
                )
                jt_forward_ms = _validated_timing_median(
                    jt_timing["forward"],
                    reps=reps,
                    n_poses=n_poses,
                    label=f"jtmol {scenario} forward",
                )
                up_vg_ms = _validated_timing_median(
                    up_timing["value_and_grad"],
                    reps=reps,
                    n_poses=n_poses,
                    label=f"upstream {scenario} value_and_grad",
                )
                jt_vg_ms = _validated_timing_median(
                    jt_timing["value_and_grad"],
                    reps=reps,
                    n_poses=n_poses,
                    label=f"jtmol {scenario} value_and_grad",
                )
                scenario_report["speed_ratios"] = {
                    "forward": up_forward_ms / jt_forward_ms,
                    "value_and_grad": up_vg_ms / jt_vg_ms,
                    "certifying_for_general_no_compromise_claim": bool(
                        exact_representation_pass
                        and native_coordinate_pass
                        and general_panel_pass
                        and complete_matrix
                        and native_batch_timing_pass
                        and self_consistency_pass
                    ),
                    "scope": (
                        "non-certifying 128-dispatch host-stream diagnostic"
                        if not native_batch_timing_pass
                        else (
                            "conditional exact-workload diagnostic"
                            if not (
                                exact_representation_pass
                                and native_coordinate_pass
                                and general_panel_pass
                                and complete_matrix
                                and self_consistency_pass
                            )
                            else "general no-compromise certification"
                        )
                    ),
                }
                scenario_report["timings"] = {
                    "upstream": up_timing,
                    "jtmol": jt_timing,
                }
            else:
                all_pass = False
            report["scenarios"][scenario] = scenario_report

        uncovered_channels: list[str] = []
        for name, coverage in channel_coverage.items():
            energy_active = coverage["upstream_energy_nonzero_values"] > 0
            gradient_active = coverage["upstream_gradient_nonzero_values"] > 0
            # Reference energy is coordinate-independent by definition.
            coverage["expected_coordinate_independent"] = name == "ref"
            coverage["coverage_pass"] = bool(
                energy_active and (gradient_active or name == "ref")
            )
            if not coverage["coverage_pass"]:
                uncovered_channels.append(name)
        report["channel_coverage"] = {
            "pass": not uncovered_channels,
            "uncovered_channels": uncovered_channels,
            "channels": channel_coverage,
        }
        if uncovered_channels:
            all_pass = False
        report["status"] = "ok" if all_pass else "failed"
    except BaseException as exc:
        report["contract_errors"].append(
            {"error_type": type(exc).__name__, "error": str(exc)}
        )
        report["status"] = "failed"

    _write_json(output, report)
    print(json.dumps({"status": report["status"], "report": str(output)}))
    return 0 if report["status"] == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict exact-coordinate jtmol versus upstream-tmol matrix",
        epilog=(
            "lk_bridge_uncpl is never automatically exempted; a mismatch is "
            "reported and remains a failure."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    upstream = sub.add_parser(
        "upstream", help="generate authoritative fixtures and score with official tmol"
    )
    upstream.add_argument("--manifest", required=True)
    upstream.add_argument(
        "--pdb-dir", help="override the rich manifest's S3 local_cache_hint"
    )
    upstream.add_argument("--out-dir", required=True)
    upstream.add_argument("--device", default="cuda", choices=("cuda",))
    upstream.add_argument("--warmup", type=int, default=5)
    upstream.add_argument("--reps", type=int, default=20)
    upstream.add_argument(
        "--skip-s3",
        action="store_true",
        help="run only S1/S2 (diagnostic; never satisfies the three-scenario goal)",
    )
    upstream.add_argument(
        "--s1-only",
        action="store_true",
        help="run only S1 (diagnostic; never satisfies the three-scenario goal)",
    )
    upstream.set_defaults(func=run_upstream)

    jtmol_parser = sub.add_parser(
        "jtmol", help="consume upstream fixtures verbatim and score with dense jtmol"
    )
    jtmol_parser.add_argument("--manifest", required=True)
    jtmol_parser.add_argument(
        "--pdb-dir", help="override the rich manifest's S3 local_cache_hint"
    )
    jtmol_parser.add_argument("--upstream-dir", required=True)
    jtmol_parser.add_argument("--out-dir", required=True)
    jtmol_parser.add_argument("--s2-batch-size", type=int, default=16)
    jtmol_parser.add_argument(
        "--s2-execution",
        choices=("host_stream", "lax_map"),
        default="host_stream",
        help=(
            "exact S2 execution strategy; host_stream is the OOM-safe 128-dispatch "
            "diagnostic, lax_map is the native compiled batch attempt"
        ),
    )
    jtmol_parser.add_argument("--warmup", type=int, default=5)
    jtmol_parser.add_argument("--reps", type=int, default=20)
    jtmol_parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="run only S1/S2 (diagnostic; never satisfies the three-scenario goal)",
    )
    jtmol_parser.add_argument(
        "--s1-only",
        action="store_true",
        help="run only S1 (diagnostic; never satisfies the three-scenario goal)",
    )
    jtmol_parser.set_defaults(func=run_jtmol)

    compare = sub.add_parser(
        "compare", help="fail closed on contract/parity mismatches, then report ratios"
    )
    compare.add_argument("--upstream-dir", required=True)
    compare.add_argument("--jtmol-dir", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument(
        "--energy-atol", type=float, default=CERTIFICATION_TOLERANCES["energy_atol"]
    )
    compare.add_argument(
        "--energy-rtol", type=float, default=CERTIFICATION_TOLERANCES["energy_rtol"]
    )
    compare.add_argument(
        "--total-atol", type=float, default=CERTIFICATION_TOLERANCES["total_atol"]
    )
    compare.add_argument(
        "--total-rtol", type=float, default=CERTIFICATION_TOLERANCES["total_rtol"]
    )
    compare.add_argument(
        "--grad-atol", type=float, default=CERTIFICATION_TOLERANCES["grad_atol"]
    )
    compare.add_argument(
        "--grad-rtol", type=float, default=CERTIFICATION_TOLERANCES["grad_rtol"]
    )
    compare.add_argument(
        "--grad-rel-l2", type=float, default=CERTIFICATION_TOLERANCES["grad_rel_l2"]
    )
    compare.set_defaults(func=run_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "warmup", 1) < 1:
        parser.error("--warmup must be >= 1")
    if getattr(args, "reps", 1) < 1:
        parser.error("--reps must be >= 1")
    if getattr(args, "s2_batch_size", 1) < 1:
        parser.error("--s2-batch-size must be >= 1")
    if getattr(args, "skip_s3", False) and getattr(args, "s1_only", False):
        parser.error("--skip-s3 and --s1-only are mutually exclusive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
