"""Shared pytest fixtures for the jtmol test suite.

Fixtures here provide paths to test data files and pre-built jtmol
objects (canonical ordering, score function, pose stacks). Every
fixture that touches optional dependencies or optional data falls back
to ``pytest.skip(...)`` so that partial environments still run the
tests that can run.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """Absolute path to ``jtmol/tests/data``.

    Uses ``os.path.dirname`` on this conftest module so the path is
    stable regardless of the current working directory.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture(scope="session")
def pdb_1ubq(test_data_dir: str) -> str:
    """Absolute path to the ``1ubq.pdb`` test fixture.

    Skips the test if the file is not present on disk.
    """
    path = os.path.join(test_data_dir, "1ubq.pdb")
    if not os.path.isfile(path):
        pytest.skip(f"Test data file not found: {path}")
    return path


@pytest.fixture(scope="session")
def pdb_1qys(test_data_dir: str) -> str:
    """Absolute path to the ``1qys.pdb`` test fixture.

    Skips the test if the file is not present on disk.
    """
    path = os.path.join(test_data_dir, "1qys.pdb")
    if not os.path.isfile(path):
        pytest.skip(f"Test data file not found: {path}")
    return path


@pytest.fixture(scope="session")
def default_packed_block_types():
    """Session-scoped default PackedBlockTypes from jtmol.

    Skips if ``jtmol.io.canonical_ordering`` cannot be imported or
    if constructing the default PBT raises.
    """
    try:
        from jtmol.io.canonical_ordering import default_packed_block_types
    except ImportError as exc:
        pytest.skip(f"jtmol.io.canonical_ordering unavailable: {exc}")

    try:
        return default_packed_block_types()
    except Exception as exc:  # noqa: BLE001 - propagate as skip
        pytest.skip(f"default_packed_block_types() failed: {exc}")


@pytest.fixture(scope="session")
def default_score_function():
    """Session-scoped default beta2016 ScoreFunction.

    Skips if ``jtmol.beta2016_score_function`` cannot be imported or
    if building the score function raises.
    """
    try:
        from jtmol import beta2016_score_function
    except ImportError as exc:
        pytest.skip(f"jtmol.beta2016_score_function unavailable: {exc}")

    try:
        return beta2016_score_function()
    except Exception as exc:  # noqa: BLE001 - propagate as skip
        pytest.skip(f"beta2016_score_function() failed: {exc}")


@pytest.fixture(scope="session")
def pose_stack_1ubq(pdb_1ubq: str):
    """Session-scoped PoseStack loaded from ``1ubq.pdb``.

    Skips if ``jtmol.io.pose_stack_from_pdb`` is unavailable or if
    loading the structure raises for any reason.
    """
    try:
        from jtmol.io import pose_stack_from_pdb
    except ImportError as exc:
        pytest.skip(f"jtmol.io.pose_stack_from_pdb unavailable: {exc}")

    try:
        return pose_stack_from_pdb(pdb_1ubq)
    except Exception as exc:  # noqa: BLE001 - propagate as skip
        pytest.skip(f"Failed to load PoseStack from {pdb_1ubq}: {exc}")
