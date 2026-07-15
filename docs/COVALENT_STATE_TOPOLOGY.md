# Covalent state topology boundary

`pose_stack_from_biotite(..., covalent_bonds=...)` can install an explicit
inter-residue bond before block-type selection. A Ser `OG` request selects the
`SER_acyl` topology (no `HG`, an `acyl` connection on `OG`), while a ligand
params file must declare the mapped connection atom. The builder then wires the
two connection slots and propagates atom-level bond separation. A Ser-to-Ala
knockout cannot acquire the connection and loading fails.

`SER_acyl` is derived from the active ordinary `SER` and appended only to the
per-call covalent palette. It is deliberately absent from the default chemical
database and default packed palette: inserting an extra residue type there can
shift score-parameter tables and breaks frozen ordinary-protein parity.

The current supported mode is deliberately explicit:

```python
pose = pose_stack_from_biotite(
    atoms,
    ligand_params_files=[ti1_params],
    covalent_bonds=[(("A", 160, "OG"), ("L", 501, "C_REACT"))],
    covalent_state_mode="topology_only",
)
```

This mode is suitable for graph checks, mapped-atom geometry, and count-pair
inspection. `ScoreFunction` rejects it.

## Why scoring is locked

Three missing capabilities make the old finite-score path chemically
non-certifying:

1. `RawResidueType.bonds` stores only atom pairs. The params parser preserves
   `SINGLE`/`DOUBLE`/aromatic records as audit metadata, but scoring still
   discards bond order.
2. `LigandParamsFile.cartbonded_params` is parsed but no current jtmol energy
   term consumes state-specific bonded parameters, including the new
   inter-residue Ser-O--C bond.
3. `SER_acyl` has no built-in partial-charge table. Supplying ordinary SER
   charges would be chemically wrong; silently assigning zeros is also wrong.

Accordingly, `covalent_state_mode="parameterized"` raises
`NotImplementedError`. No free-energy or barrier interpretation is supported.

## Minimal port to unlock a parameterized backend

1. Carry typed bond order through `RawResidueType`, `RefinedResidueType`, and
   `PackedBlockTypes` without changing ordinary-protein parity.
2. Consume per-state cartbonded length, angle, proper, and improper parameters,
   including terms spanning an external connection.
3. Require complete, provenance-hashed state charge tables for `SER_acyl`, the
   mapped ligand, and the selected His microstate; reject missing atoms.
4. Add GS/TI1 fixtures with conserved atom maps, distinct C=O/C-O topology,
   explicit formal charges, and finite value/gradient tests against a trusted
   reference implementation.
5. Only then add a `parameterized` mode and a same-pocket two-arm objective.

`HIS_POS` and `HIS_D` already exist in the default chemical and electrostatic
databases. Inputs with both `HD1` and `HE2` select `HIS_POS`; this is covered by
the covalent-state tests.
