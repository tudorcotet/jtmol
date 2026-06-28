# NPZ ↔ BIN/ZIP parity check

One-time sanity check flagged in
[`UPSTREAM_ALIGNMENT.md`](UPSTREAM_ALIGNMENT.md) §2: do jtmol's pure-numpy
`.npz` scoring tables carry the same numeric content as upstream's
torch-pickled `.bin`/`.zip` originals?

**Verdict: yes. Every comparable array is bit-equal.**

| Table | Source | Converted | Fields compared | Result |
|---|---|---|---|---|
| `rama.npz` | `rama.zip` (40 tables) | `convert_binary_tables.convert_rama` | `table_id`, `table` (float32 grid), `bbstep`, `bbstart` | 160/160 bit-equal |
| `omega_bbdep.npz` | `omega_bbdep.zip` (20 tables) | `convert_omega_bbdep` | `table_id`, `mu`, `sigma`, `bbstep`, `bbstart` | 100/100 bit-equal |
| `dunbrack.npz` | `dunbrack.bin` (10 rotameric + 8 semi-rotameric libraries) | `convert_dunbrack` | `rotameric_data.{rotamers, rotamer_probabilities, rotamer_means, rotamer_stdvs, prob_sorted_rot_inds, backbone_dihedral_start, backbone_dihedral_step, rotamer_alias}` + semi extras (`non_rot_chi_start/step/period`, `rotameric_chi_rotamers`, `nonrotameric_chi_probabilities`, `rotamer_boundaries`) | 192/192 bit-equal |

Total: **377 / 377 fields bit-equal** (no `~equal` slop, no failures).
Run on 2026-06-28 against the in-tree
`tmol_jax/database/default/scoring/{rama,omega_bbdep,dunbrack}.{npz,zip,bin}`.

## How it works

`scripts/verify_npz_bin_parity.py` reuses the existing dual loaders in
`tmol_jax/database/scoring/`:

- `RamaDatabase.load_tables_from_npz` vs `.load_tables_from_zip`
- `OmegaBBDepDatabase.load_tables_from_npz` vs `.load_tables_from_zip`
- `DunbrackRotamerLibrary.load_libraries_from_npz` vs `.load_libraries_from_bin`

Both paths produce the same attrs dataclasses with `numpy.ndarray`
fields; the `.zip`/`.bin` path goes through `torch.load` + a shim that
re-registers the upstream `tmol.database.scoring.*` classes so the
pickle reconstructs without the real `tmol` package on `sys.path`.
We then walk the fields and compare with `np.array_equal` (ints / strings)
or `np.allclose(rtol=1e-6, atol=1e-8)` followed by a dtype + bit-equality
check (floats).

## Reproducing

```bash
uv run python scripts/verify_npz_bin_parity.py
```

Requires `torch` (already in `pyproject.toml` extras / dev for this
exact purpose). Full per-field log is written to
`docs/_npz_parity_run.log` (gitignored). Re-run whenever:

- `scripts/convert_binary_tables.py` is edited
- the upstream `.bin`/`.zip` is refreshed from a new tmol release
- the `.npz` is regenerated

If the run reports anything other than `OK bit-equal` for every line,
do not ship. Re-run `python scripts/convert_binary_tables.py` from a
clean torch env and re-verify.

## Modal-side recipe (if a future host lacks torch)

Skip if torch installs locally. Recipe for an alt-arch host:

```python
# modal_verify_npz_parity.py
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("torch==2.12.0", "numpy", "attrs", "pyyaml")
    .add_local_dir(".", "/root/jtmol")
)
app = modal.App("jtmol-npz-parity", image=image)

@app.function()
def run():
    import subprocess, sys
    return subprocess.run(
        [sys.executable, "scripts/verify_npz_bin_parity.py"],
        cwd="/root/jtmol", capture_output=True, text=True,
    ).stdout

if __name__ == "__main__":
    with app.run():
        print(run.remote())
```

`modal run modal_verify_npz_parity.py` produces the same per-field log
as the local run.
