"""Read /tmp/parity_driver.log lines and produce docs/UPSTREAM_PARITY.md.

Helper kept in-tree so parity_driver crashes don't leave us without a
report — we can always replay the log and regenerate the markdown.
"""
from __future__ import annotations

import os
import re
import sys

REPO = "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"

def parse_log(path):
    """Parse [row] lines from the driver stdout log.

    Accepts three row shapes:

    1. Fully-evaluated live: ``live fwd={P|F} grad={P|F} Δe=...
       max|dg|=... (duration)``
    2. Shape mismatch (1r21-style): ``live fwd={P|F} grad=-
       Δe=... shape_mismatch (...)``
    3. ``SKIP (reason)`` when the row couldn't be evaluated at all.
    """
    rows = []
    live_re = re.compile(
        r"\[row\]\s+(\S+)\s+(\S+)\s+live\s+fwd=(\w)\s+grad=(\w)\s+"
        r"Δe=([+\-\d.e]+)\s+max\|dg\|=([\d.e+\-]+)\s+\(([\d.]+)s\)"
    )
    # NOTE the greedy ``.*`` at the end: the shape string itself contains
    # parentheses (e.g. ``jax=(1610, 3), torch=(1615, 3)``) so a
    # non-greedy ``.*?`` would stop at the first ``)`` and truncate the
    # note mid-shape. We anchor on the duration suffix to bound the
    # capture.
    shape_re = re.compile(
        r"\[row\]\s+(\S+)\s+(\S+)\s+live\s+fwd=(\w)\s+grad=-\s+"
        r"Δe=([+\-\d.e]+)\s+shape_mismatch\s+\((.*)\)\s+\([\d.]+s\)"
    )
    skip_re = re.compile(r"\[row\]\s+(\S+)\s+(\S+)\s+SKIP \((.*)\)")
    with open(path, "r") as f:
        for line in f:
            m = live_re.search(line)
            if m:
                pdb, st, fwd, grad, de, dg, dur = m.groups()
                rows.append({
                    "pdb": pdb, "st": st, "mode": "live",
                    "fwd_ok": fwd == "P", "grad_ok": grad == "P",
                    "delta_e": float(de), "max_abs_dg": float(dg),
                    "duration": float(dur), "note": "",
                })
                continue
            m = shape_re.search(line)
            if m:
                pdb, st, fwd, de, shapes = m.groups()
                rows.append({
                    "pdb": pdb, "st": st, "mode": "live",
                    "fwd_ok": fwd == "P", "grad_ok": None,
                    "delta_e": float(de), "max_abs_dg": None,
                    "duration": None,
                    "note": f"grad shape mismatch: {shapes}",
                })
                continue
            m = skip_re.search(line)
            if m:
                pdb, st, note = m.groups()
                rows.append({
                    "pdb": pdb, "st": st, "mode": "skip",
                    "fwd_ok": None, "grad_ok": None,
                    "delta_e": None, "max_abs_dg": None,
                    "duration": None, "note": note,
                })
    return rows


def render(rows, up_avail, up_reason):
    out = []
    out.append("# Upstream parity (tmol_jax vs tmol)\n\n")
    out.append(f"- Upstream live: **{'yes' if up_avail else 'no'}**\n")
    if not up_avail:
        out.append(f"- Fallback reason: `{up_reason}`\n")
    out.append("- Tolerances: atol=1e-05, rtol=1e-03\n")
    out.append("- Comparison rule: `|E_jax - E_ref| <= atol + rtol * |E_ref|` "
               "(matches torch.autograd.gradcheck default)\n\n")

    out.append("## Caveats\n\n")
    out.append(
        "- **1BL8 (tetramer)**: Upstream `pose_stack_from_pdb` refuses to "
        "load 1BL8 because several sidechains are truncated in the "
        "crystal structure (Arg, Glu, Ile with missing non-leaf atoms). "
        "All 20 1BL8 rows are therefore marked as *skipped* — tmol_jax "
        "loads the structure fine, but we have no upstream reference to "
        "compare against.\n"
    )
    out.append(
        "- **1r21 (disulfide)**: The two frameworks build pose stacks "
        "with different atom counts (tmol_jax: 1610, upstream: 1615). "
        "This is a legitimate divergence — likely different handling of "
        "the disulfide-bridged Cys pair where upstream records the HG "
        "atoms that tmol_jax removes post-cystine-formation. Forward "
        "energies therefore diverge by a few kcal/mol across terms, "
        "and elementwise gradient comparison is not meaningful with "
        "shape mismatch. All 20 1r21 rows are thus *expected* FAIL on "
        "forward energy and *skipped* on gradient. A follow-up fix "
        "should harmonise the pose-stack builder's disulfide handling "
        "across frameworks.\n"
    )
    out.append(
        "- **hbond**: Has a known ~0.8% residual vs upstream on 1ubq "
        "and ~0.5% on 1qys. The residual comes from a different "
        "acceptor-base selection heuristic (bond-graph vs icoor-tree "
        "walk) on ring acceptors (HIS NE2) and sp3 hydroxyls. See the "
        "`docs/PARITY.md` diagnostic for the full investigation.\n"
    )
    out.append(
        "- **Gradients on periodic / spline-based terms**: `omega`, "
        "`cart_torsions`, `cart_hxltorsions`, `dunbrack_rotdev`, "
        "`lk_bridge`, `lk_bridge_uncpl` show per-atom gradient "
        "diffs O(1e-3 - 1e-1) on some atoms. Forward energies match, "
        "so these are spot-where-the-discontinuity-falls differences, "
        "not term-wide bugs.\n\n"
    )

    total = len(rows)
    fwd_p = sum(1 for r in rows if r["fwd_ok"] is True)
    fwd_f = sum(1 for r in rows if r["fwd_ok"] is False)
    fwd_s = sum(1 for r in rows if r["fwd_ok"] is None)
    gr_p = sum(1 for r in rows if r["grad_ok"] is True)
    gr_f = sum(1 for r in rows if r["grad_ok"] is False)
    gr_s = sum(1 for r in rows if r["grad_ok"] is None)

    out.append("## Summary\n\n")
    out.append(f"- Total rows: **{total}**\n")
    out.append(f"- Forward: **{fwd_p} PASS**, {fwd_f} FAIL, {fwd_s} SKIP\n")
    out.append(f"- Gradient: **{gr_p} PASS**, {gr_f} FAIL, {gr_s} SKIP "
               f"(skip when upstream load fails or gradient is "
               f"trivially zero on both sides)\n\n")

    seen = []
    for r in rows:
        if r["pdb"] not in seen:
            seen.append(r["pdb"])

    for pdb in seen:
        out.append(f"## {pdb}\n\n")
        out.append("| Term | Mode | Fwd | Grad | Δ(E) | max\\|Δg\\| | Note |\n"
                   "|---|---|:---:|:---:|---:|---:|---|\n")
        for r in rows:
            if r["pdb"] != pdb:
                continue
            m = lambda v: "—" if v is None else ("PASS" if v else "FAIL")
            de = f"{r['delta_e']:+.3e}" if r["delta_e"] is not None else ""
            mg = f"{r['max_abs_dg']:.3e}" if r["max_abs_dg"] is not None else ""
            out.append(
                f"| {r['st']} | {r['mode']} | {m(r['fwd_ok'])} "
                f"| {m(r['grad_ok'])} | {de} | {mg} | {r['note']} |\n"
            )
        out.append("\n")

    return "".join(out)


def main():
    rows = []
    main_log = "/tmp/parity_driver.log"
    sub_log = "/tmp/parity_driver_1r21.log"
    if os.path.isfile(main_log):
        rows.extend(parse_log(main_log))
    if os.path.isfile(sub_log):
        rows.extend(parse_log(sub_log))

    if not rows:
        print("no data found", file=sys.stderr)
        return 1

    md = render(rows, up_avail=True, up_reason=None)
    out_path = os.path.join(REPO, "docs", "UPSTREAM_PARITY.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
