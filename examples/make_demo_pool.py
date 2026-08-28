"""Materialise a demo pool from the study's predictions, in the layout `cofoldrank select` expects.

    python examples/make_demo_pool.py                          # default target
    python examples/make_demo_pool.py --target pdb_00008dcn_A_B --per-source 4

Writes examples/demo_pool/<predictor>/<pose files> by copying real predicted structures, and prints the
DockQ of every copied pose alongside what each predictor's own confidence score would have picked from
ALL of its 100 samples. That is the comparison the method is about: a small pool ranked by cross-pose
agreement against each predictor sampled deeply and ranked by its own confidence.

DockQ and confidence are shown only to describe the choice. Neither is an input to selection.

Requires the study's prediction cohort on disk. With your own predictions, skip this and point
`cofoldrank select` at your own directory.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

OUT = Path("/data/cb/varunullanat/cofoldrank_out")
DATA = Path("/data/cb/varunullanat/cofoldrank_data")
COHORT = OUT / "results" / "fig2_cohort.parquet"
CONFIDENCE = DATA / "manifests" / "candidate_confidence_fig2_n100_clean.parquet"

# Each predictor's own recommended ranking quantity -- the field its authors tell users to rank by.
CONF_FIELD = {"boltz2": "confidence_score", "promera": "iCS_prob", "protenix_v2": "iptm",
              "esmfold2_fast": "iptm", "opendde": "iptm", "esmfold2": "iptm"}
MEDIUM = 0.49


def confidence_picks(target, cohort):
    """For each predictor: the DockQ of the pose its own confidence ranks first, over all its samples."""
    import polars as pl

    if not CONFIDENCE.exists():
        return None
    cf = pl.read_parquet(CONFIDENCE).filter(pl.col("target_id") == target)
    if cf.is_empty():
        return None
    out = {}
    for (src,), g in cohort.group_by("folder"):
        field = CONF_FIELD.get(src)
        if field is None or field not in cf.columns:
            continue
        j = (g.select(["seed", "sample", "dockq"])
              .join(cf.filter(pl.col("folder") == src).select(["seed", "sample", field]),
                    on=["seed", "sample"], how="inner")
              .drop_nulls([field, "dockq"]))
        if j.is_empty():
            continue
        out[src] = {"dockq": float(j.sort(field, descending=True)["dockq"][0]),
                    "n_samples": j.height}
    return out or None


def main():
    ap = argparse.ArgumentParser()
    # Default: only OpenDDE's poses are near-native here, so the selector has to find the two correct
    # poses among eight badly wrong ones -- and it beats every predictor's own confidence.
    ap.add_argument("--target", default="pdb_00009cq8_C_E")
    ap.add_argument("--per-source", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).parent / "demo_pool"))
    ap.add_argument("--cohort", default=str(COHORT))
    a = ap.parse_args()

    import polars as pl

    coh = pl.read_parquet(a.cohort).filter(pl.col("target_id") == a.target)
    if coh.is_empty():
        raise SystemExit(f"target {a.target} not in {a.cohort}")

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    rows = []
    for (src,), g in coh.sort(["folder", "cand_idx"]).group_by("folder", maintain_order=True):
        d = out / src
        d.mkdir(parents=True, exist_ok=True)
        for i in range(min(a.per_source, g.height)):
            srcp = Path(g["struct_path"][i])
            dst = d / f"pose_{i}{srcp.suffix}"
            shutil.copyfile(srcp, dst)
            rows.append({"source": src, "pose": dst.name, "dockq": g["dockq"][i]})

    t = pl.DataFrame(rows)
    print(f"target {a.target}")
    print(f"wrote {out} -- {t.height} poses from {t['source'].n_unique()} predictors\n")
    print(t.sort("dockq", descending=True).to_pandas().to_string(index=False))
    oracle = t["dockq"].max()
    print(f"\nbest pose in this {t.height}-structure pool: DockQ = {oracle:.3f}"
          f"   (medium-or-better threshold is {MEDIUM})")

    picks = confidence_picks(a.target, coh)
    if picks:
        print(f"\nfor comparison -- what each predictor's OWN confidence picks from all its samples:")
        for src, v in sorted(picks.items(), key=lambda kv: -kv[1]["dockq"]):
            flag = "" if v["dockq"] >= MEDIUM else "   <- below medium"
            print(f"  {src:<16} {v['n_samples']:>4} samples -> DockQ {v['dockq']:.3f}{flag}")
        print(f"\n  best of those: {max(v['dockq'] for v in picks.values()):.3f} "
              f"from {sum(v['n_samples'] for v in picks.values())} structures")
        print(f"  the pool below: {oracle:.3f} available from {t.height} structures")

    print(f"\nnow run:  cofoldrank select {out}")


if __name__ == "__main__":
    main()
