"""Train and save a selector checkpoint.

    python -m abpoolrank.select.train --out /path/to/agnostic_struct.joblib

Draws training pools from per-target pair-matrix files (`.npz` with `cand_ids`, `geom_sim`, `epi_jac`),
computes pool features, and fits the model to the within-pool percentile of DockQ from a labels parquet.
The default paths point at the study's data; pass --cache and --labels to train on your own. Ranking a
new pool needs only the checkpoint, never DockQ labels.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .features import percentile, pool_features
from .model import DEFAULT_PATHS, fit

# Default training inputs (study data).
DATA = Path("/data/cb/varunullanat/cofoldrank_data")
OUT = Path("/data/cb/varunullanat/cofoldrank_out")

#: Sources of the released five-predictor training pools.
TRAIN_SOURCES = ("boltz2", "promera", "protenix_v2", "esmfold2_fast", "opendde")
#: Poses per source in the training pools.
POSES_PER_SOURCE = (1, 2, 4, 10, 20)
#: Draw-seed offset for training pools.
SEED0 = 7


def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from the draw identifiers (unaffected by PYTHONHASHSEED)."""
    h = hashlib.blake2b("|".join(map(repr, parts)).encode(), digest_size=4).digest()
    return int.from_bytes(h, "big")


def load_labels(path, split="train"):
    """target|source|seed|sample -> DockQ, from a labels parquet.

    The parquet needs columns target_id, folder, seed, sample, dockq, split.
    """
    import polars as pl

    L = pl.read_parquet(path).filter(pl.col("split") == split).drop_nulls(["dockq"])
    return {f"{t}|{g}|s{s}|n{n}": d for t, g, s, n, d in
            zip(L["target_id"], L["folder"], L["seed"], L["sample"], L["dockq"])}


def build_pools(cache_dir, targets, labels, sources, poses_per_source, reps, seed0=SEED0, skipped=None):
    """Feature matrices and rank targets for every (target, m, rep) training pool.

    A target is skipped if its pair cache is missing, a source has no poses, or any pose of any source lacks
    a finite DockQ label. Pass a dict as `skipped` to receive {target: reason}.
    """
    skipped = {} if skipped is None else skipped
    cache_dir = Path(cache_dir)
    n_src = len(sources)
    Xs, ys = [], []
    for t in targets:
        f = cache_dir / f"{t}.npz"
        if not f.exists():
            skipped[t] = "no pair cache"
            continue
        z = np.load(f)
        ids = [str(x) for x in z["cand_ids"]]
        folders = np.array([i.split("|")[1] for i in ids])
        idx_by_src = {g: np.where(folders == g)[0] for g in sources}
        if any(len(idx_by_src[g]) == 0 for g in sources):
            skipped[t] = "no poses from " + ",".join(g for g in sources if len(idx_by_src[g]) == 0)
            continue
        dq = np.array([labels.get(i, np.nan) for i in ids], float)
        unlabelled = [g for g in sources if not np.isfinite(dq[idx_by_src[g]]).all()]
        if unlabelled:
            skipped[t] = "missing DockQ labels for " + ",".join(unlabelled)
            continue
        S = z["geom_sim"].astype(np.float32)
        E = z["epi_jac"].astype(np.float32)
        for m in poses_per_source:
            if any(len(idx_by_src[g]) < m for g in sources):
                continue
            for rep in range(reps):
                rng = np.random.default_rng(stable_seed(t, m, rep, seed0))
                sel, codes = [], []
                for h, g in enumerate(sources):
                    sel.append(rng.choice(idx_by_src[g], m, replace=False))
                    codes.append(np.full(m, h))
                sel = np.concatenate(sel)
                codes = np.concatenate(codes)
                ix = np.ix_(sel, sel)
                Xs.append(pool_features(S[ix], E[ix], codes, n_src))
                ys.append(percentile(dq[sel]))            # within-pool rank target
    return Xs, ys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_PATHS[0]), help="checkpoint path to write")
    ap.add_argument("--cache", default=str(DATA / "features" / "pairmat2"),
                    help="directory of per-target pair-matrix npz files (cand_ids, geom_sim, epi_jac)")
    ap.add_argument("--labels", default=str(OUT / "results" / "labels_all_merged.parquet"),
                    help="parquet with target_id, folder, seed, sample, dockq, split")
    ap.add_argument("--holdout", default=str(OUT / "sabdab2_test106.parquet"),
                    help="optional parquet with a `name` column of targets to exclude from training")
    ap.add_argument("--sources", default=",".join(TRAIN_SOURCES),
                    help="comma-separated source names present in the pair cache")
    ap.add_argument("--reps", type=int, default=2, help="pool draws per (target, poses-per-source)")
    ap.add_argument("--n-seeds", type=int, default=3)
    a = ap.parse_args()

    import polars as pl

    sources = tuple(s for s in a.sources.split(",") if s)
    labels = load_labels(a.labels, "train")
    holdout = set()
    if a.holdout and Path(a.holdout).exists():
        holdout = set(pl.read_parquet(a.holdout)["name"].to_list())
    targets = sorted({k.split("|")[0] for k in labels} - holdout)
    print(f"{len(targets)} training targets ({len(holdout)} held-out targets excluded)", flush=True)

    skipped: dict = {}
    Xs, ys = build_pools(a.cache, targets, labels, sources, POSES_PER_SOURCE, a.reps, skipped=skipped)
    if not Xs:
        raise SystemExit(f"no training pools built from {a.cache}")
    n_used = len(targets) - len(skipped)
    print(f"{n_used} of {len(targets)} training targets used; {len(skipped)} skipped", flush=True)
    if skipped:
        from collections import Counter

        for reason, n in Counter(skipped.values()).most_common():
            print(f"  WARNING skipped {n} targets: {reason}", flush=True)
    X, y = np.vstack(Xs), np.concatenate(ys)
    print(f"{len(Xs)} pools, {X.shape[0]} poses", flush=True)

    model = fit(X, y, n_seeds=a.n_seeds,
                metadata={"sources_trained_on": list(sources),
                          "n_train_targets": n_used, "n_candidate_train_targets": len(targets),
                          "n_train_pools": len(Xs),
                          "poses_per_source": list(POSES_PER_SOURCE), "pool_draws_per_setting": a.reps,
                          "pair_cache": str(a.cache), "seed0": SEED0})
    path = model.save(a.out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"features: {model.features}")
    print(f"metadata: {model.metadata}")


if __name__ == "__main__":
    main()
