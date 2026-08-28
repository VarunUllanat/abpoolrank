"""Fit and export the predictor-agnostic selector checkpoint.

    python -m cofoldrank.select.train --out /path/to/agnostic_struct.joblib

This reproduces the fit reported in the paper: pools drawn from cached prediction-to-prediction
similarity matrices, features as in `features.pool_features`, target = the within-pool percentile of
DockQ, three seeds averaged. It needs two inputs, both of which are study artifacts and not shipped
with the package: a directory of per-target `.npz` pair matrices (`cand_ids`, `geom_sim`, `epi_jac`)
and a labels parquet. The default paths point at the study filesystem; pass --cache and --labels to
train on your own.

Training consumes DockQ labels; INFERENCE never does. Nothing here is needed to rank a new pool -- that
only requires the exported checkpoint.

Pool draws are seeded with a BLAKE2b digest of the draw identifiers rather than Python's built-in `hash`,
which is salted per process when PYTHONHASHSEED is unset. Under the old seeding, repeated runs of an
identical command drew different pose subsets and absolute Success@1 moved by up to ~2 pp between runs.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .features import percentile, pool_features
from .model import DEFAULT_PATHS, fit

# Locations of the frozen development artifacts. These are inputs to TRAINING only.
DATA = Path("/data/cb/varunullanat/cofoldrank_data")
OUT = Path("/data/cb/varunullanat/cofoldrank_out")

#: Sources of the released five-predictor training pools.
TRAIN_SOURCES = ("boltz2", "promera", "protenix_v2", "esmfold2_fast", "opendde")
#: Poses per source in the training pools, matching the published protocol.
POSES_PER_SOURCE = (1, 2, 4, 10, 20)
#: Draw-seed offset for training pools, frozen so the fit is reproducible.
SEED0 = 7


def stable_seed(*parts) -> int:
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


def build_pools(cache_dir, targets, labels, sources, poses_per_source, reps, seed0=SEED0):
    """Feature matrices and rank targets for every (target, m, rep) training pool."""
    cache_dir = Path(cache_dir)
    n_src = len(sources)
    Xs, ys = [], []
    for t in targets:
        f = cache_dir / f"{t}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        ids = [str(x) for x in z["cand_ids"]]
        folders = np.array([i.split("|")[1] for i in ids])
        idx_by_src = {g: np.where(folders == g)[0] for g in sources}
        if any(len(idx_by_src[g]) == 0 for g in sources):
            continue
        dq = np.array([labels.get(i, np.nan) for i in ids], float)
        if any(not np.isfinite(dq[idx_by_src[g]]).all() for g in sources):
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

    Xs, ys = build_pools(a.cache, targets, labels, sources, POSES_PER_SOURCE, a.reps)
    if not Xs:
        raise SystemExit(f"no training pools built from {a.cache}")
    X, y = np.vstack(Xs), np.concatenate(ys)
    print(f"{len(Xs)} pools, {X.shape[0]} poses", flush=True)

    model = fit(X, y, n_seeds=a.n_seeds,
                metadata={"sources_trained_on": list(sources),
                          "n_train_targets": len(targets), "n_train_pools": len(Xs),
                          "poses_per_source": list(POSES_PER_SOURCE), "pool_draws_per_setting": a.reps,
                          "pair_cache": str(a.cache), "seed0": SEED0})
    path = model.save(a.out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"features: {model.features}")
    print(f"metadata: {model.metadata}")


if __name__ == "__main__":
    main()
