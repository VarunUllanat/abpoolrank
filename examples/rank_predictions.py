"""Rank a folder of predicted antibody-antigen structures.

    python examples/rank_predictions.py examples/demo_pool

The only input is a directory with one subdirectory per predictor. No confidence scores, no predictor
names the selector recognises, and no reference structure are involved.
"""
from __future__ import annotations

import argparse

from cofoldrank import PoolSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="examples/demo_pool",
                    help="directory laid out as <directory>/<predictor>/<pose files>")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--antibody-chains", default="H,L")
    a = ap.parse_args()

    selector = PoolSelector(antibody_chains=tuple(a.antibody_chains.split(",")),
                            workers=a.workers)
    result = selector.rank(a.directory)

    # --- what to use -------------------------------------------------------
    print(f"pool: {len(result.pool)} poses from {result.pool.n_sources} predictors "
          f"{result.pool.counts()}")
    print(f"\nselected pose: {result.best.path}")
    print(f"  predictor:   {result.best.source}")
    print(f"  score:       {result.best.score:.4f}")

    # --- the full ranking, as a polars DataFrame ---------------------------
    print("\nfull ranking (score is a within-pool ordering, not a quality estimate):")
    print(result.table)

    # --- a diverse shortlist: best pose from each predictor -----------------
    seen, shortlist = set(), []
    for c in result.candidates:
        if c.source not in seen:
            seen.add(c.source)
            shortlist.append(c)
    print("\nbest pose per predictor (a shortlist when you can afford to inspect several):")
    for c in shortlist:
        print(f"  {c.source:<18} rank {c.rank:>2}  {c.path.name}")

    # The features behind the ranking, if you want to inspect or plot them.
    # Columns are ('pss_pct', 'srcbal_pct', 'epi_pct', 'log_pool').
    print(f"\nfeature matrix: {result.features.shape} columns={selector.model.features}")

    if result.warnings:
        print("\nwarnings:")
        for w in result.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
