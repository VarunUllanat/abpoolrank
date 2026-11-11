"""Pool features for the selector.

Four features per pose, computed from the pose-similarity matrices and the source partition:

  pss_pct     within-pool percentile of mean geometric similarity to every other pose
  srcbal_pct  within-pool percentile of source-balanced support: for each other source, the mean of the
              top-k similarities to that source's poses, averaged over sources
  epi_pct     within-pool percentile of the same statistic computed on epitope overlap
  log_pool    log of the pool size

With one pose per source, srcbal_pct equals pss_pct; the selector needs at least two poses per source.
"""
from __future__ import annotations

import numpy as np

#: Feature columns, in order.
FEATURES = ("pss_pct", "srcbal_pct", "epi_pct", "log_pool")

#: Number of best matches averaged per other source in the source-balanced statistic.
TOPK = 3

#: Poses per source below which srcbal_pct equals pss_pct.
MIN_POSES_PER_SOURCE = 2


def percentile(v) -> np.ndarray:
    """Rank of each element rescaled to [0, 1]. A pool of one gets 0."""
    v = np.asarray(v)
    n = len(v)
    if n < 2:
        return np.zeros(n)
    return np.argsort(np.argsort(v)) / (n - 1)


def support(sim, codes, n_sources):
    """Plain and source-balanced support from one similarity matrix.

    `codes` maps each pose to its source index in [0, n_sources).
    """
    sim = np.asarray(sim, dtype=float)
    codes = np.asarray(codes)
    n = len(codes)
    offdiag = ~np.eye(n, dtype=bool)
    pss = (sim * offdiag).sum(1) / max(n - 1, 1)

    per = np.empty((n_sources, n))
    for h in range(n_sources):
        idx = np.where(codes == h)[0]
        if len(idx) == 0:
            per[h] = np.nan
            continue
        k = max(1, min(TOPK, len(idx)))
        per[h] = np.sort(sim[:, idx], axis=1)[:, ::-1][:, :k].mean(1)

    own = np.zeros((n_sources, n), bool)
    own[codes, np.arange(n)] = True                     # exclude a pose's own source
    with np.errstate(invalid="ignore"):
        srcbal = np.nanmean(np.where(own, np.nan, per), axis=0)
    return pss, np.nan_to_num(srcbal)


def pool_features(geom_sim, epi_jac, codes, n_sources=None) -> np.ndarray:
    """(n, 4) feature matrix for one pool, columns in FEATURES order."""
    codes = np.asarray(codes)
    n_sources = int(codes.max()) + 1 if n_sources is None else n_sources
    pss, srcbal = support(geom_sim, codes, n_sources)
    epi, _ = support(epi_jac, codes, n_sources)
    n = len(codes)
    return np.column_stack([percentile(pss), percentile(srcbal), percentile(epi),
                            np.full(n, np.log(n))])
