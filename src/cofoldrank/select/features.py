"""Pool features for the predictor-agnostic selector.

Four features, all derived from the pose-similarity matrices and the source partition:

  pss_pct     within-pool percentile of pose-similarity support (mean similarity to every other pose)
  srcbal_pct  within-pool percentile of SOURCE-BALANCED support: for each OTHER source, the top-k mean
              similarity to that source's poses, then averaged over sources. This is the only feature
              that uses source membership, and it uses it anonymously -- which partition a pose belongs
              to, never which predictor that partition is.
  epi_pct     the same support statistic computed on epitope-overlap instead of geometry
  log_pool    log of the pool size, so one model covers pools of different sizes

Everything enters as a WITHIN-POOL PERCENTILE. Absolute consensus similarity does not transfer: poses for
complexes inside a predictor's training window agree with each other far more than poses for complexes
outside it (median geom_sim 0.134 against 0.036), so a model fitted on absolute similarity would read
"inside the training window" rather than "correct". Percentiles are invariant to that shift, which is what
makes the selector portable across predictors and cohorts.

srcbal reduces EXACTLY to pss when every source contributes one pose: the top-k mean with k=1 is the
single similarity, and the 1/(G-1) normaliser is constant within a pool so it cancels from the percentile.
That is why the selector needs at least two poses per predictor -- below that, two of its four features
are one column and within-source reproducibility is unobservable.
"""
from __future__ import annotations

import numpy as np

#: Feature columns of the released predictor-agnostic selector, in order.
FEATURES = ("pss_pct", "srcbal_pct", "epi_pct", "log_pool")

#: Number of best matches averaged per other source in the source-balanced statistic.
TOPK = 3

#: Poses per source below which srcbal degenerates to pss (see module docstring).
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
