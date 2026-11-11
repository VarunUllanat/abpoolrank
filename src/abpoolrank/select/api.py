"""User-facing API: rank a pool of predicted antibody-antigen structures.

    from abpoolrank import PoolSelector

    selector = PoolSelector()                       # loads the released checkpoint
    result = selector.rank("predictions/")          # predictions/<predictor>/<poses>

    print(result.best.path)                         # the pose to use
    print(result.table)                             # full ranking

The selector reads only coordinates. It needs no confidence score, no predictor identity and no labelled
structures, so it ranks a pool containing a model it has never seen -- including one that reports no
confidence at all.
"""
from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import MIN_POSES_PER_SOURCE, pool_features
from .model import SelectorModel
from .pairwise import matrices_from_paths
from .pool import Pool, discover


@dataclass(frozen=True)
class Candidate:
    """One ranked pose."""

    path: Path
    source: str
    score: float
    rank: int


@dataclass
class RankedPool:
    """Result of ranking a pool: ordered candidates plus the features behind the order."""

    candidates: list          # best first
    scores: np.ndarray        # in input order
    features: np.ndarray      # (n, 4) in input order
    pool: Pool
    unparsed: list            # paths that could not be turned into a descriptor
    warnings: list

    @property
    def best(self) -> Candidate:
        return self.candidates[0]

    def top(self, k: int = 5) -> list:
        return self.candidates[:k]

    @property
    def table(self):
        """Ranking as a polars DataFrame, best first."""
        import polars as pl

        from .features import FEATURES

        order = np.argsort(-self.scores, kind="stable")
        d = {"rank": np.arange(1, len(order) + 1),
             "source": [self.pool.sources[i] for i in order],
             "score": self.scores[order],
             "path": [str(self.pool.paths[i]) for i in order]}
        for j, name in enumerate(FEATURES):
            d[name] = self.features[order, j]
        return pl.DataFrame(d)

    def to_csv(self, path) -> Path:
        path = Path(path)
        self.table.write_csv(path)
        return path

    def __repr__(self) -> str:
        return (f"RankedPool(n={len(self.candidates)}, sources={self.pool.n_sources}, "
                f"best={self.best.source}:{self.best.path.name})")


class PoolSelector:
    """The predictor-agnostic structural selector.

    checkpoint       path to a checkpoint (or a loaded SelectorModel); default resolves $ABPOOLRANK_SELECTOR
                     then the bundled "both" selector
    antibody_chains  chain IDs of the antibody; everything else is the antigen assembly
    workers          processes used to parse structures (parsing dominates runtime)
    model            a bundled selector by name instead of `checkpoint`: "both" (default; trained on paired
                     antibodies and VHHs), "paired-only" or "vhh-only"
    """

    def __init__(self, checkpoint=None, antibody_chains=("H", "L"), workers=1, model=None):
        if isinstance(checkpoint, SelectorModel):
            if model is not None:
                raise ValueError("pass either a checkpoint or a model name, not both")
            self.model = checkpoint
        else:
            self.model = SelectorModel.load(checkpoint, model=model)
        self.antibody_chains = tuple(antibody_chains)
        self.workers = workers

    # ---- main entry points -------------------------------------------------
    def rank(self, directory, sources=None, max_per_source=None) -> RankedPool:
        """Discover a pool under `directory` and rank it."""
        pool = discover(directory, sources=sources, max_per_source=max_per_source)
        return self.rank_pool(pool)

    def rank_paths(self, paths_by_source: dict) -> RankedPool:
        """Rank an explicitly specified pool: {source_name: [pose paths]}."""
        source_names = list(paths_by_source)
        paths, srcs, codes = [], [], []
        for i, s in enumerate(source_names):
            for p in paths_by_source[s]:
                paths.append(Path(p))
                srcs.append(s)
                codes.append(i)
        if not paths:
            raise ValueError("no poses given")
        pool = Pool(paths=paths, sources=srcs, source_names=source_names,
                    codes=np.asarray(codes), warnings=[])
        return self.rank_pool(pool)

    def rank_pool(self, pool: Pool) -> RankedPool:
        """Rank an already-constructed Pool."""
        if len(pool) < 2:
            raise ValueError(f"need at least 2 poses to rank a pool, got {len(pool)}")

        mats, descriptors = matrices_from_paths(
            pool.paths, antibody_chains=self.antibody_chains, workers=self.workers,
            channels=("geom_sim", "epi_jac"))
        unparsed = [p for p, d in zip(pool.paths, descriptors) if d is None]
        if len(unparsed) == len(pool):
            raise ValueError(
                f"none of the {len(pool)} structures yielded an interface. The most common cause is "
                f"chain naming: this pool was read with antibody chains {self.antibody_chains}, and "
                "every other chain treated as antigen. Pass antibody_chains=(...) to match your files.")

        X = pool_features(mats["geom_sim"], mats["epi_jac"], pool.codes, pool.n_sources)
        scores = self.model.predict(X)

        order = np.argsort(-scores, kind="stable")
        candidates = [Candidate(path=pool.paths[i], source=pool.sources[i],
                                score=float(scores[i]), rank=r)
                      for r, i in enumerate(order, start=1)]

        msgs = list(pool.warnings)
        if unparsed:
            msgs.append(f"{len(unparsed)} of {len(pool)} structures could not be parsed into an "
                        f"interface and rank last: {[p.name for p in unparsed[:5]]}"
                        + (" ..." if len(unparsed) > 5 else ""))
        counts = pool.counts()
        if pool.n_sources > 1 and min(counts.values()) < MIN_POSES_PER_SOURCE:
            msgs.append("at least one source has a single pose, so the selector is operating below its "
                        "designed regime (two poses per predictor).")
        for m in msgs:
            _warnings.warn(m, RuntimeWarning, stacklevel=2)
        return RankedPool(candidates=candidates, scores=scores, features=X, pool=pool,
                          unparsed=unparsed, warnings=msgs)


def rank_predictions(directory, checkpoint=None, antibody_chains=("H", "L"), workers=1,
                     sources=None, max_per_source=None, model=None) -> RankedPool:
    """One-call convenience wrapper around PoolSelector.rank."""
    return PoolSelector(checkpoint=checkpoint, antibody_chains=antibody_chains,
                        workers=workers, model=model).rank(directory, sources=sources,
                                              max_per_source=max_per_source)
