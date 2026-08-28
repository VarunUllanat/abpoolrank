"""Predictor-agnostic structural selection over pools of predicted antibody-antigen complexes.

Given several predicted structures per target -- ideally from several different co-folding predictors --
this picks the one most likely to be correct, using pose geometry alone: no native structure, no
confidence score, and no knowledge of which predictor produced which pose.

    from cofoldrank import PoolSelector
    result = PoolSelector().rank("predictions/")
    result.best.path
"""
from .api import Candidate, PoolSelector, RankedPool, rank_predictions
from .features import FEATURES, MIN_POSES_PER_SOURCE, pool_features
from .model import SelectorModel
from .pool import Pool, discover

__all__ = ["PoolSelector", "RankedPool", "Candidate", "rank_predictions",
           "SelectorModel", "Pool", "discover", "pool_features", "FEATURES",
           "MIN_POSES_PER_SOURCE"]
