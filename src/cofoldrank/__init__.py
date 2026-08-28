"""cofoldrank -- predictor-agnostic structural selection for antibody-antigen co-folding.

Sampling a few structures from each of several co-folding predictors and picking one by cross-pose
agreement outperforms sampling one predictor deeply and trusting its confidence score. This package
implements the selection step:

    from cofoldrank import PoolSelector

    result = PoolSelector().rank("predictions/")   # predictions/<predictor>/<poses>
    print(result.best.path)

The selector reads coordinates only. It uses no confidence value, no predictor identity and no labelled
structures, so a predictor released after it was trained can be added to the pool without retraining.
"""

__version__ = "1.0.0"


def __getattr__(name):
    # Imported lazily so `import cofoldrank` stays cheap and does not pull in sklearn/biotite for
    # callers that only want the dataset tooling.
    if name in ("PoolSelector", "RankedPool", "Candidate", "rank_predictions", "SelectorModel"):
        from . import select

        return getattr(select, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PoolSelector", "RankedPool", "Candidate", "rank_predictions", "SelectorModel",
           "__version__"]
