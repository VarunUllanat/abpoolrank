"""abpoolrank -- rank a pool of predicted antibody-antigen structures and pick the best one.

    from abpoolrank import PoolSelector

    result = PoolSelector().rank("predictions/")   # predictions/<predictor>/<poses>
    print(result.best.path)
"""

__version__ = "1.3.0"


def __getattr__(name):
    # Imported lazily so `import abpoolrank` does not load sklearn/biotite.
    if name in ("PoolSelector", "RankedPool", "Candidate", "rank_predictions", "SelectorModel"):
        from . import select

        return getattr(select, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PoolSelector", "RankedPool", "Candidate", "rank_predictions", "SelectorModel",
           "__version__"]
