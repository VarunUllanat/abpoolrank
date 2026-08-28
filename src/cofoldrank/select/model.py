"""The selector model: a small seed-averaged gradient-boosted ranker, plus persistence.

The model maps the four pool features to a within-pool score; only the ORDER of scores matters, and the
selected pose is the argmax. Two choices are load-bearing and are baked into the checkpoint rather than
left to the caller:

* **Within-pool rank target.** Fitting absolute DockQ is misspecified: training pools average 0.467 DockQ
  against 0.283 on held-out complexes, because training complexes sit inside the predictors' training
  windows. A pointwise fit to that level distorts the within-pool ordering that is actually being asked
  for -- measured Spearman(prediction, pss_pct) was +0.83, i.e. the model re-ranked away from its own
  strongest input and scored BELOW it. The target is therefore the within-pool percentile of DockQ.
* **Seed averaging.** HistGradientBoosting's threaded histogram build is not bit-exact, and a single seed
  moved the headline number by ~1 pp -- the same size as the effects being compared. Averaging over three
  seeds makes the released selector a fixed, reportable object.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FEATURES

#: Hyperparameters of the released selector. Frozen; changing these invalidates the checkpoint.
HPARAMS = {"max_iter": 250, "learning_rate": 0.07, "min_samples_leaf": 40}
N_SEEDS = 3

#: Where `load()` looks when given no explicit path, in order.
ENV_VAR = "COFOLDRANK_SELECTOR"
DEFAULT_PATHS = (Path(__file__).resolve().parent / "checkpoints" / "agnostic_struct.joblib",)


@dataclass
class SelectorModel:
    """Seed-averaged ranker over `features`."""

    models: list
    features: tuple = FEATURES
    metadata: dict = field(default_factory=dict)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.shape[1] != len(self.features):
            raise ValueError(f"expected {len(self.features)} features {self.features}, "
                             f"got array with {X.shape[1]} columns")
        return np.mean([m.predict(X) for m in self.models], axis=0)

    # ---- persistence -------------------------------------------------------
    def save(self, path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "features": tuple(self.features),
                     "metadata": self.metadata}, path, compress=3)
        return path

    @classmethod
    def load(cls, path=None) -> "SelectorModel":
        """Load a checkpoint. With no path, try $COFOLDRANK_SELECTOR then the default locations."""
        import joblib

        if path is None:
            candidates = []
            if os.environ.get(ENV_VAR):
                candidates.append(Path(os.environ[ENV_VAR]))
            candidates.extend(DEFAULT_PATHS)
            for c in candidates:
                if c.exists():
                    path = c
                    break
            else:
                raise FileNotFoundError(
                    "no selector checkpoint found. Train one with "
                    "`python -m cofoldrank.select.train`, or set "
                    f"${ENV_VAR} to a checkpoint path. Looked in: "
                    + ", ".join(str(c) for c in candidates))
        blob = joblib.load(Path(path))
        return cls(models=blob["models"], features=tuple(blob["features"]),
                   metadata=blob.get("metadata", {}))


def fit(X, y, n_seeds: int = N_SEEDS, metadata: dict | None = None,
        features=FEATURES) -> SelectorModel:
    """Fit the seed-averaged ranker on stacked pool features and within-pool rank targets."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    models = [HistGradientBoostingRegressor(random_state=k, **HPARAMS).fit(X, y)
              for k in range(n_seeds)]
    meta = {"n_seeds": n_seeds, "hparams": dict(HPARAMS), "n_train_poses": int(X.shape[0]),
            "target": "within-pool percentile of DockQ"}
    meta.update(metadata or {})
    return SelectorModel(models=models, features=tuple(features), metadata=meta)
