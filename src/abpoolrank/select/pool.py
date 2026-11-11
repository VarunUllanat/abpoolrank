"""Turn a directory of predicted structures into a pool.

Expected layout -- one subdirectory per predictor, named however the user likes:

    predictions/
      boltz2/        pose_0.cif  pose_1.cif  ...
      protenix/      sample_00.cif  ...
      my_new_model/  a.pdb  b.pdb  ...

Each subdirectory is one source. A flat directory of structures is treated as a single source, with a
warning in `Pool.warnings`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STRUCTURE_SUFFIXES = (".cif", ".mmcif", ".pdb", ".ent")


@dataclass
class Pool:
    """A set of candidate poses grouped by source."""

    paths: list           # pose file paths, grouped by source
    sources: list         # source label per pose, parallel to paths
    source_names: list    # unique source labels, in pool order
    codes: np.ndarray     # index into source_names per pose
    warnings: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def n_sources(self) -> int:
        return len(self.source_names)

    def counts(self) -> dict:
        return {s: int((self.codes == i).sum()) for i, s in enumerate(self.source_names)}


def _structures_in(d: Path) -> list:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in STRUCTURE_SUFFIXES)


def discover(root, sources=None, max_per_source=None) -> Pool:
    """Build a Pool from a directory tree.

    sources          restrict to these subdirectory names, in this order (default: all, sorted)
    max_per_source   keep only the first N poses per source, by sorted filename
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    warnings: list = []
    grouped: dict = {}

    if subdirs:
        names = [d.name for d in subdirs]
        if sources is not None:
            missing = [s for s in sources if s not in names]
            if missing:
                raise ValueError(f"requested sources not found under {root}: {missing}. "
                                 f"Available: {names}")
            subdirs = [root / s for s in sources]
        for d in subdirs:
            found = _structures_in(d)
            if not found:
                warnings.append(f"source {d.name!r} contains no structure files; skipped")
                continue
            grouped[d.name] = found[:max_per_source] if max_per_source else found
    else:
        flat = _structures_in(root)
        if not flat:
            raise FileNotFoundError(
                f"no structure files ({', '.join(STRUCTURE_SUFFIXES)}) found under {root}. "
                "Expected one subdirectory per predictor, each containing predicted structures.")
        grouped[root.name] = flat[:max_per_source] if max_per_source else flat
        warnings.append(
            f"{root} has no subdirectories, so all {len(flat)} poses were treated as ONE source. "
            "The source-balanced feature is undefined for a single source; put each predictor's "
            "structures in its own subdirectory to use the selector as published.")

    if not grouped:
        raise FileNotFoundError(f"no usable structures under {root}")

    source_names = list(grouped)
    paths, srcs, codes = [], [], []
    for i, s in enumerate(source_names):
        for p in grouped[s]:
            paths.append(p)
            srcs.append(s)
            codes.append(i)

    counts = {s: len(v) for s, v in grouped.items()}
    thin = {s: n for s, n in counts.items() if n < 2}
    if thin:
        warnings.append(
            f"sources with fewer than 2 poses: {thin}. Source-balanced support degenerates to plain "
            "pose-similarity support when a source contributes one pose, and within-source "
            "reproducibility is unobservable. Two or more poses per predictor is the operating regime.")
    if len(source_names) < 2:
        warnings.append(
            "only one source in the pool. The selector still ranks by pose-similarity support, but the "
            "cross-source agreement it was designed around is absent, so expect single-predictor "
            "performance.")
    return Pool(paths=paths, sources=srcs, source_names=source_names,
                codes=np.asarray(codes), warnings=warnings)
