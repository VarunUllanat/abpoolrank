"""All-pairs similarity channels between predicted poses.

Two poses are compared by superposing their ANTIBODY frameworks on shared residues and then asking how
far the antigen moved. That makes every channel invariant to global placement and sensitive only to the
relative docking geometry, which is the quantity that actually distinguishes epitope modes.

Only `geom_sim` and `epi_jac` feed the released selector; the other five channels are computed because
they are cheap given the descriptors and are used by the feature ablation reported in the paper. A
seven-channel selector was measured to be statistically indistinguishable from the four-feature one.

No native structure is involved: these are prediction-to-prediction comparisons, never DockQ.
"""
from __future__ import annotations

import numpy as np

from .descriptors import Descriptor, describe

#: Channel names, in the order they were frozen when the published cache was built.
CHANNELS = ("epi_jac", "cpair_jac", "geom_sim", "para_jac", "com_dist", "rel_rot", "soft_epi")

# Superposition needs enough shared residues to be meaningful.
MIN_SHARED_AB, MIN_SHARED_AG = 10, 5


def _kabsch(P, Q):
    """Rotation taking centred P onto centred Q."""
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def soft_jaccard(ka: dict, kb: dict) -> float:
    """Weighted Jaccard over the soft epitope profiles."""
    inter = sum(min(ka[k], kb[k]) for k in set(ka) & set(kb))
    union = sum(ka.values()) + sum(kb.values()) - inter
    return float(inter / union) if union > 0 else 0.0


def pair_channels(di: Descriptor, dj: Descriptor) -> dict:
    """All seven channels for one ordered pair of descriptors (they are symmetric)."""
    geom = com = rot = 0.0
    shared_ab = [k for k in sorted(di.ab_ca) if k in dj.ab_ca]
    if len(shared_ab) >= MIN_SHARED_AB:
        P = np.array([dj.ab_ca[k] for k in shared_ab])
        Q = np.array([di.ab_ca[k] for k in shared_ab])
        pc, qc = P.mean(0), Q.mean(0)
        R = _kabsch(P - pc, Q - qc)
        shared_ag = [k for k in sorted(di.ag_ca) if k in dj.ag_ca]
        if len(shared_ag) >= MIN_SHARED_AG:
            X = (np.array([dj.ag_ca[k] for k in shared_ag]) - pc) @ R.T + qc
            Y = np.array([di.ag_ca[k] for k in shared_ag])
            com = float(np.linalg.norm(X.mean(0) - Y.mean(0)))          # RAW angstroms
            R2 = _kabsch(X - X.mean(0), Y - Y.mean(0))
            rot = float(np.arccos(np.clip((np.trace(R2) - 1) / 2, -1, 1)) / np.pi)
            geom = float(1.0 / (1.0 + float(np.sqrt(((X - Y) ** 2).sum(1).mean()))))
    return {"epi_jac": jaccard(di.epitope, dj.epitope),
            "cpair_jac": jaccard(di.contact_pairs, dj.contact_pairs),
            "geom_sim": geom,
            "para_jac": jaccard(di.paratope, dj.paratope),
            "com_dist": com,
            "rel_rot": rot,
            "soft_epi": soft_jaccard(di.soft_epitope, dj.soft_epitope)}


def pair_matrices(descriptors, channels=CHANNELS) -> dict:
    """Symmetric (n, n) matrix per channel. A None descriptor yields zero similarity to everything."""
    n = len(descriptors)
    M = {c: np.zeros((n, n), dtype=np.float32) for c in channels}
    for i in range(n):
        di = descriptors[i]
        if di is None:
            continue
        for j in range(i + 1, n):
            dj = descriptors[j]
            if dj is None:
                continue
            vals = pair_channels(di, dj)
            for c in channels:
                M[c][i, j] = M[c][j, i] = vals[c]
    return M


def matrices_from_paths(paths, antibody_chains=("H", "L"), workers=1, channels=CHANNELS):
    """Descriptors then all-pairs matrices, optionally parsing structures in parallel.

    Parsing dominates runtime, so it is the part worth parallelising. joblib's loky backend is used
    rather than a raw ProcessPoolExecutor because the latter's spawn context re-imports __main__ in each
    worker, which fails outright when the caller is a notebook, a REPL, or a `python -` script.

    Returns (matrices, descriptors) so the caller can report which poses failed to parse.
    """
    paths = [str(p) for p in paths]
    ab = tuple(antibody_chains)
    if workers and workers > 1 and len(paths) > 1:
        from joblib import Parallel, delayed

        descriptors = Parallel(n_jobs=workers, backend="loky")(
            delayed(describe)(p, ab) for p in paths)
    else:
        descriptors = [describe(p, ab) for p in paths]
    return pair_matrices(descriptors, channels), descriptors
