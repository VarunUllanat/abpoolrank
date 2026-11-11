"""Per-pose interface descriptor.

One pass over a predicted complex produces everything the pairwise channels need: the epitope and
paratope residue sets, the antibody-antigen contact pairs, a soft (distance-weighted) epitope profile,
and CA coordinates split into antibody and antigen. Pair channels are then pure set/geometry operations
on two descriptors, so a pool of N poses costs N parses and N*(N-1)/2 cheap comparisons.

Extracted verbatim from the analysis code that produced the published pair-feature cache
(scripts/paper/method/xgen_pairfeats.py); the constants and conventions below are frozen because the
released selector was fitted against features computed exactly this way.

No native/reference structure is ever read here -- descriptors are computed from predictions alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Frozen geometry conventions. CUT is the heavy-atom contact cutoff in angstroms; SOFT_D0/SOFT_W are the
# midpoint and width of the logistic used for the soft epitope profile.
CUT, SOFT_D0, SOFT_W = 5.0, 5.0, 1.0

#: Chain IDs treated as the antibody. Everything else in the file is the antigen assembly.
DEFAULT_ANTIBODY_CHAINS = ("H", "L")

# A descriptor needs enough of both partners to define an interface and to superpose on later.
MIN_AB_CA, MIN_AG_CA = 10, 5


@dataclass(frozen=True)
class Descriptor:
    """Interface fingerprint of one predicted pose."""

    epitope: frozenset          # antigen residues contacted, as (chain, resid)
    contact_pairs: frozenset    # (antibody_residue, antigen_residue) pairs within CUT
    paratope: frozenset         # antibody residues contacting the antigen
    soft_epitope: dict          # antigen residue -> summed logistic contact weight
    ab_ca: dict                 # antibody (chain, resid) -> CA coordinate
    ag_ca: dict                 # antigen  (chain, resid) -> CA coordinate


def load_structure(path):
    """Read one model from a .pdb or .cif/.mmcif file into a biotite AtomArray.

    biotite warns once per file when an mmCIF omits `auth_atom_id` and it falls back to `label_atom_id`.
    Predictor output routinely omits it, so on a 100-pose pool that warning is pure noise; the fallback
    is the correct behaviour and the atom names it yields are the ones used throughout.
    """
    import warnings

    p = str(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if p.endswith((".pdb", ".ent")):
            import biotite.structure.io.pdb as pdb

            return pdb.get_structure(pdb.PDBFile.read(p), model=1)
        import biotite.structure.io.pdbx as pdbx

        return pdbx.get_structure(pdbx.CIFFile.read(p), model=1)


def describe(path, antibody_chains=DEFAULT_ANTIBODY_CHAINS) -> Descriptor | None:
    """Build the interface descriptor for one pose, or None if it is unusable.

    Returns None -- rather than raising -- when the file cannot be parsed, when either partner is
    absent, or when too few CA atoms are present to superpose. Callers treat a None descriptor as a
    pose that contributes no similarity, which is how unparseable poses were handled when the released
    features were built.
    """
    from scipy.spatial import cKDTree

    try:
        st = load_structure(path)
    except Exception:
        return None
    heavy = st[st.element != "H"]
    is_ab = np.isin(heavy.chain_id, list(antibody_chains))
    is_ag = ~is_ab
    if not is_ab.any() or not is_ag.any():
        return None

    A, G = heavy.coord[is_ab], heavy.coord[is_ag]
    ab_key = list(zip(heavy.chain_id[is_ab], heavy.res_id[is_ab]))
    ag_key = list(zip(heavy.chain_id[is_ag], heavy.res_id[is_ag]))

    soft: dict = {}
    epitope, pairs, paratope = set(), set(), set()
    neighbours = cKDTree(A).query_ball_tree(cKDTree(G), SOFT_D0 + 4 * SOFT_W)
    for i, js in enumerate(neighbours):
        for j in js:
            d = float(np.linalg.norm(A[i] - G[j]))
            w = 1.0 / (1.0 + np.exp((d - SOFT_D0) / SOFT_W))
            if w >= 1e-3:
                soft[ag_key[j]] = soft.get(ag_key[j], 0.0) + w
            if d <= CUT:
                epitope.add(ag_key[j])
                pairs.add((ab_key[i], ag_key[j]))
                paratope.add(ab_key[i])

    ca = st[st.atom_name == "CA"]
    abset = set(antibody_chains)
    ab_ca = {(c, int(r)): x for c, r, x in zip(ca.chain_id, ca.res_id, ca.coord) if c in abset}
    ag_ca = {(c, int(r)): x for c, r, x in zip(ca.chain_id, ca.res_id, ca.coord) if c not in abset}
    if len(ab_ca) < MIN_AB_CA or len(ag_ca) < MIN_AG_CA:
        return None
    return Descriptor(frozenset(epitope), frozenset(pairs), frozenset(paratope), soft, ab_ca, ag_ca)
