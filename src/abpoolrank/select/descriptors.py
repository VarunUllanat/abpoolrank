"""Per-pose interface descriptor.

`describe` parses one predicted complex into the epitope, paratope and contact-pair residue sets, a
distance-weighted epitope profile, and antibody and antigen CA coordinates. `pairwise` compares these
descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# CUT is the heavy-atom contact cutoff in angstroms; SOFT_D0/SOFT_W are the midpoint and width of the
# logistic used for the soft epitope profile. The bundled models were trained with these values.
CUT, SOFT_D0, SOFT_W = 5.0, 5.0, 1.0

#: Chain IDs treated as the antibody. Everything else in the file is the antigen assembly.
DEFAULT_ANTIBODY_CHAINS = ("H", "L")

# Minimum CA atoms per partner for a usable descriptor.
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
    """Read the first model of a .pdb/.ent or .cif/.mmcif file into a biotite AtomArray.

    biotite's per-file UserWarnings (e.g. the missing `auth_atom_id` fallback) are suppressed.
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

    Returns None when the file cannot be parsed, when either partner is absent, or when too few CA
    atoms are present. A None descriptor has zero similarity to every other pose.
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
