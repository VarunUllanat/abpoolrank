"""Regression tests for the released selector.

The equivalence tests are the important ones: they pin the refactored package to the implementation that
produced the published numbers, so a future change to descriptors, pair channels or features cannot
silently move the results.

Tests needing the study's cached artifacts skip when those are absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cofoldrank.select import FEATURES, PoolSelector, discover, pool_features
from cofoldrank.select.features import TOPK, percentile, support

CACHE = Path("/data/cb/varunullanat/cofoldrank_data/features/xgen_pair")
COHORT = Path("/data/cb/varunullanat/cofoldrank_out/results/fig2_cohort.parquet")
DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo_pool"


# --------------------------------------------------------------------------- unit
def test_percentile_is_rank_scaled_to_unit_interval():
    assert np.allclose(percentile([5.0, 1.0, 3.0]), [1.0, 0.0, 0.5])
    assert np.allclose(percentile([7.0]), [0.0])          # a pool of one has no ordering


def test_srcbal_equals_pss_at_one_pose_per_source():
    """The B=5 degeneracy: with m=1 the top-k mean has k=1, so the two features coincide."""
    rng = np.random.default_rng(0)
    n = 5
    sim = rng.random((n, n))
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 0.0)
    pss, srcbal = support(sim, np.arange(n), n)
    # srcbal averages over the G-1 other sources; that constant cancels in the percentile
    assert np.allclose(percentile(pss), percentile(srcbal))


def test_srcbal_differs_from_pss_at_two_poses_per_source():
    rng = np.random.default_rng(1)
    n = 10
    sim = rng.random((n, n))
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 0.0)
    pss, srcbal = support(sim, np.repeat(np.arange(5), 2), 5)
    assert not np.allclose(percentile(pss), percentile(srcbal))


def test_features_are_invariant_to_source_count_scaling():
    """1/(G-1) is constant within a pool, so it cannot survive the percentile. This is why a selector
    fitted on five sources transfers to a pool with six."""
    rng = np.random.default_rng(2)
    for g in (4, 5, 6):
        n = 2 * g
        sim = rng.random((n, n))
        sim = (sim + sim.T) / 2
        np.fill_diagonal(sim, 0.0)
        _, srcbal = support(sim, np.repeat(np.arange(g), 2), g)
        scaled = srcbal * (g - 1)
        assert np.allclose(percentile(srcbal), percentile(scaled))


def test_pool_features_shape_and_column_order():
    n, g = 10, 5
    codes = np.repeat(np.arange(g), 2)
    X = pool_features(np.zeros((n, n)), np.zeros((n, n)), codes, g)
    assert X.shape == (n, len(FEATURES))
    assert np.allclose(X[:, 3], np.log(n))                # log_pool is the last column


# ------------------------------------------------------------- equivalence with the paper code
# Frozen output of the implementation that produced the published results, for the fixed synthetic pool
# built below (numpy default_rng(0), 25 poses, 5 sources of 5). Values are exact multiples of 1/24, so
# equality is safe. If a change to features.py moves these, it has changed the published method.
REFERENCE_FEATURES = (
    (0.2916666666666667, 0.4583333333333333, 0.125),
    (0.75, 0.625, 0.4583333333333333),
    (0.08333333333333333, 0.25, 0.625),
    (0.9583333333333334, 0.9166666666666666, 0.5416666666666666),
    (0.3333333333333333, 0.16666666666666666, 1.0),
    (0.875, 0.7083333333333334, 0.8333333333333334),
    (0.4583333333333333, 0.375, 0.08333333333333333),
    (1.0, 1.0, 0.4166666666666667),
    (0.041666666666666664, 0.0, 0.25),
    (0.9166666666666666, 0.7916666666666666, 0.5),
    (0.7916666666666666, 0.6666666666666666, 0.7083333333333334),
    (0.6666666666666666, 0.2916666666666667, 0.2916666666666667),
    (0.7083333333333334, 0.8333333333333334, 0.9583333333333334),
    (0.0, 0.08333333333333333, 0.75),
    (0.4166666666666667, 0.75, 0.7916666666666666),
    (0.25, 0.041666666666666664, 0.041666666666666664),
    (0.8333333333333334, 0.875, 0.0),
    (0.375, 0.3333333333333333, 0.9166666666666666),
    (0.5, 0.5, 0.5833333333333334),
    (0.625, 0.5416666666666666, 0.20833333333333334),
    (0.5416666666666666, 0.9583333333333334, 0.875),
    (0.20833333333333334, 0.125, 0.6666666666666666),
    (0.16666666666666666, 0.4166666666666667, 0.3333333333333333),
    (0.125, 0.20833333333333334, 0.16666666666666666),
    (0.5833333333333334, 0.5833333333333334, 0.375),
)


def _fixture_pool():
    """The exact synthetic pool the reference features above were computed on."""
    rng = np.random.default_rng(0)
    n, g = 25, 5
    codes = np.repeat(np.arange(g), 5)
    S = rng.random((n, n)); S = (S + S.T) / 2; np.fill_diagonal(S, 0.0)
    E = rng.random((n, n)); E = (E + E.T) / 2; np.fill_diagonal(E, 0.0)
    return S, E, codes, g


def test_features_match_published_implementation():
    """pool_features must reproduce the features behind the published numbers, exactly."""
    S, E, codes, g = _fixture_pool()
    X = pool_features(S, E, codes, g)
    assert X.shape == (25, 4)
    assert np.array_equal(X[:, :3], np.array(REFERENCE_FEATURES))
    assert np.allclose(X[:, 3], np.log(25))
    assert TOPK == 3                                      # the frozen top-k of the source-balanced stat


def test_selector_scores_are_reproducible():
    """The bundled checkpoint must keep scoring the fixture pool the same way."""
    S, E, codes, g = _fixture_pool()
    scores = PoolSelector().model.predict(pool_features(S, E, codes, g))
    assert scores.shape == (25,)
    assert np.isfinite(scores).all()
    # seed-averaged over three fitted models, so identical inputs give identical outputs
    again = PoolSelector().model.predict(pool_features(S, E, codes, g))
    assert np.array_equal(scores, again)


@pytest.mark.skipif(not (CACHE.exists() and COHORT.exists()),
                    reason="study prediction cache not available")
def test_pair_channels_reproduce_published_cache():
    """Recomputing from the original structure files must match the frozen pair-feature cache."""
    import polars as pl

    from cofoldrank.select.pairwise import matrices_from_paths

    target = "pdb_00009e6k_H_L"
    npz = CACHE / f"{target}.npz"
    if not npz.exists():
        pytest.skip(f"{target} not cached")
    z = np.load(npz)
    ids = [str(x) for x in z["cand_ids"]]
    coh = (pl.read_parquet(COHORT).filter(pl.col("target_id") == target)
           .sort(["target_id", "folder", "cand_idx"]))

    sub = []
    for src in ["boltz2", "promera", "protenix_v2", "esmfold2_fast", "opendde"]:
        sub += [i for i, x in enumerate(ids) if x.split("|")[1] == src][:3]
    paths = [coh["struct_path"][i] for i in sub]

    M, desc = matrices_from_paths(paths, channels=("geom_sim", "epi_jac"))
    assert all(d is not None for d in desc)
    ix = np.ix_(sub, sub)
    assert np.array_equal(z["epi_jac"][ix], M["epi_jac"])
    # geom_sim is stored float32; allow only float32 rounding
    assert np.abs(z["geom_sim"][ix] - M["geom_sim"]).max() < 1e-6


# --------------------------------------------------------------------------- API
@pytest.mark.skipif(not DEMO.exists(), reason="run examples/make_demo_pool.py first")
def test_discover_partitions_by_subdirectory():
    pool = discover(DEMO)
    assert pool.n_sources == 5
    assert set(pool.counts().values()) == {2}
    assert len(pool) == 10
    assert not pool.warnings                              # 5 sources x 2 poses is the intended regime


@pytest.mark.skipif(not DEMO.exists(), reason="run examples/make_demo_pool.py first")
def test_ranking_is_complete_ordered_and_source_label_invariant(tmp_path):
    import shutil

    selector = PoolSelector()
    result = selector.rank(DEMO)
    assert len(result.candidates) == 10
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert result.best is result.candidates[0]
    assert result.table.height == 10

    # renaming the predictor directories must not change the ranking: the selector is given the
    # partition, never the identity
    renamed = tmp_path / "renamed"
    for i, src in enumerate(result.pool.source_names):
        shutil.copytree(DEMO / src, renamed / f"model_{i}")
    other = selector.rank(renamed)
    assert np.allclose(sorted(other.scores), sorted(result.scores))


@pytest.mark.skipif(not DEMO.exists(), reason="run examples/make_demo_pool.py first")
def test_single_pose_per_source_warns_about_degeneracy():
    selector = PoolSelector()
    with pytest.warns(RuntimeWarning, match="one pose|fewer than 2"):
        selector.rank(DEMO, max_per_source=1)


def test_missing_pool_directory_raises():
    with pytest.raises(NotADirectoryError):
        discover("/nonexistent/path/for/tests")


def test_too_few_poses_raises(tmp_path):
    d = tmp_path / "pool" / "only"
    d.mkdir(parents=True)
    (d / "a.pdb").write_text("")
    with pytest.raises(ValueError, match="at least 2 poses"):
        PoolSelector().rank_pool(discover(tmp_path / "pool"))


# --------------------------------------------------------------------------- bundled model choice
def test_bundled_models_load_and_differ():
    """'both', 'paired-only' and 'vhh-only' all ship with the package and are distinct selectors."""
    from cofoldrank.select.model import MODELS, SelectorModel

    S, E, codes, g = _fixture_pool()
    X = pool_features(S, E, codes, g)
    scores = {}
    for name in MODELS:
        m = SelectorModel.load(model=name)
        assert m.features == FEATURES
        scores[name] = m.predict(X)
        assert np.isfinite(scores[name]).all()
    assert np.array_equal(scores["both"], SelectorModel.load().predict(X))   # default is "both"
    assert not np.array_equal(scores["both"], scores["paired-only"])
    assert not np.array_equal(scores["both"], scores["vhh-only"])
    assert np.array_equal(PoolSelector(model="vhh-only").model.predict(X), scores["vhh-only"])


def test_model_choice_rejects_bad_input(monkeypatch):
    from cofoldrank.select.model import MODELS, SelectorModel

    with pytest.raises(ValueError, match="unknown model"):
        SelectorModel.load(model="nanobody")
    with pytest.raises(ValueError, match="not both"):
        SelectorModel.load(path=MODELS["both"], model="vhh-only")
    # a named model is loaded as named even when the environment points elsewhere
    monkeypatch.setenv("COFOLDRANK_SELECTOR", str(MODELS["paired-only"]))
    S, E, codes, g = _fixture_pool()
    X = pool_features(S, E, codes, g)
    assert np.array_equal(SelectorModel.load(model="vhh-only").predict(X),
                          SelectorModel.load(path=MODELS["vhh-only"]).predict(X))
    assert np.array_equal(SelectorModel.load().predict(X),                 # unnamed: env still honoured
                          SelectorModel.load(path=MODELS["paired-only"]).predict(X))


def test_cli_model_option():
    from cofoldrank.cli import build_parser

    p = build_parser()
    assert p.parse_args(["select", "d"]).model is None
    assert p.parse_args(["select", "d", "--model", "vhh-only"]).model == "vhh-only"
    with pytest.raises(SystemExit):
        p.parse_args(["select", "d", "--model", "vhh-only", "--checkpoint", "x.joblib"])
    with pytest.raises(SystemExit):
        p.parse_args(["select", "d", "--model", "camelid"])
