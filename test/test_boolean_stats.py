import warnings

import numpy as np
import pandas as pd
import pytest

from selection.boolean_stats import (
    BooleanFeatureStats,
    _fisher_exact,
    compute_boolean_feature_stats,
)


def _simple_dataset():
    target = pd.Series([1, 1, 0, 0, 1, 0], name="target")
    df = pd.DataFrame(
        {
            "f_bool": [1, 1, 1, 0, 0, 0],
            "f_noise": [0, 1, 0, 1, 0, 1],
        }
    )
    return df, target


def test_boolean_stats_counts_and_pvalues_match_manual():
    df, target = _simple_dataset()
    stats = compute_boolean_feature_stats(df, target, ["f_bool", "f_noise"], bootstrap=None)

    bool_stats: BooleanFeatureStats = stats["f_bool"]
    noise_stats: BooleanFeatureStats = stats["f_noise"]

    assert bool_stats.counts == {"a": 2, "b": 1, "c": 1, "d": 2}
    assert noise_stats.counts == {"a": 1, "b": 2, "c": 2, "d": 1}

    # Fisher exact should reduce to 2x2 hypergeometric tail; deterministic check.
    assert pytest.approx(_fisher_exact(2, 1, 1, 2)) == bool_stats.fisher_pvalue
    assert pytest.approx(_fisher_exact(1, 2, 2, 1)) == noise_stats.fisher_pvalue


def test_boolean_stats_bootstrap_respects_rng():
    df, target = _simple_dataset()
    params = {"samples": 16, "block_size": 3, "alpha": 0.1}

    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)

    stats1 = compute_boolean_feature_stats(df, target, ["f_bool"], bootstrap=params, rng=rng1)
    stats2 = compute_boolean_feature_stats(df, target, ["f_bool"], bootstrap=params, rng=rng2)

    ci1 = stats1["f_bool"].bootstrap_ci
    ci2 = stats2["f_bool"].bootstrap_ci
    assert np.allclose(ci1, ci2)

    rng3 = np.random.default_rng(9876)
    stats3 = compute_boolean_feature_stats(df, target, ["f_bool"], bootstrap=params, rng=rng3)
    ci3 = stats3["f_bool"].bootstrap_ci

    # With a distinct seed, the bootstrap interval should typically differ.
    assert not np.allclose(ci1, ci3)


def test_boolean_stats_emits_no_math_utils_warning():
    df, target = _simple_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        compute_boolean_feature_stats(df, target, ["f_bool"], bootstrap=None)
