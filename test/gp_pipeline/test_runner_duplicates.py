import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from deap import creator, gp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gp_pipeline.config import (  # noqa: E402
    BackendConfig,
    DataConfig,
    FeatureConfig,
    FitnessConfig,
    GPConfig,
    OutputConfig,
    PopulationConfig,
    TimeframePrimitiveConfig,
    ValidatorConfig,
)
from gp_pipeline.runner import GPFeatureTrainer  # noqa: E402


def _make_trainer(base_dir: Path) -> GPFeatureTrainer:
    base_dir.mkdir(parents=True, exist_ok=True)
    data_path = base_dir / "data.csv"
    artifacts_path = base_dir / "artifacts"
    artifacts_path.mkdir(exist_ok=True)
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0, 4.0],
            "Close": [2.0, 3.0, 4.0, 5.0],
        }
    )
    df.to_csv(data_path, index=False)

    cfg = GPConfig(
        data=DataConfig(
            path=data_path,
            target_column="Close",
            feature_columns=("Open", "Close"),
        ),
        feature=FeatureConfig(mode="numeric"),
        timeframe_primitives=TimeframePrimitiveConfig(enabled=False),
        population=PopulationConfig(
            size=2,
            generations=1,
            min_depth=1,
            max_depth=2,
            mutation_prob=0.0,
            crossover_prob=0.0,
            elitism=0,
            seed=123,
        ),
        backend=BackendConfig(
            primary="numpy",
            fallbacks=(),
            cache_subtrees=False,
            low_fidelity_ratio=0.0,
            full_eval_top_k=0.0,
        ),
        fitness=FitnessConfig(),
        validator=ValidatorConfig(enabled=False, numeric_kwargs={}, boolean_kwargs={}),
        output=OutputConfig(artifacts_dir=artifacts_path, save_best=5, save_population=False),
    ).resolve()
    return GPFeatureTrainer(cfg)


def test_value_signature_prefers_simpler(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path / "case_simple_pref")

    objectives_template = {
        "neg_log_pvalue": 2.0,
        "effect_size": 1.0,
        "cliffs_delta": 1.0,
        "significance_margin": 1.0,
        "mean_method_effect": 1.0,
        "effect_stability": 1.0,
        "pvalue_consistency": 1.0,
        "min_pvalue": 0.01,
    }
    goals_template = {
        name: ("minimize" if name == "min_pvalue" else "maximize")
        for name in objectives_template
    }

    def fake_validator(self, df_param, values_param):
        return SimpleNamespace(
            fitness_objectives=dict(objectives_template),
            fitness_goals=dict(goals_template),
            fitness_metrics=dict(objectives_template),
        )

    monkeypatch.setattr(GPFeatureTrainer, "_run_validator", fake_validator)

    pset = trainer.toolbox.pset
    complex_tree = gp.PrimitiveTree.from_string("max(COL_Close, COL_Open)", pset)
    simple_tree = gp.PrimitiveTree.from_string("COL_Close", pset)

    complex_ind = creator.GPIndividual(complex_tree)
    simple_ind = creator.GPIndividual(simple_tree)

    trainer._evaluate_population([complex_ind, simple_ind], generation=0)

    assert getattr(complex_ind, "is_duplicate", False) is True
    assert getattr(simple_ind, "is_duplicate", False) is False
    assert simple_ind.full_signature is not None
    assert trainer._full_signatures[simple_ind.full_signature].individual is simple_ind

    record_values = list(trainer.best_records.values())
    assert len(record_values) == 1
    assert record_values[0].expression == "Close"
    assert record_values[0].signature == simple_ind.full_signature

    complex_metrics = getattr(complex_ind, "metrics", {})
    assert complex_metrics, "duplicate individual should hold worst metrics"
    assert np.isneginf(complex_ind.fitness.values[0])

    # Introducing another complex duplicate should be rejected immediately.
    complex_ind_2 = creator.GPIndividual(
        gp.PrimitiveTree.from_string("max(COL_Close, COL_Open)", pset)
    )
    trainer._evaluate_population([complex_ind_2], generation=1)
    assert getattr(complex_ind_2, "is_duplicate", False) is True
    assert np.isneginf(complex_ind_2.fitness.values[0])
    assert len(trainer.best_records) == 1


def test_vector_signature_distinguishes_nonfinite(tmp_path):
    trainer = _make_trainer(tmp_path / "case_nonfinite")

    zeros = np.array([0.0, 1.0, -2.5], dtype=np.float64)
    zeros_sig = trainer._vector_signature(zeros)

    with_nan = np.array([np.nan, 1.0, -2.5], dtype=np.float64)
    nan_sig = trainer._vector_signature(with_nan)
    assert nan_sig != zeros_sig

    with_posinf = np.array([np.inf, 1.0, -2.5], dtype=np.float64)
    posinf_sig = trainer._vector_signature(with_posinf)
    assert posinf_sig != zeros_sig
    assert posinf_sig != nan_sig

    with_neginf = np.array([-np.inf, 1.0, -2.5], dtype=np.float64)
    neginf_sig = trainer._vector_signature(with_neginf)
    assert neginf_sig != zeros_sig
    assert neginf_sig != nan_sig
    assert neginf_sig != posinf_sig
