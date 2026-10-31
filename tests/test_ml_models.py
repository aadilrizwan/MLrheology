"""
Unit tests for the Conventional Machine Learning module.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import MLConfig
from src.data_processing import generate_synthetic_dataset
from src.ml_models import (
    build_preprocessor,
    create_group_labels,
    evaluate_model,
    generalization_experiment,
    prepare_features,
    split_data,
    train_random_forest,
    train_xgboost,
)


@pytest.fixture
def sample_df():
    return generate_synthetic_dataset(seed=42)


class TestFeaturePreparation:

    def test_prepare_features_auto_detect(self, sample_df):
        X, y, num_f, cat_f = prepare_features(sample_df)
        assert len(X) == len(sample_df)
        assert len(y) == len(sample_df)
        assert "shear_rate" in num_f
        assert "concentration" in num_f
        assert "temperature" in num_f
        assert "fluid_type" in cat_f

    def test_categorical_detection_with_string_and_category(self):
        df = pd.DataFrame({
            "shear_rate": [1.0, 10.0, 100.0],
            "fluid_type": pd.Series(["XG", "CMC", "XG"], dtype="string"),
            "category_col": pd.Series(["A", "B", "A"], dtype="category"),
            "apparent_viscosity": [0.5, 0.2, 0.05],
        })
        X, y, num_f, cat_f = prepare_features(
            df, feature_cols=["shear_rate", "fluid_type", "category_col"]
        )
        assert num_f == ["shear_rate"]
        assert "fluid_type" in cat_f
        assert "category_col" in cat_f

    def test_missing_target_raises(self, sample_df):
        with pytest.raises(ValueError, match="Target column"):
            prepare_features(sample_df, target_col="non_existent")


class TestSplittingAndGrouping:

    def test_grouped_split_no_leakage(self, sample_df):
        """Groups must be mutually exclusive between train and test."""
        X, y, _, _ = prepare_features(sample_df)
        groups = create_group_labels(sample_df)
        assert groups is not None

        X_train, X_test, y_train, y_test = split_data(
            X, y, groups=groups, test_size=0.25, random_state=42
        )

        train_groups = set(groups.iloc[X_train.index])
        test_groups = set(groups.iloc[X_test.index])

        # Strict disjoint check
        assert train_groups.isdisjoint(test_groups), "Data leakage: groups overlap!"
        assert len(X_train) + len(X_test) == len(X)


class TestModelTrainingAndEvaluation:

    def test_random_forest_training(self, sample_df):
        X, y, num_f, cat_f = prepare_features(sample_df)
        config = MLConfig(n_estimators_rf=20, random_state=42)
        pipe, train_metrics = train_random_forest(X, y, num_f, cat_f, config=config)
        assert train_metrics["train_R2"] > 0.90

        y_pred, test_metrics = evaluate_model(pipe, X, y, model_name="RF")
        assert (y_pred > 0).all(), "Viscosity predictions must be strictly positive"
        assert test_metrics["test_R2"] > 0.90

    def test_xgboost_training(self, sample_df):
        X, y, num_f, cat_f = prepare_features(sample_df)
        config = MLConfig(n_estimators_xgb=20, random_state=42)
        pipe, train_metrics = train_xgboost(X, y, num_f, cat_f, config=config)
        assert train_metrics["train_R2"] > 0.90

        y_pred, test_metrics = evaluate_model(pipe, X, y, model_name="XGB")
        assert (y_pred > 0).all(), "Viscosity predictions must be strictly positive"
        assert test_metrics["test_R2"] > 0.90

    def test_negative_prediction_clipping(self):
        """Dummy estimator returning negative predictions should be clipped to positive bound."""
        from sklearn.base import BaseEstimator, RegressorMixin

        class NegativeDummy(BaseEstimator, RegressorMixin):
            def fit(self, X, y):
                return self
            def predict(self, X):
                return np.array([-5.0, -1.0, 2.0])

        pipe = NegativeDummy()
        X_dummy = pd.DataFrame({"feat": [1, 2, 3]})
        y_dummy = pd.Series([1.0, 2.0, 3.0])

        y_pred, metrics = evaluate_model(pipe, X_dummy, y_dummy, model_name="Dummy")
        assert (y_pred >= 1e-10).all()


class TestGeneralizationExperiment:

    def test_generalization_execution(self, sample_df):
        config = MLConfig(n_estimators_rf=10, n_estimators_xgb=10, random_state=42)
        results = generalization_experiment(
            sample_df, holdout_col="concentration", config=config
        )
        assert len(results) > 0
        for res in results:
            assert "holdout_value" in res
            assert "RF_metrics" in res
            assert "XGB_metrics" in res
            assert res["n_train"] > 0
            assert res["n_test"] > 0

