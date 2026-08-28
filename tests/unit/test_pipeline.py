import os
import pytest
import numpy as np
import pandas as pd
import chokkhu as ck
from chokkhu.pipeline import PipelineResult


@pytest.fixture
def sample_classification_df():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "num_1": np.random.randn(n) * 10 + 50,
            "num_2": np.random.randn(n) * 2 - 5,
            "cat_1": np.random.choice(["A", "B", "C"], size=n),
            "target": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        }
    )
    return df


@pytest.fixture
def sample_regression_df():
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n) * 5
    x2 = np.random.randn(n) * 2
    df = pd.DataFrame(
        {
            "num_1": x1,
            "num_2": x2,
            "cat_1": np.random.choice(["low", "medium", "high"], size=n),
            "target": 3.0 * x1 - 1.5 * x2 + np.random.randn(n) * 0.5,
        }
    )
    return df


def test_pipeline_classification_auto(sample_classification_df):
    result = ck.pipeline(
        data=sample_classification_df,
        target="target",
        model="auto",
        task="classification",
        clean=True,
        preprocess=True,
        test_size=0.2,
        random_state=42,
        verbose=False,
    )
    assert isinstance(result, PipelineResult)
    assert result.task == "classification"
    assert result.model is not None
    assert result.model_name != ""
    assert "accuracy" in result.evaluation or len(result.evaluation) > 0

    # Test inference with zero leakage
    new_data = pd.DataFrame(
        {
            "num_1": [45.0, 55.0],
            "num_2": [-4.0, -6.0],
            "cat_1": ["A", "B"],
        }
    )
    preds = result.predict(new_data)
    assert len(preds) == 2


def test_pipeline_regression_auto(sample_regression_df):
    result = ck.pipeline(
        data=sample_regression_df,
        target="target",
        model="auto",
        task="regression",
        clean=True,
        preprocess=True,
        test_size=0.2,
        random_state=42,
        verbose=False,
    )
    assert isinstance(result, PipelineResult)
    assert result.task == "regression"
    assert result.model is not None

    new_data = pd.DataFrame(
        {
            "num_1": [2.0, -1.0],
            "num_2": [1.0, 0.5],
            "cat_1": ["low", "high"],
        }
    )
    preds = result.predict(new_data)
    assert len(preds) == 2
    assert isinstance(preds[0], (float, np.floating, int, np.integer))


def test_pipeline_with_transformation_and_resampling(sample_classification_df):
    result = ck.pipeline(
        data=sample_classification_df,
        target="target",
        model="random_forest",
        task="classification",
        transform={"pca": 2},
        resample="smote",
        test_size=0.2,
        random_state=42,
        verbose=False,
    )
    assert isinstance(result, PipelineResult)
    assert result.transformation_state is not None
    assert result.transformation_state.pca_model is not None

    # Verify predictions work seamlessly through PCA and Preprocessor
    new_data = sample_classification_df.drop(columns=["target"]).iloc[:3]
    preds = result.predict(new_data)
    assert len(preds) == 3


def test_pipeline_summary_and_serialization(sample_classification_df, tmp_path):
    result = ck.pipeline(
        data=sample_classification_df,
        target="target",
        model="logistic_regression",
        task="classification",
        clean=True,
        preprocess=True,
        test_size=0.2,
        random_state=42,
        verbose=False,
    )
    summary_str = result.summary()
    assert "CHOKKHU END-TO-END PIPELINE SUMMARY" in summary_str
    assert "Selected Model" in summary_str

    save_path = str(tmp_path / "pipeline.pkl")
    result.save(save_path)
    assert os.path.exists(save_path)

    loaded_result = PipelineResult.load(save_path)
    assert loaded_result.model_name == result.model_name

    new_data = sample_classification_df.drop(columns=["target"]).iloc[:2]
    orig_preds = result.predict(new_data)
    loaded_preds = loaded_result.predict(new_data)
    np.testing.assert_array_equal(orig_preds, loaded_preds)
