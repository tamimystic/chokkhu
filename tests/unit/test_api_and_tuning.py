import numpy as np
import pandas as pd
import chokkhu as ck


def test_train_hyperparameter_tuning():
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = ck.train(
        model="random_forest",
        X_train=X,
        y_train=y,
        task="classification",
        tune=True,
        param_grid={"n_estimators": [5, 10], "max_depth": [2, 4]},
        cv=2,
        random_state=42,
        verbose=False,
    )
    assert hasattr(model, "best_params_")
    assert "n_estimators" in model.best_params_
    assert "max_depth" in model.best_params_

    preds = model.predict(X[:5])
    assert len(preds) == 5


def test_clean_with_str_path(tmp_path):
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 20.0, 30.0]})
    df.to_csv(csv_file, index=False)

    cleaned = ck.clean(str(csv_file), missing="mean", verbose=False)
    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.isna().any().any()


def test_evaluate_engine_with_probabilities():
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)

    model = ck.train("logistic_regression", X_train=X, y_train=y, verbose=False)
    eval_dict = ck.evaluate(model, X, y, save_reports=False)

    assert isinstance(eval_dict, dict)
    assert "accuracy" in eval_dict
    assert "precision" in eval_dict
    assert "f1_score" in eval_dict
    assert "roc_auc" in eval_dict
