import os
import numpy as np
import pandas as pd
import chokkhu as ck
from chokkhu.pipeline import PipelineResult


def test_full_lifecycle_workflow(tmp_path):
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "num1": np.random.randn(n) * 10 + 50,
            "num2": np.random.randn(n) * 2,
            "skewed": np.exp(np.random.randn(n)),
            "cat1": np.random.choice(["Low", "Medium", "High"], size=n),
            "cat2": np.random.choice(["TypeA", "TypeB"], size=n),
            "target": np.random.choice([0, 1], size=n),
        }
    )
    # Introduce missing values & outliers
    df.loc[0:4, "num1"] = np.nan
    df.loc[10, "num2"] = 500.0

    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)

    # 1. Load
    loaded_df = ck.load(str(csv_path))
    assert loaded_df.shape == (n, 6)

    # 2. Clean
    cleaned_df = ck.clean(loaded_df, missing="median", outliers="iqr", verbose=False)
    assert not cleaned_df.isna().any().any()

    # 3. Preprocess
    proc_df, prep_state = ck.preprocess(
        cleaned_df,
        target="target",
        scale="standard",
        encode="onehot",
        select_features="correlation",
        verbose=False,
    )
    assert "target" in proc_df.columns

    # 4. Transform
    trans_df = ck.transform(
        proc_df, target="target", pca=3, resample="smote", smote_k=2, verbose=False
    )
    assert "target" in trans_df.columns

    # 5. Split
    X_train, X_test, y_train, y_test = ck.split(
        trans_df, target="target", test_size=0.2, random_state=42
    )
    assert len(X_train) + len(X_test) == len(trans_df)

    # 6. Train with Hyperparameter Tuning
    model = ck.train(
        model="random_forest",
        X_train=X_train,
        y_train=y_train,
        tune=True,
        param_grid={"n_estimators": [5, 10], "max_depth": [2, 4]},
        cv=2,
        random_state=42,
        verbose=False,
    )
    assert hasattr(model, "best_params_")

    # 7. Train Deep Learning Neural Network
    dl_model = ck.train(
        model="neural_network",
        X_train=X_train,
        y_train=y_train,
        task="classification",
        layers=[16, 8],
        epochs=20,
        learning_rate=0.05,
        random_state=42,
        verbose=False,
    )
    dl_preds = dl_model.predict(X_test)
    assert len(dl_preds) == len(X_test)

    # 8. Evaluate
    eval_res = ck.evaluate(model, X_test, y_test, save_reports=False)
    assert "accuracy" in eval_res
    assert "roc_auc" in eval_res

    # 9. Explain
    exp_imp = ck.explain(
        model, X_test, y_test, method="feature_importance", n_repeats=2, verbose=False
    )
    assert exp_imp.importances is not None
    assert len(exp_imp.importances) == X_test.shape[1]

    exp_shap = ck.explain(model, X_test, method="shap", n_samples=10, verbose=False)
    assert exp_shap.shap_values is not None

    exp_pdp = ck.explain(model, X_test, method="pdp", pdp_feature=0, verbose=False)
    assert "grid_values" in exp_pdp.pdp_data

    # 10. Automated Leak-Free Pipeline
    pipe_res = ck.pipeline(
        data=str(csv_path),
        target="target",
        model="random_forest",
        clean_missing="median",
        clean_outliers="iqr",
        scale="robust",
        encode="onehot",
        pca=2,
        resample="smote",
        test_size=0.2,
        random_state=42,
        verbose=False,
    )
    assert isinstance(pipe_res, PipelineResult)
    assert pipe_res.metrics["accuracy"] > 0.0

    # Save & Load pipeline
    pipe_file = tmp_path / "pipeline.pkl"
    pipe_res.save(str(pipe_file))
    assert os.path.exists(pipe_file)

    loaded_pipe = PipelineResult.load(str(pipe_file))
    new_data = pd.DataFrame(
        {
            "num1": [55.0, 48.0],
            "num2": [1.2, -0.5],
            "skewed": [2.5, 1.1],
            "cat1": ["Low", "High"],
            "cat2": ["TypeA", "TypeB"],
        }
    )
    preds_new = loaded_pipe.predict(new_data)
    assert len(preds_new) == 2
