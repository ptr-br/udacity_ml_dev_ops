import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from census.data.process_data import process
from census.model.model import compute_model_metrics, inference, slice_performance, train_model


def make_toy_data(n=100):
    rng = np.random.RandomState(0)
    X = rng.randn(n, 4)
    y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
    return X, y


def test_train_model_returns_fitted_estimator():
    X, y = make_toy_data(200)
    model = train_model(X, y, random_state=0)
    assert isinstance(model, RandomForestClassifier)
    # model should be fitted: predict_proba exists and returns length n predictions
    preds = model.predict(X[:5])
    assert preds.shape[0] == 5


def test_compute_model_metrics_perfect_and_worst_cases():
    y = np.array([0, 1, 1, 0, 1])
    preds_perfect = np.array([0, 1, 1, 0, 1])
    p, r, f = compute_model_metrics(y, preds_perfect)
    assert pytest.approx(p) == 1.0
    assert pytest.approx(r) == 1.0
    assert pytest.approx(f) == 1.0

    preds_none = np.array([0, 0, 0, 0, 0])
    p, r, f = compute_model_metrics(y, preds_none)
    # precision = 1 (zero division handling), recall = 0 for no positives predicted
    assert p >= 0.0 and p <= 1.0
    assert pytest.approx(r) == 0.0


def test_inference_produces_valid_shape():
    X, y = make_toy_data(50)
    model = train_model(X, y, random_state=1)
    preds = inference(model, X)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_slice_performance_basic():
    # small DataFrame to exercise slice_performance
    df = pd.DataFrame(
        {
            "workclass": ["a", "a", "b", "b", "c", "c"],
            "education": ["x", "y", "x", "y", "x", "y"],
            "age": [25, 30, 22, 40, 50, 60],
            "salary": ["<=50k", ">50k", "<=50k", ">50k", "<=50k", ">50k"],
        }
    )

    cat_features = ["workclass", "education"]
    X, y, encoder, lb = process(
        df, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y, random_state=0)
    results = slice_performance(
        df, model, encoder, lb, categorical_features=cat_features, label="salary"
    )
    assert isinstance(results, dict)
    assert set(results["workclass"].keys()) == {"a", "b", "c"}
    assert set(results["education"].keys()) == {"x", "y"}
