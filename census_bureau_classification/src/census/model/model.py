from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """
    Train a RandomForestClassifier on the provided data.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels (binary).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    model : RandomForestClassifier
        Fitted RandomForestClassifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute precision, recall and F1 (fbeta with beta=1).

    Parameters
    ----------
    y : np.ndarray
        True binary labels.
    preds : np.ndarray
        Predicted binary labels.

    Returns
    -------
    precision, recall, fbeta
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """
    Run model inference and return binary predictions.

    Parameters
    ----------
    model : RandomForestClassifier
        Fitted model.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    preds : np.ndarray
        Binary predictions (0/1).
    """
    preds = model.predict(X)
    return np.array(preds)


def slice_performance(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    encoder,
    lb,
    categorical_features: Iterable[str],
    label: str = "salary",
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    Compute model performance metrics for each value of the given categorical features.

    For each categorical feature and each distinct value v, this function:
      - selects rows where feature == v
      - processes the slice with the provided encoder/lb (assumes process_data style)
      - runs inference and computes precision/recall/f1

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame including the label column.
    model : RandomForestClassifier
        Trained model.
    encoder :
        OneHotEncoder used during training.
    lb :
        LabelBinarizer used during training.
    categorical_features : iterable of str
        Categorical feature names to slice on.
    label : str
        Label column name.

    Returns
    -------
    results : dict
        Nested dict results[feature][value] = (precision, recall, f1)
    """
    from census.data.clean_data import process  # local import to avoid circulars and keep API clear

    results = {}
    for cat in categorical_features:
        results[cat] = {}
        for val in df[cat].dropna().unique():
            slice_df = df[df[cat] == val].copy()
            if slice_df.shape[0] == 0:
                continue
            X_slice, y_slice, _, _ = process(
                slice_df,
                categorical_features=list(categorical_features),
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            if X_slice.shape[0] == 0:
                continue
            preds = inference(model, X_slice)
            metrics = compute_model_metrics(y_slice, preds)
            results[cat][str(val)] = metrics
    return results
