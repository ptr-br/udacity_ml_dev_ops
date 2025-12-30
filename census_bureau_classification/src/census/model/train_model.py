import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from census.config import CAT_FEATURES, DATA_PATH, ENCODER_PATH, LB_PATH, MODEL_PATH
from census.data.process_data import process
from census.model.model import compute_model_metrics, inference, slice_performance, train_model


def main():
    data = pd.read_csv(DATA_PATH)
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    X_test, y_test, _, _ = process(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Test precision: {precision:.4f}, recall: {recall:.4f}, f1: {fbeta:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(lb, LB_PATH)

    slice_results = slice_performance(
        test, model, encoder, lb, categorical_features=CAT_FEATURES, label="salary"
    )
    for feat, mapping in slice_results.items():
        print(f"Slice results for {feat}:")
        for val, metrics in mapping.items():
            p, r, f = metrics
            print(f"  {val}: precision={p:.3f}, recall={r:.3f}, f1={f:.3f}")


if __name__ == "__main__":
    main()
