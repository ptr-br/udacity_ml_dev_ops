# app.py
from flask import Flask, jsonify, request
import pandas as pd
import json
import os

# Our project modules
import diagnostics
import scoring

###################### Set up variables
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])


####################### Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    # Expect JSON: {"filepath": "testdata/testdata.csv"}
    fp = None
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        fp = body.get("filepath")
    if not fp:
        # Default to test data if not provided
        fp = os.path.join(test_data_path, "testdata.csv")
    df = pd.read_csv(fp)
    preds = diagnostics.model_predictions(df)
    return jsonify({"predictions": preds}), 200


####################### Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring_endpoint():
    f1 = scoring.score_model()
    return jsonify({"f1_score": f1}), 200


####################### Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summarystats():
    stats = diagnostics.dataframe_summary()
    return jsonify({"summary_statistics": stats}), 200


####################### Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diags():
    times = diagnostics.execution_time()
    na_frac = diagnostics.missing_data()
    deps = diagnostics.outdated_packages_list()
    return jsonify(
        {
            "execution_time_sec": {"ingestion": times[0], "training": times[1]},
            "missing_data_fraction_per_column": na_frac,
            "outdated_packages": deps,
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
