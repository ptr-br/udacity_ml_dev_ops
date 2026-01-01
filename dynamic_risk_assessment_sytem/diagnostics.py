# diagnostics.py

import pandas as pd
import numpy as np
import time
import subprocess
import os
import json
import pickle

################## Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


################## Function to get model predictions
def model_predictions(df: pd.DataFrame = None):
    """
    Read the deployed model and a dataset, calculate predictions.
    If df is None, loads testdata/testdata.csv from test_data_path.
    Returns a list of predictions (ints for classification).
    """
    model_file = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Deployed model not found at {model_file}")

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    if df is None:
        test_file = os.path.join(test_data_path, "testdata.csv")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test data not found at {test_file}")
        df = pd.read_csv(test_file)

    feature_cols = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X = df[feature_cols].copy()

    preds = model.predict(X)
    return list(map(int, preds))


################## Function to get summary statistics
def dataframe_summary():
    """
    Return a flattened list: for each numeric column -> [mean, median, mode].
    If multiple modes, use the smallest; if no mode (all unique), return NaN for mode.
    """
    final_file = os.path.join(dataset_csv_path, 'finaldata.csv')
    if not os.path.exists(final_file):
        raise FileNotFoundError(f"Final dataset not found at {final_file}")
    df = pd.read_csv(final_file)
    num_df = df.select_dtypes(include=[np.number])
    stats = []
    for col in num_df.columns:
        series = num_df[col].dropna()
        mean = float(series.mean())
        median = float(series.median())
        mode_vals = series.mode()
        mode = float(mode_vals.iloc[0]) if not mode_vals.empty else float('nan')
        stats.extend([mean, median, mode])
    return stats


################## Function to check missing data
def missing_data():
    """
    Count NA values by column for finaldata.csv and return the percent NA per column
    as a list aligned to the dataset’s column order.
    """
    final_file = os.path.join(dataset_csv_path, "finaldata.csv")
    if not os.path.exists(final_file):
        raise FileNotFoundError(f"Final dataset not found at {final_file}")

    df = pd.read_csv(final_file)
    na_percent = df.isna().mean().tolist()  # fraction per column
    return na_percent


################## Function to get timings
def execution_time():
    """
    Calculate timing (in seconds) of ingestion.py and training.py.
    Returns [ingestion_time_sec, training_time_sec].
    """
    times = []

    for script in ["ingestion.py", "training.py"]:
        if not os.path.exists(script):
            raise FileNotFoundError(f"{script} not found in current directory.")
        start = time.perf_counter()
        result = subprocess.run(["python", script], capture_output=True, text=True)
        end = time.perf_counter()
        # Optional: surface errors if script fails
        if result.returncode != 0:
            raise RuntimeError(f"{script} failed:\n{result.stderr}")
        times.append(round(end - start, 4))

    return times


################## Function to check dependencies
def outdated_packages_list():
    """
    Use uv to list outdated packages and return a list of dicts:
    [{'name':..., 'current':..., 'latest':...}, ...]
    """
    proc = subprocess.run(
        ["uv", "pip", "list", "--outdated", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    return [
        {"name": d["name"], "current": d["version"], "latest": d["latest_version"]}
        for d in data
    ]


if __name__ == "__main__":
    # Run all diagnostics when executed directly
    preds = model_predictions()  # uses testdata by default
    print(f"Sample predictions (first 10): {preds[:10] if preds else preds}")

    stats = dataframe_summary()
    print(f"Summary stats (flattened): {stats}")

    na_pct = missing_data()
    print(f"Missing data fraction per column: {na_pct}")

    times = execution_time()
    print(f"Execution times [ingestion, training] (s): {times}")

    deps = outdated_packages_list()
    print(f"Outdated packages (name/current/latest): {deps}")
