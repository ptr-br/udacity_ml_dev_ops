# ingestion.py
import pandas as pd
import os
import json

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe():
    os.makedirs(output_folder_path, exist_ok=True)

    all_files = [f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))]
    csv_files = sorted([f for f in all_files if f.lower().endswith(".csv")])

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_folder_path}")

    # Read, concat, and de-duplicate
    frames = []
    for fname in csv_files:
        path = os.path.join(input_folder_path, fname)
        df = pd.read_csv(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    # Write outputs
    final_path = os.path.join(output_folder_path, "finaldata.csv")
    combined.to_csv(final_path, index=False)

    record_path = os.path.join(output_folder_path, "ingestedfiles.txt")
    with open(record_path, "w") as f:
        f.write(str(csv_files)) 

    print(f"Ingestion complete. Rows: {len(combined)}. Saved: {final_path}. Files: {csv_files}")

if __name__ == '__main__':
    merge_multiple_dataframe()