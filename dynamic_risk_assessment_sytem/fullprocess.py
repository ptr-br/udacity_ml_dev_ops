# fullprocess.py

import os
import json
import subprocess
import training
import scoring
import deployment
import ast
import reporting



# Load config
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path       = os.path.join(config['input_folder_path'])
output_folder_path      = os.path.join(config['output_folder_path'])
output_model_path       = os.path.join(config['output_model_path'])
prod_deployment_path    = os.path.join(config['prod_deployment_path'])
test_data_path          = os.path.join(config['test_data_path'])

def list_csv(folder):
    return sorted([f for f in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.csv')])

def run():
    ################## Check and read new data
    # first, read ingestedfiles.txt (from deployment dir)
    ingested_record = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    already_ingested = []
    if os.path.exists(ingested_record):
        content = open(ingested_record, 'r').read().strip()
        if content:
            try:
                already_ingested = ast.literal_eval(content)  # safely parse ["...","..."] or ['...','...']
            except Exception:
                # fallback: treat as empty if parsing fails
                already_ingested = []

    # second, determine whether source data has files not listed in ingestedfiles.txt
    source_files = list_csv(input_folder_path)
    new_files = [f for f in source_files if f not in already_ingested]

    ################## Deciding whether to proceed, part 1
    if not new_files:
        print("No new data found. Ending process.")
        raise SystemExit(0)

    print(f"New data detected: {new_files}. Running ingestion...")
    # Run ingestion.py to ingest all data (updates finaldata.csv and ingestedfiles.txt)
    ret = subprocess.run(['python', 'ingestion.py'], capture_output=True, text=True)
    if ret.returncode != 0:
        raise RuntimeError(f"Ingestion failed:\n{ret.stderr}")
    print("Ingestion complete.")

    ################## Checking for model drift
    # 1) read score from deployed model (latestscore.txt in production_deployment)
    deployed_score_path = os.path.join(prod_deployment_path, 'latestscore.txt')
    if not os.path.exists(deployed_score_path):
        print("No deployed latestscore.txt found; treating as drift.")
        old_score = None
    else:
        with open(deployed_score_path, 'r') as f:
            try:
                old_score = float(f.read().strip())
            except Exception:
                old_score = None

    # 2/3) compute a new score with current code (uses test data); raw comparison
    print("Scoring current model on test data...")
    new_score = scoring.score_model()

    print(f"Old (deployed) score: {old_score}, New score: {new_score}")
    drift_detected = (old_score is not None) and (new_score < old_score)
    if old_score is None:
        drift_detected = True  # if we cannot read old score, force retrain/deploy

    ################## Deciding whether to proceed, part 2
    if not drift_detected:
        print("No model drift detected. Ending process.")
        raise SystemExit(0)

    ################## Re-training
    print("Model drift detected. Re-training on latest data...")
    training.train_model()

    ################## Re-deployment
    print("Re-deploying the newly trained model and artifacts...")
    deployment.store_model_into_pickle()

    ################## Diagnostics and reporting
    print("Running reporting (confusion matrix) on deployed model...")
    reporting.score_model()

    print("Running API calls and saving combined outputs...")
    # Assumes app.py is running separately on port 8000; call apicalls.py to capture outputs
    api_run = subprocess.run(['python', 'apicalls.py'], capture_output=True, text=True)
    if api_run.returncode != 0:
        # Not fatal—just warn if API wasn’t reachable
        print(f"Warning: apicalls.py failed (is app.py running?):\n{api_run.stderr}")
    else:
        print(api_run.stdout)

    print("Full process complete.")

if __name__ =="__main__":
    run()