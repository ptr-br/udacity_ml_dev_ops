from flask import Flask, session, jsonify, request  
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])

####################function for deployment
def store_model_into_pickle(model=None):
    os.makedirs(prod_deployment_path, exist_ok=True)

    # Define source files
    trained_model_src = os.path.join(model_path, 'trainedmodel.pkl')
    latest_score_src = os.path.join(model_path, 'latestscore.txt')
    ingested_files_src = os.path.join(dataset_csv_path, 'ingestedfiles.txt')

    # Check existence
    missing = [p for p in [trained_model_src, latest_score_src, ingested_files_src] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required file(s) for deployment: {missing}")

    # Copy to deployment directory
    for src in [trained_model_src, latest_score_src, ingested_files_src]:
        dst = os.path.join(prod_deployment_path, os.path.basename(src))
        shutil.copy2(src, dst)

    print(f"Deployment complete. Files copied to {prod_deployment_path}")

if __name__ == '__main__':
    store_model_into_pickle()