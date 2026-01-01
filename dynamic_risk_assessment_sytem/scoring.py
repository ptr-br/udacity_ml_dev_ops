from flask import Flask, session, jsonify, request  
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])
latest_score_path = os.path.join(model_path, 'latestscore.txt')

#################Function for model scoring
def score_model():
    # Load model
    model_file = os.path.join(model_path, 'trainedmodel.pkl')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    test_file = os.path.join(test_data_path, 'testdata.csv')  # adjust filename if different
    df_test = pd.read_csv(test_file)

    feature_cols = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    target_col = 'exited'

    X_test = df_test[feature_cols].copy()
    y_test = df_test[target_col].astype(int).copy()

    # F1 score
    y_pred = model.predict(X_test)
    f1 = metrics.f1_score(y_test, y_pred)

    # Write the result to file
    os.makedirs(model_path, exist_ok=True)
    with open(latest_score_path, 'w') as f:
        f.write(str(f1))

    print(f"F1 score: {f1:.6f} written to {latest_score_path}")
    return f1

if __name__ == '__main__':
    score_model()