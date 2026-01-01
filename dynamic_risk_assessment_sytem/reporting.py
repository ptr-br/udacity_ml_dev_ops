# reporting.py
import pickle  # unused but kept to match blueprint
from sklearn.model_selection import train_test_split  # unused, kept
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Reuse your diagnostics for predictions
import diagnostics

############### Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

############## Function for reporting
def score_model():
    # Calculate a confusion matrix using the test data and the deployed model
    test_file = os.path.join(test_data_path, 'testdata.csv')
    df_test = pd.read_csv(test_file)
    y_true = df_test['exited'].astype(int).values
    # Get predictions from deployed model
    preds = diagnostics.model_predictions(df_test)
    y_pred = np.array(preds, dtype=int)

    cm = metrics.confusion_matrix(y_true, y_pred)
    os.makedirs(output_model_path, exist_ok=True)
    fig_path = os.path.join(output_model_path, 'confusionmatrix.png')

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Exited', 'Exited'],
                yticklabels=['Not Exited', 'Exited'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix – Deployed Model')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print(f"Confusion matrix saved to {fig_path}")
    return cm.tolist()

if __name__ == '__main__':
    score_model()