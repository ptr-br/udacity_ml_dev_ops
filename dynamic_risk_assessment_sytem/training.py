import os, json, pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

#################Function for training the model
# training.py (replace train_model body with this)

def train_model():
    os.makedirs(model_path, exist_ok=True)
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    feature_cols = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
    X = df[feature_cols]
    y = df['exited'].astype(int)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=0))
    ])

    best_c, best_score = None, -1
    for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
        pipe.set_params(lr__C=C)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        # Use F1 to match your scoring metric
        score = cross_val_score(pipe, X, y, cv=cv, scoring='f1').mean()
        if score > best_score:
            best_score, best_c = score, C

    pipe.set_params(lr__C=best_c)
    pipe.fit(X, y)

    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as f:
        pickle.dump(pipe, f)

    # Save metadata (best C) so we know what was chosen
    with open(os.path.join(model_path, 'model_metadata.json'), 'w') as f:
        json.dump({"feature_cols": feature_cols, "target_col": "exited", "best_C": best_c}, f)

    print(f"Model trained. Best C={best_c:.2f}, CV F1={best_score:.3f}")

if __name__ == '__main__':
    train_model()