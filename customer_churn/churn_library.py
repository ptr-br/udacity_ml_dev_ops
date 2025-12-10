"""
Module for customer churn prediction and analysis.

This module contains functions for performing exploratory data analysis,
feature engineering, model training, and evaluation for customer churn prediction.

Author: ptr-br
Date: December 10, 2025
"""

# Import libraries
import os
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    Returns dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    Perform eda on df and save figures to images folder.

    input:
            df: pandas dataframe

    output:
            None
    '''
    # Create Churn column
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Create images/eda directory if it doesn't exist
    os.makedirs('./images/eda', exist_ok=True)

    # Churn distribution
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Frequency')
    plt.savefig('./images/eda/churn_distribution.png')
    plt.close()

    # Customer age distribution
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('./images/eda/customer_age_distribution.png')
    plt.close()

    # Marital status distribution
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Marital Status Distribution')
    plt.xlabel('Marital Status')
    plt.ylabel('Proportion')
    plt.savefig('./images/eda/marital_status_distribution.png')
    plt.close()

    # Total transaction distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Total Transaction Count Distribution')
    plt.xlabel('Total Trans Ct')
    plt.ylabel('Density')
    plt.savefig('./images/eda/total_transaction_distribution.png')
    plt.close()

    # Correlation heatmap (numeric columns only)
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=False,
                cmap='Dark2_r', linewidths=2)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('./images/eda/heatmap.png')
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook.

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
                     be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for encoded categorical features
    '''
    for category in category_lst:
        category_list = []
        category_groups = df.groupby(category).mean(
            numeric_only=True)[response]

        for val in df[category]:
            category_list.append(category_groups.loc[val])

        df[category + '_' + response] = category_list

    return df


def perform_feature_engineering(df, response):
    '''
    Perform feature engineering on the dataframe.

    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
                       be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Create Churn column if not exists
    if 'Churn' not in df.columns:
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

    # Categorical columns to encode
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # Encode categorical columns
    df = encoder_helper(df, cat_columns, response)

    # Target variable
    y = df['Churn']

    # Feature columns
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    # pylint: disable=invalid-name
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


# pylint: disable=too-many-arguments,too-many-positional-arguments
def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Produces classification report for training and testing results and stores report as image
    in images folder.

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Create images/results directory if it doesn't exist
    os.makedirs('./images/results', exist_ok=True)

    # Random Forest results
    plt.figure(figsize=(6, 8))
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./images/results/rf_results.png', bbox_inches='tight')
    plt.close()

    # Logistic Regression results
    plt.figure(figsize=(6, 8))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./images/results/logistic_results.png', bbox_inches='tight')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):  # pylint: disable=invalid-name
    '''
    Creates and stores the feature importances in pth.

    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):  # pylint: disable=invalid-name
    '''
    Train, store model results: images + scores, and store models.

    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Create models directory if it doesn't exist
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./images/results', exist_ok=True)

    # Grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Generate predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Generate classification reports
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Generate ROC curves
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    RocCurveDisplay.from_estimator(lrc, X_test, y_test, ax=ax, alpha=0.8)
    plt.title('ROC Curves')
    plt.savefig('./images/results/roc_curve_result.png')
    plt.close()

    # Generate feature importance plot
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        './images/results/feature_importances.png')


if __name__ == "__main__":
    # Example usage
    # Import data
    DF = import_data("./data/bank_data.csv")

    # Perform EDA
    perform_eda(DF)

    # Feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DF, response='Churn')

    # Train models
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
