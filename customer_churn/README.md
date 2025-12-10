# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This project implements a machine learning pipeline to identify credit card customers that are most likely to churn. The solution follows clean code principles and best practices for production-ready ML systems, including:

- **Exploratory Data Analysis (EDA)**: Automated generation of visualizations to understand data distributions and correlations
- **Feature Engineering**: Encoding categorical variables and selecting relevant features
- **Model Training**: Training both Random Forest and Logistic Regression classifiers with hyperparameter tuning
- **Model Evaluation**: Comprehensive evaluation with classification reports, ROC curves, and feature importance analysis
- **Testing & Logging**: Full test suite with detailed logging for monitoring and debugging
- **Code Quality**: PEP8 compliant code with automated linting

The pipeline processes customer data, trains models, and saves all results (models, visualizations, and reports) to the appropriate directories for easy access and deployment.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Setup with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh


# Install all dependencies (creates .venv automatically)
uv sync

# Run the main pipeline
uv run python churn_library.py

# Run tests
uv run python churn_script_logging_and_tests.py

# Check code quality
uv run pylint churn_library.py
```

## Running the Project

### 1. Run the Main Pipeline

Execute the complete ML pipeline:

```bash
uv run python churn_library.py
```

**This will:**
- Load the dataset from `./data/bank_data.csv`
- Perform exploratory data analysis and save visualizations to `./images/eda/`
- Engineer features and prepare training/test datasets
- Train Random Forest and Logistic Regression models with hyperparameter tuning
- Generate and save model evaluation reports to `./images/results/`
- Save trained models to `./models/` directory

**Expected outputs:**
- `./images/eda/`: EDA visualizations (churn distribution, age distribution, heatmap, etc.)
- `./images/results/`: Model performance reports (ROC curves, feature importance, classification reports)
- `./models/`: Trained models (`rfc_model.pkl`, `logistic_model.pkl`, `scaler.pkl`)

### 2. Run Tests

Execute the test suite to validate all functions:

```bash
uv run python churn_script_logging_and_tests.py
```

**Test results:**
- All test outputs logged to `./logs/churn_library.log`
- SUCCESS/ERROR status for each function
- Detailed assertions and validations

### 3. Code Quality Checks

Run linting to ensure code quality:

```bash
# Run pylint on main library
uv run pylint churn_library.py

# Run pylint on test suite
uv run pylint churn_script_logging_and_tests.py

# Format code with autopep8 (if needed)
uv run autopep8 --in-place --aggressive --aggressive churn_library.py
```

## Project Structure

```
customer_churn/
├── churn_library.py                    # Main ML pipeline implementation
├── churn_script_logging_and_tests.py  # Test suite with logging
├── churn_notebook.ipynb                # Jupyter notebook for exploration
├── Guide.ipynb                         # Project guide
├── README.md                           # This file
├── pyproject.toml                      # Project dependencies (uv/pip)
├── requirements_py3.*.txt              # Alternative requirements files
├── data/
│   └── bank_data.csv                   # Input dataset
├── images/
│   ├── eda/                            # EDA visualizations (generated)
│   └── results/                        # Model evaluation results (generated)
├── logs/
│   └── churn_library.log               # Test execution logs (generated)
└── models/
    ├── rfc_model.pkl                   # Trained Random Forest model (generated)
    ├── logistic_model.pkl              # Trained Logistic Regression model (generated)
    └── scaler.pkl                      # Feature scaler (generated)
```

## Dependencies

All dependencies are managed via `pyproject.toml` and can be installed with `uv sync`.

## Features

### Data Processing
- Automated feature engineering with categorical encoding
- Standard scaling for logistic regression to ensure convergence
- Train/test split with stratification

### Models
- **Random Forest Classifier**: With grid search for hyperparameter optimization
- **Logistic Regression**: With scaled features and optimized solver settings

### Evaluation Metrics
- Classification reports (precision, recall, F1-score)
- ROC curves with AUC scores
- Feature importance analysis for Random Forest

## Author

ptr-br  
Date: December 10, 2025



