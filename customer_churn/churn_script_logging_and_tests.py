"""
Test suite for churn_library module.

This module contains unit tests for the customer churn prediction functions.
"""

import logging

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
    Test data import - this example is completed for you to assist with the other test functions.
    """
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    """
    Test perform eda function.
    """
    # Implementation placeholder
    _ = perform_eda


def test_encoder_helper(encoder_helper):
    """
    Test encoder helper.
    """
    # Implementation placeholder
    _ = encoder_helper


def test_perform_feature_engineering(perform_feature_engineering):
    """
    Test perform_feature_engineering.
    """
    # Implementation placeholder
    _ = perform_feature_engineering


def test_train_models(train_models):
    """
    Test train_models.
    """
    # Implementation placeholder
    _ = train_models


if __name__ == "__main__":
    pass
