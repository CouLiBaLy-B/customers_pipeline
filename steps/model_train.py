from zenml import step
import pandas as pd
from src.model_dev import SpacyModel
import spacy
import logging
from .config import ModelNameConfig


@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
) -> spacy.language.Language:
    """
    Train a machine learning model using the input data frame.

    Args:
        df (pd.DataFrame): The cleaned and preprocessed data frame.
        target_column (str): The name of the column containing the target
                            variable.

    Returns:
        object: The trained machine learning model.
    """
    model = None
    if config.model_name == 'spacy_model':
        try:
            model = SpacyModel()
            trained_model = model.fit(X_train, X_test, y_train, y_test)
            return trained_model
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise e
    else:
        logging.error(f"Unsupported model name: {config.model_name}")
        raise ValueError(f"Unsupported model name: {config.model_name}")
