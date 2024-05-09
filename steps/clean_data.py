import pandas as pd
# import numpy as np
import logging
from zenml import step
from src.data_cleaning import (DataCleaning,
                               DataPreProcessingStrategy,
                               DataDivideStrategy)
from typing import Tuple
from typing_extensions import Annotated


@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Clean the input data frame by handling missing values and performing any
    necessary transformations.

    Args:
        df (pd.DataFrame): The input data frame to be cleaned and splitted.
        Returns:
        pd.DataFrame: X_train, X_test, y_train, y_test.
    """
    try:
        process_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        divide_strategy = DataDivideStrategy()
        data_splitting = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_splitting.handle_data()
        logging.info("Data cleaning and splitting completed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise e
