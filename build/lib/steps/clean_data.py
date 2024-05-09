import pandas as pd
# import numpy as np
import texthero as hero
import logging
from zenml import step


@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input data frame by handling missing values and performing any
    necessary transformations.

    Args:
        df (pd.DataFrame): The input data frame to be cleaned.
        Returns:
        pd.DataFrame: The cleaned data frame.
    """
    try:
        # Handle missing values
        df = df.dropna()

        # Perform any necessary data transformations
        df['ITEM_DESC'] = df['ITEM_DESC'].pipe(hero.clean)

        return df
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise e
