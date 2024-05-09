import logging
from abc import ABC, abstractmethod
import texthero as hero
import pandas as pd
# import numpy as np
from typing import Union
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract base class for data cleaning operations.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class DataPreProcessingStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values and perform any necessary data transformations.

        Args:
            data (pd.DataFrame): The input data frame to be handled.
        Returns: clean data
        """
        try:
            data = data.drop([
                "COUNTRY_KEY",
                "BARCODE",
                "BEM_CLASS_DESC_FR"
            ], axis=1)
            data = data.drop_duplicates()
            data['ITEM_DESC'] = data['ITEM_DESC'].pipe(hero.clean)
            data = data.dropna(subset=['ITEM_DESC', 'Regroupement_de_Class'])
            return data
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise e


class DataDivideStrategy(DataStrategy):

    def handle_data(self,
                    data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide the data into training and validation sets.

        Args:
            data (pd.DataFrame): The input data frame to be divided.

        Returns:
            pd.DataFrame, pd.DataFrame: The training and validation data
            frames.
        """
        try:
            X = data.drop(['Regroupement_de_Class'], axis=1)
            y = data['Regroupement_de_Class']

            X_train, X_val, y_train, y_val = train_test_split(X,
                                                              y,
                                                              test_size=0.2,
                                                              random_state=42)
            return X_train, X_val, y_train, y_val
        except Exception as e:
            logging.error(f"Error dividing data: {e}")
            raise e


class DataCleaning:
    def __init__(self, data: pd.DataFrame, data_strategy: DataStrategy):
        self.data_strategy = data_strategy
        self.data = data

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Clean the input data or divide it into training and validation sets.

        Args:
            data (pd.DataFrame): The input data frame to be cleaned or divided.
        Returns:
            pd.DataFrame, pd.DataFrame, pd.Series, pd.Series:
            The training and validation data frames and labels.
        """
        try:
            return self.data_strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error handling data: {e}")
            raise e
