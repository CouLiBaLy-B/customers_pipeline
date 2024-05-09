from abc import ABC, abstractmethod
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error


class Evalution(ABC):
    """
    Abstract base class for model evaluation.
    """

    @abstractmethod
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        pass


class R2(Evalution):
    """
    Evaluate the model using Mean Squared Error (MSE).
    """

    def calculate_score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Evaluate the performance of a machine learning model using MSE.

        Args:
            y_true (pd.Series): The true target values.
            y_pred (pd.Series): The predicted target values.
        Returns:
            float: the MSE evaluation metric
        """
        try:
            logging.info("Calculating Mean Squared Error (MSE)...")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse:.4f}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            raise e


class RMSE(Evalution):
    """
    Evaluate the model using Root Mean Squared Error (RMSE).
    """

    def calculate_score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Evaluate the performance of a machine learning model using RMSE.

        Args:
            y_true (pd.Series): The true target values.
            y_pred (pd.Series): The predicted target values.
        Returns:
            float: the RMSE evaluation metric
        """
        try:
            logging.info("Calculating Root Mean Squared Error (RMSE)...")
            rmse = (mean_squared_error(y_true, y_pred))**0.5
            logging.info(f"RMSE: {rmse:.4f}")
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE: {e}")
            raise e
