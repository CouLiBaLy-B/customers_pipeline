import pandas as pd
from src.evaluation import R2, RMSE
import logging
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
import spacy


@step
def evaluator(
    model: spacy.language.Language,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "RMSE"],
    Annotated[float, "R2"]
]:
    """
    Evaluate the performance of a machine learning model on the input
    data frame.

    Args:
        X_test (pd.DataFrame): The test input features.
        y_test (pd.DataFrame): The test target variables.
    Returns:
        float: The evaluation metric (e.g., accuracy, F1-score) for the model.
    """
    try:
        pred = model.predict(X_test)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, pred)

        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, pred)
        return rmse, r2
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise e
