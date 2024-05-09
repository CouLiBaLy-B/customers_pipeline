from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_df
from steps.evaluation import evaluator
from steps.model_train import train_model


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    """
    Define a machine learning pipeline for training a model.

    Args:
        data_path (str): The path to the input data file.
        target_column (str): The name of the column containing the target
                             variable.
        """
    df = ingest_data(data_path)

    X_train, X_test, y_train, y_test = clean_df(df)

    model = train_model(X_train, X_test, y_train, y_test)
    rmse, r2 = evaluator(model, X_test, y_test)
