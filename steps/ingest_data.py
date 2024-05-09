import pandas as pd
import logging
from zenml import step


class IngestData:
    """
    Ingest the data from the specified data path and return a pandas DataFrame.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("Ingesting data from path: %s", self.data_path)
        df = pd.read_csv(self.data_path)
        return df


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingest data from the specified data path.

    Args:
        data_path (str): The path to the data file.
        Returns:
        pd.DataFrame: The ingested data as a pandas DataFrame.
    """
    try:
        ingester = IngestData(data_path)
        df = ingester.run()
        return df
    except Exception as e:
        logging.error(f"Error ingesting data from {data_path}: {e}")
        raise e
