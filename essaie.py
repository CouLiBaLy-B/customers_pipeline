import pandas as pd
from src.data_cleaning import (DataCleaning,
                               DataPreProcessingStrategy,
                               DataDivideStrategy
                               )

df = pd.read_csv('data/data.csv')


process_strategy = DataPreProcessingStrategy()
data_cleaning = DataCleaning(df, process_strategy)
processed_data = data_cleaning.handle_data()

divide_strategy = DataDivideStrategy()
data_splitting = DataCleaning(processed_data, divide_strategy)
X_train, X_test, y_train, y_test = data_splitting.handle_data()
print(y_train.unique().tolist())
