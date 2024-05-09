from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    data_path = "data/data.csv"
    target_column = "label"

    train_pipeline(data_path)
    # # Evaluate the trained model
    # model_performance = evaluator(model,  df, target_column)
