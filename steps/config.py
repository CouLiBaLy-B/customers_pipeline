from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """
    Configuration parameters for the model name.
    """
    model_name: str = "spacy_model"
