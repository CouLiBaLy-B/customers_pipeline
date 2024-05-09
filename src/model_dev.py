import logging
from abc import ABC, abstractmethod
import pandas as pd

import os

from typing import Set, List, Tuple
from spacy.tokens import DocBin
import spacy
from spacy.cli.train import train as spacy_train


class Model(ABC):
    def __init__(self,
                 nlp: spacy.language.Language = spacy.load('fr_core_news_sm')):
        self.nlp = nlp

    @abstractmethod
    def make_docs(self,
                  data: List[Tuple[str, str]],
                  target_file: str,
                  cats: Set[str]):
        pass

    @abstractmethod
    def fit(self, x_train, x_test, y_train, y_test, config_path: str):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass


class SpacyModel(Model):
    """
    Spacy Model for classification
    Args:
        Model (_type_): _description_
    """
    def make_docs(self,
                  data: List[Tuple[str, str]],
                  target_file: str,
                  cats: Set[str]
                  ):
        docs = DocBin()

        for doc, label in self.nlp.pipe(data, as_tuples=True):
            for cat in cats:
                doc.cats[cat] = 1 if cat == label else 0
            docs.add(doc)
        docs.to_disk(target_file)
        return docs

    def fit(self,
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_train,
            y_test,
            config_path="models/config.cfg"):

        cats = list(set(y_train))
        self.make_docs(data=list(zip(x_train, y_train)),
                       target_file='models/train.spacy',
                       cats=cats)
        self.make_docs(data=list(zip(x_test, y_test)),
                       target_file='models/valid.spacy',
                       cats=cats)
        try:
            spacy_train(config_path,
                        output_path="models/",
                        overrides={
                            "paths.train": 'models/train.spacy',
                            "paths.dev": "models/valid.spacy",
                            }
                        )
            logging.info("Spacy model training completed.")
            return spacy.load("models/spacy_textcat1/model-best")
        except Exception as e:
            logging.error(f"Error training Spacy model: {e}")
            raise e

    def predict_single(self, x):
        try:
            model_path = "models/spacy_textcat1/model-best"
            if os.path.exists(model_path):
                trained_model = spacy.load(model_path)
                doc = trained_model(x)
                return max(doc.cats, key=doc.cats.get)
            else:
                raise FileNotFoundError(f"""Model doesn't exist in this
                                        directory '{model_path}'""")
        except Exception as e:
            print(f"Error in model loading: {e}")
            return None

    def predict(self, X):
        pred = [self.predict_single(x) for x in X]
        return pred
