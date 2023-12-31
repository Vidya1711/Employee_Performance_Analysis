import mlflow
import pandas as pd
import logging
from zenml import step
from src.model_dev import MLPClassifierModel 
from .config import ModelNameConfig
from sklearn.base import ClassifierMixin 
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name,enable_cache=False)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.Series,
    Y_test: pd.Series,
    config: ModelNameConfig,
) -> ClassifierMixin:
    try:
        model = None
        if config.model_name == "MLPClassifier":
            mlflow.sklearn.autolog()
            model = MLPClassifierModel()
            trained_model = model.train(X_train, Y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} is not supported")
    except Exception as e:
        logging.error(f"Error in training the model: {e}")
        raise e