import logging
from abc import ABC, abstractmethod
from sklearn.neural_network import MLPClassifier
import pandas as pd

class Model(ABC):
    @abstractmethod
    def train(self, X_train:pd.DataFrame, Y_train:pd.Series):
        pass    

class MLPClassifierModel(Model):
    def train(self, X_train:pd.DataFrame, Y_train:pd.Series, **kwargs):
        try:
            model_mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),batch_size=10,learning_rate_init=0.01,max_iter=2000,random_state=10)
            model_mlp.fit(X_train,Y_train)
            logging.info("Model training completed")
            return model_mlp
        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            raise e