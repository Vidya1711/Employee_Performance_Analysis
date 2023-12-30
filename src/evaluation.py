import logging
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from abc import ABC, abstractmethod

class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating a specific metric
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class PrecisionMetric(Evaluation):
    
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculate Precision")
            precision = precision_score(y_true, y_pred,average='weighted')
            logging.info(f"Precision: {precision}")
            return precision
        except Exception as e:
            logging.error(f"Error in calculating Precision: {e}")
            raise e

class RecallMetric(Evaluation):
    
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculate Recall")
            recall = recall_score(y_true, y_pred, average='weighted')
            logging.info(f"Recall: {recall}")
            return recall
        except Exception as e:
            logging.error(f"Error in calculating Recall: {e}")
            raise e

class F1ScoreMetric(Evaluation):
    
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculate F1 Score")
            # Use a different variable name for the F1 score calculation
            f1_score_value = f1_score(y_true, y_pred, average='weighted')
            logging.info(f"F1 Score: {f1_score_value}")
            return f1_score_value
        except Exception as e:
            logging.error(f"Error in calculating F1 Score: {e}")
            raise e

class AccuracyMetric(Evaluation):
    
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculate Accuracy")
            accuracy = accuracy_score(y_true, y_pred )
            logging.info(f"Accuracy: {accuracy}")
            return accuracy
        except Exception as e:
            logging.error(f"Error in calculating Accuracy: {e}")
            raise e
