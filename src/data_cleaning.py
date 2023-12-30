import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
#here, abc refers to the module and ABC refers to the Abstract Base Classes
from typing import Union
# Import OneHotEncoder from sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    
    """This class is used to preprocess the given dataset"""
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try: 
            enc = LabelEncoder()
            for i in (2, 3, 4, 5, 6, 7, 16, 26):
                data.iloc[:, i] = enc.fit_transform(data.iloc[:, i])
            data.drop(['EmpNumber'], axis=1, inplace=True)
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing the data: {e}")
            raise e



class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            # Check if 'Attrition' is present in the data
              
            X = data.drop('PerformanceRating',axis=1)
            Y = data['PerformanceRating']
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
            numerical_col = X_test.select_dtypes(include=['float64', 'int64']).columns
            X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            X_test[numerical_col] = scaler.fit_transform(X_test[numerical_cols])
            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            logging.error(f"Error in data handling: {str(e)}")
            raise e

class DataCleaning:
            def __init__(self,data:pd.DataFrame,strategy:DataStrategy)->None:
                self.data=data
                self.strategy=strategy
            def handle_data(self)->Union[pd.DataFrame,pd.Series]:
                try:
                    return self.strategy.handle_data(self.data)
                except Exception as e:
                    logging.error(f"There is a error in dataHandling{e}")
                    raise e
                        
                 




###I need to do this 