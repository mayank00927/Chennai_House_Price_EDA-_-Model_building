import os
import sys
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
from src.path_file import trained_model_file_path


class ModelTrainer:
    """
    This class shall  be used to train model
    """
    def __init__(self):
        pass

    def initiate_model_trainer(self,train_array,test_array):

        """
        Method Name: initiate_model_trainer
        Description: This method applies model training on ML algorithms mentioned
        Output: return r2_score(accuracy)
        On Failure: Raise Exception

        """
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                     'n_estimators': [8,16,32,64,128,256]
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[0.01, 0.05, 0.1,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[0.01, 0.05, 0.1,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [3,6,8,10,12,16,18],
                    'learning_rate': [0.01, 0.05, 0.1,.001],
                    'iterations': [10,30, 50,60,70,90,100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,0.05,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }


            logging.info(f"training and Tuning the model")
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset {best_model_name} with score {best_model_score}")

            save_object(
                file_path=trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            #print actual and predicted House_price- top 5
            print(f"actual price : {np.round(y_test[0:5],decimals=1)}")
            print(f"predicted price : {np.round(predicted[0:5],decimals=1)}")

            r2_square = r2_score(y_test, predicted)
            return r2_square
            
            
        except Exception as e:
            logging.info(f"Model training is failed : {e}")
            raise CustomException(e,sys)