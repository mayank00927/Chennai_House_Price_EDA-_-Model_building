
import sys
import os
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.path_file import preprocessor_obj_file_path
from src.utils import save_object


# from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

class DataTransformation:

    def __init__(self):
        pass

    def get_data_transformer_obj(self):

        """
        Method Name: get_data_transformer
        Description: This method checks consists of transformers and pipelines.
        Output: returns the preprocessor file to work on dataframe
        On Failure: Raise Exception

        """

        try:
            numerical_columns = ['INT_SQFT', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM',
                                 'PRICE_PER_SQ_FT', 'PROP_AGE',
                                ]
            categorical_columns= ['AREA', 'SALE_COND', 'PARK_FACIL', 'BUILDTYPE', 'STREET', 'UTILITY']

            num_pipeline =Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("scaler",StandardScaler(with_mean=False))
            ]

            )

            cat_pipeline = Pipeline(
                steps =[("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))

                ]
            )

            logging.info(f"categorical_columns : {categorical_columns}")
            logging.info(f"numerical columns : {numerical_columns}")

            logging.info('categorical and numerical pipeline completed')

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)

                ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path, test_path):
        """
        Method Name: initiate_data_transformation
        Description: This method applies preprocessor object on train and test data
                     and returns train and test array.
        Output: returns the preprocessor file to work on dataframe
        On Failure: Raise Exception

        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print(train_df.columns)
         
            logging.info("entered in initiate_data_transformation method of DataTransformation class")
            logging.info("reading train and test data is completed")

            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = 'TOTAL_SALES_PRICE'

            # numerical_columns = ["writing score", "reading score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr =preprocessing_obj.transform(input_feature_test_df)

            train_arr= np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            
            
   
            logging.info(f"saved preprocessing object.")



            save_object(

                file_path = preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info(f"initiate_data transformation is successful of DataTransformation class, exiting initiate_data_transformation class")
            return (
                    train_arr,
                    test_arr,
                    preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("initiate_data transformation method is failed, exiting initiate_data_transformation method of DataTaransformation class : " + str(e))
            raise CustomException(e,sys)



