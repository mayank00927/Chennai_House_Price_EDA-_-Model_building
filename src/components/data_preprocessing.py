
# importing general libraries
import os
import sys
import pandas as pd
import numpy as np

# importing needed libraries
from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest,f_classif

# importing custom packages
from src.logger import logging
from src.exception import CustomException
from src.path_file import (train_data_path,
                           test_data_path,
                           missing_column_path)

class Preprocessor:
    """
    This class shall  be used to clean the data before training

    """
    def __init__(self):
        pass

    def feature_extraction(self, data):
            """
            Method Name: feature_extraction
            Description: This method creates new features from the given data.
            Output: A Dataframe which has new features
            On Failure: Raise Exception

            """
            logging.info(f"entered in feature extraction method of Preprocessor class")

            try:

                data['TOTAL_SALES_PRICE'] = data['SALES_PRICE'] + data['COMMIS'] + data['REG_FEE']
                data['BUILD_YEAR'] = data['DATE_BUILD'].apply(lambda x:x.split('-')[2])
                data['SALE_YEAR'] = data['DATE_SALE'].apply(lambda x:x.split('-')[2])
                data[['BUILD_YEAR', 'SALE_YEAR']] = data[['BUILD_YEAR', 'SALE_YEAR']].apply(pd.to_numeric)
                data['PROP_AGE'] = (data['SALE_YEAR'] - data['BUILD_YEAR'])
                data['PRICE_PER_SQ_FT'] = data['TOTAL_SALES_PRICE']/data['INT_SQFT']
               
                # converting datatype of columns
                data[['PROP_AGE', 'PRICE_PER_SQ_FT','N_ROOM']] = data[['PROP_AGE', 'PRICE_PER_SQ_FT','N_ROOM']].astype(float,errors='ignore')
            
                # replacing mispelled values in columns with corect values
                
                data['AREA'] = data['AREA'].replace({'Karapakam':'Karapakkam','Ana Nagar':'Anna Nagar','Ann Nagar':'Anna Nagar'
                                                     ,'Adyr':'Adyar','Velchery':'Velachery','Chormpet':'Chrompet','Chrompt':'Chrompet'
                                                     ,'Chrmpet':'Chrompet','KKNagar':'KK Nagar','TNagar':'T Nagar'})
                data['SALE_COND'] = data['SALE_COND'].replace({'Ab Normal': 'AbNormal', 'Partiall': 'Partial','PartiaLl':'Partial','Adj Land':'AdjLand'})
                data['PARK_FACIL'] = data['PARK_FACIL'].replace('Noo','No')
                data['BUILDTYPE'] = data['BUILDTYPE'].replace({'Comercial':'Commercial','Other':'Others'})
                data['UTILITY'] = data['UTILITY_AVAIL'].replace({'All Pub':'AllPub','NoSewr ':'NoSewa','NoSeWa':'NoSewa'})
                data['STREET'] = data['STREET'].replace({'Pavd':'Paved','No Access':'NoAccess'})
                
                # dropping irrelevant column
                data.drop(columns=['PRT_ID','QS_ROOMS','QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL',
                                   'UTILITY_AVAIL','DATE_BUILD','DATE_SALE','REG_FEE', 'COMMIS', 'SALES_PRICE'
                                   ,'DIST_MAINROAD','MZZONE','BUILD_YEAR','SALE_YEAR'],axis=1,inplace = True)
                
                logging.info(f'feature extraction is successful. Exited the feature_extraction method of the Preprocessor class')
               
                print(data.columns)
                return data

            except Exception as e:
                logging.info('Exception occured in feature_extraction method of the Preprocessor class. Exception message: ' + str(
                                e))
                raise CustomException(e, sys)
            
    def outlier_removal(Self, data):
        """
        Method Name: outlier_removal
        Description: This method removes outliers from existing data.
        Output: A Dataframe without outlier.
        On Failure: Raise Exception

        """
        try:
            pass
        except:
            pass

    def is_null_present(self,data):
        """
        Method Name: is_null_present
        Description: This method checks whether there are null values present in the pandas Dataframe or not.
        Output: returns the list of columns for which null values are present.
        On Failure: Raise Exception

        """
        logging.info('Entered the is_null_present method of the Preprocessor class')
        null_present = False
        cols_with_missing_values=[]  #empty list
        cols = data.columns          #column names in dataset

        try:
            null_counts=data.isnull().sum() # check for the count of null values per column
            for i in range(len(null_counts)):
                if null_counts[i]>0:
                    null_present=True
                    cols_with_missing_values.append(cols[i])

            if(null_present): # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isnull().sum())

                os.makedirs(os.path.dirname(missing_column_path),exist_ok=True)  #making directory artifacts if not exist
                dataframe_with_null.to_csv(missing_column_path,index=False,header=True) # storing the null column information to csv file
            logging.info('Finding missing values is successful.Data written to the missing_data_column.csv file. Exited the is_null_present method of the Preprocessor class')
            return cols_with_missing_values

        except Exception as e:
            logging.info('Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            logging.info('Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise CustomException(e,sys)


    def split_data(self, data):
        """
        Method Name: split_data
        Description: This method split data in to training and test data
        Output: A pandas DataFrame.
        On Failure: Raise Exception

        """
        logging.info('Entered the split_data method of the Preprocessor class')
        try:
            train_set, test_set = train_test_split(data, test_size=0.33, random_state=42)
            os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
            train_set.to_csv(train_data_path, index=False,
                             header=True)  # storing the training data set to train_set csv file
            os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
            test_set.to_csv(test_data_path, index=False,
                            header=True)  # storing the testing data set to test_set csv file

            logging.info('Splitting of data is Successful.Exited the split_data method of the Preprocessor class')
            return train_data_path, test_data_path
        
        except Exception as e:
            logging.info('Exception occured in split_data method of the Preprocessor class. Exception message: ' + str(
                            e))
            raise CustomException(e, sys)

    