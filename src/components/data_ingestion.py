import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.path_file import raw_data_path

import pandas as pd

class DataIngestion:
    """
    This class shall  be used to read data

    """
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        """
        Method Name: initiate_data_ingestion
        Description: This method read csv data and save the data 
        Output: A Dataframe of raw data
        On Failure: Raise Exception

        """


        logging.info('entering the data ingestion method of DataIngestion class')

        try:
            df = pd.read_csv("notebook/data/Chennai housing sale.csv")
            logging.info('Read the data from file in to dataframe')
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

            df.to_csv(raw_data_path, index=False, header=True)

            logging.info("ingestion of data is completed. Exiting initiate_data_ingestion method of DataIngestion class")

            return df
        
        except Exception as e:
            logging.info("ingestion of data is not successful,Exiting initiate_data_ingestion method of DataIngestion class : " + str(e))
            raise CustomException(e, sys)
        


