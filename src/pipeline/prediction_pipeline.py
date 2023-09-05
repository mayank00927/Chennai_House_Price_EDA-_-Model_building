import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        AREA: str,
        INT_SQFT,
        N_BEDROOM,
        N_BATHROOM,
        N_ROOM,
        SALE_COND:str,
        PARK_FACIL:str,
        BUILDTYPE: str,
        STREET: str,
        PROP_AGE,
        PRICE_PER_SQ_FT,
        UTILITY:str):

        self.AREA = AREA

        self.INT_SQFT = INT_SQFT

        self.N_BEDROOM = N_BEDROOM

        self.N_BATHROOM = N_BATHROOM

        self.N_ROOM= N_ROOM

        self.SALE_COND = SALE_COND

        self.PARK_FACIL = PARK_FACIL

        self.BUILDTYPE = BUILDTYPE

        self.STREET = STREET
    
        self.PROP_AGE= PROP_AGE

        self.PRICE_PER_SQ_FT=PRICE_PER_SQ_FT

        self.UTILITY = UTILITY

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "AREA": [self.AREA],
                "INT_SQFT": [self.INT_SQFT],
                "N_BEDROOM": [self.N_BEDROOM],
                "N_BATHROOM": [self.N_BATHROOM],
                "N_ROOM": [self.N_ROOM],
                "SALE_COND": [self.SALE_COND],
                "PARK_FACIL": [self.PARK_FACIL],
                "BUILDTYPE":[self.BUILDTYPE],
                "STREET":[self.STREET],
                "PROP_AGE":[self.PROP_AGE],
                "PRICE_PER_SQ_FT":[self.PRICE_PER_SQ_FT],
                "UTILITY":[self.UTILITY]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)