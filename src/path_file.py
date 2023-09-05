import os

base_path =os.getcwd()   # base path


train_data_path: str = os.path.join('artifacts', "train.csv")
test_data_path: str = os.path.join('artifacts', "test.csv")
raw_data_path: str = os.path.join('artifacts', "data.csv")


# after applying preprocessor object train arrray path
train_arr_path:str=os.path.join('artifacts',"train_arr.csv")

# after applying preprocessor object test arrray path
test_arr_path:str=os.path.join('artifacts',"test_arr.csv")


# missing columns data path
missing_column_path:str=os.path.join('artifacts',"missing_data_column.csv")

# after doing feature extraction data path
# feature_extracted_data_path: str = os.path.join('artifacts', "feature_extracted_data.csv")


# preprocessor object file path
preprocessor_obj_file_path= os.path.join('artifacts', "preprocessor.pkl")

# best model path after training and hyper-tuning
trained_model_file_path= os.path.join('artifacts', "model.pkl")