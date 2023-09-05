
from src.components import data_ingestion
from src.components import data_preprocessing
from src.components import data_transformation
from src.components import model_trainer



if __name__=="__main__":
    obj =  data_ingestion.DataIngestion()
    data= obj.initiate_data_ingestion()
    pre = data_preprocessing.Preprocessor()
    data2= pre.feature_extraction(data)
    pre.is_null_present(data2)
    train_path, test_path = pre.split_data(data2)

    Dt= data_transformation.DataTransformation()
    train_array,test_array,processor = Dt.initiate_data_transformation(train_path,test_path)
    Mt = model_trainer.ModelTrainer()
    Mt.initiate_model_trainer(train_array,test_array)


    # train_arr,test_arr,preprocessor= DT.initiate_data_transformation(train_path,test_path)
    # MT= model_trainer.ModelTrainer()
    # print(MT.initiate_model_trainer(train_arr,test_arr))
