import os,sys
import pandas as pd 
import numpy as np 
#import src as src
#import sys
#sys.path.append("C:\Users\Alisha\Desktop\7 days ML\ml_pipeline_project\")
import sys
sys.path.append('C:/Users/Alisha/Desktop/7 days ML/ml_pipeline_project')  # Adjust the path accordingly

# Add the path to the 'src' directory
#src_path = os.path.abspath("C:/Users/Alisha/Desktop/7 days ML/ml_pipeline_project/src")
#sys.path.append(src_path)

# Now you can import modules from 'src'

from src.logger import logging
from src.exception import CustomException

#from src.logger import logging
#from src.exception import CustomException

# Rest of your code

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")

# C:\Users\Alisha\Desktop\7 days ML\ml_pipeline_project\notebook\data\income_cleandata.csv
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            logging.info("Data reading using pandas libraryfrom local system")
            data = pd.read_csv(os.path.join("notebook/data","income_cleandata.csv"))
            logging.info("Data reading completed")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Data splited intotrain and test")

            train_set, test_set = train_test_split(data, test_size= .30, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False)

            logging.info("Data ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            logging.info("Error occured in data ingestion stage")
            raise CustomException(e, sys)
         
if __name__ =='__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()

# src\components\data_ingestion.py