import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load the Iris dataset (You can update the path to where your dataset is located)
            df = pd.read_csv('notebook/data/iris_data.csv')
            logging.info('Read the dataset as dataframe')

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            # Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved at: {0}".format(self.ingestion_config.raw_data_path))

            # Train-Test Split initiated
            logging.info("Train-Test Split initiated")
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            # Save the train and test split datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Ingestion Completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Initialize Data Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Model Trainer
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)
