import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pathlib import Path
from src.constant import *
from src.exception import CustomException
from src.logger import logging

from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(artifact_folder)
    raw_data_path: str = os.path.join(data_ingestion_dir, "card_data.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def export_collection_as_dataframe(self, collection_name, db_name):
        """
        Method Name :   export_collection_as_dataframe
        Description :   This method exports MongoDB collection as a pandas DataFrame
        
        Output      :   DataFrame containing collection data
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.0
        """
        try:
            mongo_client = MongoClient("mongodb://localhost:27017")
            
            database = mongo_client["Credit_card"]
            collection = database["Credit"]
            
            logging.info(f"Connecting to MongoDB - Database: {Credit_card}, Collection: {Credit}")
            
            df = pd.DataFrame(list(collection.find()))
            
            if df.empty:
                logging.warning(f"No data found in collection: {collection_name}")
                raise ValueError(f"Collection {collection_name} is empty")
            
            logging.info(f"Successfully retrieved {len(df)} records from MongoDB")
            
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            
            df.replace({"na": np.nan}, inplace=True)
            
            mongo_client.close()
            
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_raw_data_dir(self) -> pd.DataFrame:
        """
        Method Name :   export_data_into_raw_data_dir
        Description :   This method reads data from MongoDB and saves it into artifacts
        
        Output      :   dataset is returned as a pd.DataFrame
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.0
        """
        try:
            logging.info("Exporting data from MongoDB")
            
            # Create raw data directory if it doesn't exist
            raw_data_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(raw_data_dir, exist_ok=True)
            logging.info(f"Created directory: {raw_data_dir}")
            
            # Get raw data path
            raw_data_path = self.data_ingestion_config.raw_data_path
            
            # Export collection from MongoDB
            logging.info(f"Fetching data from MongoDB - Database: {MONGO_DATABASE_NAME}, Collection: {MONGO_COLLECTION_NAME}")
            dataset = self.export_collection_as_dataframe(
                db_name=MONGO_DATABASE_NAME,
                collection_name=MONGO_COLLECTION_NAME
            )
            
            # Save dataset to CSV
            logging.info(f"Saving exported data to: {raw_data_path}")
            dataset.to_csv(raw_data_path, index=False)
            logging.info(f"Successfully saved {len(dataset)} records to CSV")
            
            return dataset

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> str:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline
        
        Output      :   path to raw data CSV file
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            # Export data from MongoDB to CSV
            self.export_data_into_raw_data_dir()
            
            logging.info("Successfully retrieved data from MongoDB")
            
            # Verify file exists
            if not os.path.exists(self.data_ingestion_config.raw_data_path):
                raise FileNotFoundError(f"Raw data file not created at: {self.data_ingestion_config.raw_data_path}")
            
            logging.info(f"Data ingestion completed successfully. File saved at: {self.data_ingestion_config.raw_data_path}")
            logging.info("Exited initiate_data_ingestion method of DataIngestion class")

            return self.data_ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys) from e