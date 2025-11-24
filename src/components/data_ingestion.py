import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pathlib import Path
from dataclasses import dataclass

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(artifact_folder) 
    raw_data_path: str = os.path.join(data_ingestion_dir, "Card_data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def export_collection_as_dataframe(self, collection_name, db_name):
        try:
            # Use constants for MongoDB connection
            mongo_url = MONGO_DB_URL
            logging.info(f"Connecting to MongoDB at: {mongo_url}")
            
            mongo_client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
            database = mongo_client[db_name]
            collection = database[collection_name]
            
            logging.info(f"Connecting to MongoDB - Database: {db_name}, Collection: {collection_name}")
            
            # Test connection
            mongo_client.admin.command('ping')
            logging.info("MongoDB connection successful")
            
            df = pd.DataFrame(list(collection.find()))
            
            if df.empty:
                logging.warning(f"No data found in collection: {collection_name}")
                raise ValueError(f"Collection {collection_name} is empty. Please upload data to MongoDB first.")
            
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
        """
        try:
            logging.info("Exporting data from MongoDB")
            
            raw_data_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(raw_data_dir, exist_ok=True)
            logging.info(f"Created directory: {raw_data_dir}")

            raw_data_path = self.data_ingestion_config.raw_data_path
            
            # Use constants for database and collection names
            db_name = MONGO_DATABASE_NAME
            collection_name = MONGO_COLLECTION_NAME
            
            logging.info(f"Fetching data from MongoDB - Database: {db_name}, Collection: {collection_name}")
            
            dataset = self.export_collection_as_dataframe(
                db_name=db_name,
                collection_name=collection_name
            )
            
            logging.info(f"Saving exported data to: {raw_data_path}")
            dataset.to_csv(raw_data_path, index=False)
            logging.info(f"Successfully saved {len(dataset)} records to CSV")
            
            return dataset

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> str:
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            self.export_data_into_raw_data_dir()
            
            logging.info("Successfully retrieved data from MongoDB")
            
            if not os.path.exists(self.data_ingestion_config.raw_data_path):
                raise FileNotFoundError(f"Raw data file not created at: {self.data_ingestion_config.raw_data_path}")
            
            logging.info(f"Data ingestion completed successfully. File saved at: {self.data_ingestion_config.raw_data_path}")
            logging.info("Exited initiate_data_ingestion method of DataIngestion class")

            return self.data_ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys) from e