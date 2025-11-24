from datetime import datetime
import os

MONGO_DATABASE_NAME = "Credit_card"
MONGO_COLLECTION_NAME = "Credit"
MONGO_DB_URL =  "mongodb://localhost:27017"

TARGET_COLUMN = "Class"  # 0 = Legitimate, 1 = Fraud

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder = "artifacts"

DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2