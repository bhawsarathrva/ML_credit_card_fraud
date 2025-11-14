from datetime import datetime
import os

MONGO_DATABASE_NAME = "Credit_card"
MONGO_COLLECTION_NAME = "Credit"
MONGO_DB_URL =  "mongodb+srv://hrisikesh:hrisikeshAndineuron@cluster0.iq9nlei.mongodb.net/?retryWrites=true&w=majority"

TARGET_COLUMN = "default payment next month"

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder = "artifacts"

DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2