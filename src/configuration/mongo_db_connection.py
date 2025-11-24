import os
import sys
from pathlib import Path

import certifi
import pymongo

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constant import *
from src.exception import CustomException

ca = certifi.where()

class MongoDBClient:
    client = None

    def __init__(self, database_name=MONGO_DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv("MONGO_DB_URL")
                if mongo_db_url is None:
                    raise Exception("Environment key: MONGO_DB_URL is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client["Credit_card"]
            self.database_name = "Credit_card"
            print("MongoDBClient class initialized")
        except Exception as e:
            raise CustomException(e, sys)
