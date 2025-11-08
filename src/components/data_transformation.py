import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from src.constant import *                      # Contains artifact_folder, TARGET_COLUMN, etc.
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class DataTransformationConfig:
    """
    Configuration for paths to save transformed data and preprocessor object.
    """
    transformed_train_file_path = os.path.join(artifact_folder, 'train.npy')
    transformed_test_file_path = os.path.join(artifact_folder, 'test.npy')
    transformed_object_file_path = os.path.join(artifact_folder, 'preprocessor.pkl')


class DataTransformation:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()

    def get_data_transformer_object(self):
        """
        Creates a preprocessing pipeline that imputes missing values
        and scales the numerical features.
        """
        try:
            logging.info("Creating preprocessing pipeline...")

            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', StandardScaler())

            preprocessor = Pipeline(
                steps=[
                    imputer_step,
                    scaler_step
                ]
            )

            logging.info("Preprocessor pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error("Error occurred while creating data transformer object.")
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline.
        
        Output      :   Returns transformed training & test arrays along with preprocessor path.
        On Failure  :   Logs exception and raises CustomException.
        """
        logging.info("Entered initiate_data_transformation method of DataTransformation class.")

        try:
            # Step 1: Load the dataset
            logging.info(f"Reading raw dataset from: {self.raw_data_dir}")
            dataframe = pd.read_csv(self.raw_data_dir)

            # Step 2: Separate features and target
            X = dataframe.drop(columns=[TARGET_COLUMN])
            y = dataframe[TARGET_COLUMN]

            # Step 3: Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logging.info("Successfully split dataset into training and testing sets.")

            # Step 4: Get preprocessing object
            preprocessor = self.get_data_transformer_object()

            # Step 5: Fit on train, transform both
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            logging.info("Applied preprocessing to training and testing data.")

            # Step 6: Save preprocessor object
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

            self.utils.save_object(
                file_path=preprocessor_path,
                obj=preprocessor
            )
            logging.info(f"Preprocessor object saved at: {preprocessor_path}")

            # Step 7: Combine scaled features and target column
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            # Step 8: Save transformed arrays (optional but good practice)
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)
            logging.info("Transformed train and test arrays saved successfully.")

            logging.info("Data Transformation phase completed successfully.")
            return train_arr, test_arr, preprocessor_path

        except Exception as e:
            logging.error("Error occurred during data transformation.")
            raise CustomException(e, sys) from e
