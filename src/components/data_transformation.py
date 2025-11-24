import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTENC, ADASYN
from imblearn.combine import SMOTETomek
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constant import *
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
    smote_object_file_path = os.path.join(artifact_folder, 'smote.pkl')


class DataTransformation:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()

    def get_data_transformer_object(self):
        """
        Creates a preprocessing pipeline that imputes missing values
        and scales the numerical features using RobustScaler (better for outliers in fraud detection).
        """
        try:
            logging.info("Creating preprocessing pipeline for fraud detection...")

            # Using RobustScaler as it's more robust to outliers (common in fraud data)
            imputer_step = ('imputer', SimpleImputer(strategy='median'))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(
                steps=[
                    imputer_step,
                    scaler_step
                ]
            )

            logging.info("Preprocessor pipeline created successfully with RobustScaler.")
            return preprocessor

        except Exception as e:
            logging.error("Error occurred while creating data transformer object.")
            raise CustomException(e, sys)

    def apply_smote_technique(self, X_train, y_train):
        """
        Applies SMOTE-Tomek (combination of SMOTE oversampling and Tomek undersampling)
        to handle class imbalance in fraud detection dataset.
        
        SMOTE-Tomek is preferred over KNN-SMOTE for fraud detection because:
        - It combines oversampling minority class (fraud) with undersampling majority class
        - Removes Tomek links (borderline/noisy samples)
        - Better performance on highly imbalanced datasets
        """
        try:
            logging.info("Checking class distribution before SMOTE...")
            logging.info(f"Original class distribution: {Counter(y_train)}")
            
            # Calculate the imbalance ratio
            fraud_count = sum(y_train == 1)
            legitimate_count = sum(y_train == 0)
            imbalance_ratio = legitimate_count / fraud_count if fraud_count > 0 else 0
            
            logging.info(f"Imbalance ratio (legitimate/fraud): {imbalance_ratio:.2f}")
            
            # Use SMOTE-Tomek for better handling of imbalanced fraud detection data
            # sampling_strategy: ratio of minority class after resampling
            # We'll use 0.5 to make fraud cases 50% of legitimate cases (still imbalanced but manageable)
            smote_tomek = SMOTETomek(
                sampling_strategy=0.5,  # Fraud will be 50% of legitimate transactions
                random_state=42,
                n_jobs=-1
            )
            
            logging.info("Applying SMOTE-Tomek technique to balance the dataset...")
            X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
            
            logging.info(f"Class distribution after SMOTE-Tomek: {Counter(y_train_balanced)}")
            logging.info(f"Training samples increased from {len(y_train)} to {len(y_train_balanced)}")
            
            # Save SMOTE object for reference
            smote_path = self.data_transformation_config.smote_object_file_path
            self.utils.save_object(file_path=smote_path, obj=smote_tomek)
            logging.info(f"SMOTE-Tomek object saved at: {smote_path}")
            
            return X_train_balanced, y_train_balanced

        except Exception as e:
            logging.error("Error occurred while applying SMOTE technique.")
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for fraud detection.
                       It applies preprocessing, handles class imbalance using SMOTE-Tomek,
                       and prepares data for model training.
        
        Output      :   Returns transformed training & test arrays along with preprocessor path.
        On Failure  :   Logs exception and raises CustomException.
        """
        logging.info("Entered initiate_data_transformation method of DataTransformation class.")

        try:
            logging.info(f"Reading raw dataset from: {self.raw_data_dir}")
            dataframe = pd.read_csv(self.raw_data_dir)
            
            logging.info(f"Dataset shape: {dataframe.shape}")
            logging.info(f"Dataset columns: {dataframe.columns.tolist()}")
            
            # Force target column to be "Class" to avoid import issues
            target_col = "Class"
            logging.info(f"Forcing target column to: {target_col}")
            
            # Check if target column exists
            if target_col not in dataframe.columns:
                logging.error(f"Available columns: {dataframe.columns.tolist()}")
                raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {dataframe.columns.tolist()}")
            
            # Separate features and target
            X = dataframe.drop(columns=[target_col])
            y = dataframe[target_col]
            
            logging.info(f"Features shape: {X.shape}")
            logging.info(f"Target distribution in full dataset: {Counter(y)}")

            # Split into train and test sets (stratified to maintain class distribution)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logging.info("Successfully split dataset into training and testing sets (stratified).")
            logging.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

            # Get preprocessor pipeline
            preprocessor = self.get_data_transformer_object()

            # Fit and transform training data
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            logging.info("Applied preprocessing to training and testing data.")

            # Apply SMOTE-Tomek to handle class imbalance (only on training data)
            X_train_balanced, y_train_balanced = self.apply_smote_technique(
                X_train_scaled, y_train
            )

            # Save preprocessor object
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

            self.utils.save_object(
                file_path=preprocessor_path,
                obj=preprocessor
            )
            logging.info(f"Preprocessor object saved at: {preprocessor_path}")

            # Combine features and target into arrays
            train_arr = np.c_[X_train_balanced, np.array(y_train_balanced)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            # Save transformed arrays
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)
            logging.info("Transformed train and test arrays saved successfully.")
            
            logging.info(f"Final training data shape: {train_arr.shape}")
            logging.info(f"Final testing data shape: {test_arr.shape}")

            logging.info("Data Transformation phase completed successfully with SMOTE-Tomek.")
            return train_arr, test_arr, preprocessor_path

        except Exception as e:
            logging.error("Error occurred during data transformation.")
            raise CustomException(e, sys) from e
