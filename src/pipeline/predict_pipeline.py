import shutil
import os
import sys
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from flask import request
from fpdf import FPDF
import datetime

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.utils.main_utils import MainUtils
from src.utils.visualization_utils import VisualizationUtils
        
        
@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name:str =  "predicted_file.csv"
    report_file_name:str = "prediction_report.pdf"
    trained_model_file_path: str = os.path.join(artifact_folder,"model.pkl")
    preprocessor_path: str = os.path.join(artifact_folder,"preprocessor.pkl")
    prediction_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)
    report_file_path:str = os.path.join(prediction_output_dirname, report_file_name)



class PredictionPipeline:
    def __init__(self, request: request):

        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()



    def save_input_files(self)-> str:

        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
            
            
            input_csv_file.save(pred_file_path)
            
            return pred_file_path
        except Exception as e:
            raise CustomException(e,sys)

    def predict(self, features):
        try:
            model_path = self.prediction_pipeline_config.trained_model_file_path
            preprocessor_path = self.prediction_pipeline_config.preprocessor_path


            model = self.utils.load_object(file_path=model_path)
            preprocessor = self.utils.load_object(file_path= preprocessor_path)

            transformed_features = preprocessor.transform(features)


            preds = model.predict(transformed_features)

            return preds

        except Exception as e:
            raise CustomException(e, sys)
        
    def generate_pdf_report(self, input_dataframe: pd.DataFrame, predictions: list):
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Credit Card Fraud Detection Report", ln=True, align="C")
            pdf.ln(10)
            
            # Metadata
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.cell(0, 10, f"Total Transactions Analyzed: {len(predictions)}", ln=True)
            
            # Statistics
            fraud_count = sum(predictions)
            legit_count = len(predictions) - fraud_count
            fraud_percentage = (fraud_count / len(predictions)) * 100
            
            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Summary Statistics", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Legitimate Transactions: {legit_count}", ln=True)
            pdf.set_font("Arial", "", 12)
            # Highlight fraud count in red if > 0
            if fraud_count > 0:
                pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 10, f"Fraudulent Transactions Detected: {fraud_count}", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 10, f"Fraud Percentage: {fraud_percentage:.2f}%", ln=True)
            
            pdf.ln(10)
            
            # Detailed List of Frauds (if any)
            if fraud_count > 0:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Detected Fraudulent Transactions", ln=True)
                pdf.set_font("Arial", "I", 10)
                pdf.cell(0, 10, "(Showing first 50 detected frauds)", ln=True)
                
                pdf.set_font("Arial", "", 10)
                
                # Add table header
                # Assuming 'Time' and 'Amount' are in the dataframe, if not, we use index
                cols = input_dataframe.columns.tolist()
                has_amount = 'Amount' in cols
                has_time = 'Time' in cols
                
                header = "Index | "
                if has_time: header += "Time | "
                if has_amount: header += "Amount | "
                header += "Prediction"
                
                pdf.cell(0, 10, header, ln=True)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                
                count = 0
                for idx, pred in enumerate(predictions):
                    if pred == 1:
                        row_str = f"{idx} | "
                        if has_time: row_str += f"{input_dataframe.iloc[idx]['Time']} | "
                        if has_amount: row_str += f"{input_dataframe.iloc[idx]['Amount']} | "
                        row_str += "FRAUD"
                        
                        pdf.cell(0, 10, row_str, ln=True)
                        count += 1
                        if count >= 50:
                            pdf.cell(0, 10, "... and more", ln=True)
                            break
            else:
                pdf.set_font("Arial", "I", 12)
                pdf.cell(0, 10, "No fraudulent transactions were detected in this batch.", ln=True)
                
            # Generate Visualizations
            viz_paths = self.generate_visualizations(input_dataframe, predictions)
            
            # Add Visualizations to PDF
            if viz_paths:
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Visual Analysis", ln=True)
                pdf.ln(5)
                
                y_pos = pdf.get_y()
                
                # Fraud Distribution
                if 'fraud_dist' in viz_paths and os.path.exists(viz_paths['fraud_dist']):
                    pdf.image(viz_paths['fraud_dist'], x=10, y=y_pos, w=90)
                    
                # Amount Distribution
                if 'amount_dist' in viz_paths and os.path.exists(viz_paths['amount_dist']):
                    pdf.image(viz_paths['amount_dist'], x=110, y=y_pos, w=90)
                
                pdf.ln(80) # Move down
                
                # Timeline
                if 'timeline' in viz_paths and os.path.exists(viz_paths['timeline']):
                    pdf.image(viz_paths['timeline'], x=10, y=pdf.get_y(), w=190)
            
            # Save PDF
            pdf.output(self.prediction_pipeline_config.report_file_path)
            logging.info(f"PDF report generated at {self.prediction_pipeline_config.report_file_path}")
            
        except Exception as e:
            logging.error(f"Error generating PDF report: {str(e)}")
            raise CustomException(e, sys)

    def generate_visualizations(self, df: pd.DataFrame, predictions: list) -> dict:
        """
        Generate visualizations for the prediction report.
        """
        try:
            viz_dir = os.path.join(self.prediction_pipeline_config.prediction_output_dirname, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            viz_paths = {}
            
            # 1. Fraud Distribution
            fraud_dist_path = os.path.join(viz_dir, "fraud_distribution.png")
            VisualizationUtils.plot_fraud_distribution(
                predictions=predictions,
                save_path=fraud_dist_path
            )
            viz_paths['fraud_dist'] = fraud_dist_path
            
            # 2. Amount Distribution (if Amount column exists)
            if 'Amount' in df.columns:
                amount_dist_path = os.path.join(viz_dir, "amount_distribution.png")
                VisualizationUtils.plot_amount_distribution(
                    df=df,
                    predictions=predictions,
                    save_path=amount_dist_path
                )
                viz_paths['amount_dist'] = amount_dist_path
            
            # 3. Timeline (if Time column exists)
            if 'Time' in df.columns:
                timeline_path = os.path.join(viz_dir, "fraud_timeline.png")
                VisualizationUtils.plot_fraud_timeline(
                    df=df,
                    predictions=predictions,
                    save_path=timeline_path
                )
                viz_paths['timeline'] = timeline_path
                
            return viz_paths
            
        except Exception as e:
            logging.error(f"Error generating prediction visualizations: {str(e)}")
            return {}

    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):

        try:
            prediction_column_name : str = TARGET_COLUMN
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            
            input_dataframe =  input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe

            predictions = self.predict(input_dataframe)
            
            # Generate PDF Report before modifying dataframe
            self.generate_pdf_report(input_dataframe, predictions)
            
            input_dataframe[prediction_column_name] = [pred for pred in predictions]


            
            os.makedirs( self.prediction_pipeline_config.prediction_output_dirname, exist_ok= True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)
            logging.info("predictions completed. ")



        except Exception as e:
            raise CustomException(e, sys) from e
        

        
    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_pipeline_config


        except Exception as e:
            raise CustomException(e,sys)        