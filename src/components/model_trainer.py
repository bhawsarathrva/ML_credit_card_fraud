import sys
from typing import Dict, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from pathlib import Path
from dataclasses import dataclass

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.logger import logging
from src.utils.main_utils import MainUtils
from src.utils.visualization_utils import VisualizationUtils


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    model_report_path = os.path.join(artifact_folder, "model_report.txt")
    model_report_html_path = os.path.join(artifact_folder, "model_report.html")
    # For fraud detection, we prioritize recall (catching frauds) over accuracy
    expected_recall = 0.75  # We want to catch at least 75% of frauds
    expected_f1_score = 0.70  # Balance between precision and recall


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()
        
        # Models suitable for fraud detection with SMOTE-balanced data
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                scale_pos_weight=1,  # Will be adjusted based on class imbalance
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            "KNN_Smote": KNeighborsClassifier(
                n_neighbors=3,
                weights='distance',
                n_jobs=-1
            )
        }

    def evaluate_model_for_fraud_detection(self, model, X_train, y_train, X_test, y_test, model_name):
        """
        Comprehensive evaluation for fraud detection models.
        Returns multiple metrics with emphasis on fraud detection performance.
        """
        try:
            # Train the model
            logging.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Get prediction probabilities for ROC-AUC
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_test_proba = y_test_pred
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Fraud detection specific metrics (focus on class 1 - fraud)
            precision = precision_score(y_test, y_test_pred, zero_division=0)
            recall = recall_score(y_test, y_test_pred, zero_division=0)
            f1 = f1_score(y_test, y_test_pred, zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test, y_test_proba)
            except:
                roc_auc = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # Calculate false positive rate and false negative rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            metrics = {
                'model_name': model_name,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'false_positive_rate': fpr,
                'false_negative_rate': fnr
            }
            
            logging.info(f"{model_name} - Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, "
                        f"Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
            
            return metrics, model
            
        except Exception as e:
            logging.error(f"Error evaluating {model_name}: {str(e)}")
            raise CustomException(e, sys)

    def evaluate_all_models(self, X_train, y_train, X_test, y_test):
        """
        Evaluate all models and return comprehensive report.
        """
        try:
            logging.info("=" * 80)
            logging.info("EVALUATING ALL MODELS FOR FRAUD DETECTION")
            logging.info("=" * 80)
            
            model_results = []
            trained_models = {}
            
            for model_name, model in self.models.items():
                try:
                    metrics, trained_model = self.evaluate_model_for_fraud_detection(
                        model, X_train, y_train, X_test, y_test, model_name
                    )
                    model_results.append(metrics)
                    trained_models[model_name] = trained_model
                except Exception as e:
                    logging.warning(f"Failed to evaluate {model_name}: {str(e)}")
                    continue
            
            # Create DataFrame for easy comparison
            results_df = pd.DataFrame(model_results)
            
            # Sort by F1 score (best balance for fraud detection)
            results_df = results_df.sort_values('f1_score', ascending=False)
            
            logging.info("\n" + "=" * 80)
            logging.info("MODEL COMPARISON RESULTS")
            logging.info("=" * 80)
            logging.info("\n" + results_df.to_string(index=False))
            
            return results_df, trained_models
            
        except Exception as e:
            raise CustomException(e, sys)

    def select_best_model(self, results_df, trained_models):
        """
        Select best model based on fraud detection criteria.
        Priority: F1 Score > Recall > Precision > ROC-AUC
        """
        try:
            # Filter models that meet minimum requirements
            qualified_models = results_df[
                (results_df['recall'] >= self.model_trainer_config.expected_recall) &
                (results_df['f1_score'] >= self.model_trainer_config.expected_f1_score)
            ]
            
            if len(qualified_models) == 0:
                logging.warning("No models met the minimum requirements. Selecting best available model.")
                qualified_models = results_df
            
            # Select model with highest F1 score
            best_model_row = qualified_models.iloc[0]
            best_model_name = best_model_row['model_name']
            best_model = trained_models[best_model_name]
            
            logging.info("\n" + "=" * 80)
            logging.info(f"BEST MODEL SELECTED: {best_model_name}")
            logging.info("=" * 80)
            logging.info(f"Test Accuracy: {best_model_row['test_accuracy']:.4f}")
            logging.info(f"Precision: {best_model_row['precision']:.4f}")
            logging.info(f"Recall: {best_model_row['recall']:.4f}")
            logging.info(f"F1 Score: {best_model_row['f1_score']:.4f}")
            logging.info(f"ROC-AUC: {best_model_row['roc_auc']:.4f}")
            logging.info(f"False Positive Rate: {best_model_row['false_positive_rate']:.4f}")
            logging.info(f"False Negative Rate: {best_model_row['false_negative_rate']:.4f}")
            logging.info("=" * 80)
            
            return best_model, best_model_name, best_model_row
            
        except Exception as e:
            raise CustomException(e, sys)

    def save_model_report(self, results_df, best_model_name, best_model_metrics):
        """
        Save detailed model evaluation report to file.
        """
        try:
            report_path = self.model_trainer_config.model_report_path
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("CREDIT CARD FRAUD DETECTION - MODEL EVALUATION REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("ALL MODELS PERFORMANCE:\n")
                f.write("-" * 80 + "\n")
                f.write(results_df.to_string(index=False) + "\n\n")
                
                f.write("=" * 80 + "\n")
                f.write(f"BEST MODEL: {best_model_name}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Test Accuracy: {best_model_metrics['test_accuracy']:.4f}\n")
                f.write(f"Precision: {best_model_metrics['precision']:.4f}\n")
                f.write(f"Recall: {best_model_metrics['recall']:.4f}\n")
                f.write(f"F1 Score: {best_model_metrics['f1_score']:.4f}\n")
                f.write(f"ROC-AUC: {best_model_metrics['roc_auc']:.4f}\n\n")
                
                f.write("CONFUSION MATRIX:\n")
                f.write(f"True Positives (Frauds Caught): {best_model_metrics['true_positives']}\n")
                f.write(f"True Negatives (Legitimate Correctly Identified): {best_model_metrics['true_negatives']}\n")
                f.write(f"False Positives (Legitimate Flagged as Fraud): {best_model_metrics['false_positives']}\n")
                f.write(f"False Negatives (Frauds Missed): {best_model_metrics['false_negatives']}\n\n")
                
                f.write(f"False Positive Rate: {best_model_metrics['false_positive_rate']:.4f}\n")
                f.write(f"False Negative Rate: {best_model_metrics['false_negative_rate']:.4f}\n")
                f.write("=" * 80 + "\n")
            
            logging.info(f"Model evaluation report saved at: {report_path}")
            
        except Exception as e:
            logging.warning(f"Failed to save model report: {str(e)}")

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """
        Main method to train and evaluate fraud detection models.
        """
        try:
            logging.info("=" * 80)
            logging.info("STARTING MODEL TRAINING FOR FRAUD DETECTION")
            logging.info("=" * 80)
            
            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logging.info(f"Training data shape: {X_train.shape}")
            logging.info(f"Testing data shape: {X_test.shape}")
            logging.info(f"Training target distribution: Legitimate={sum(y_train==0)}, Fraud={sum(y_train==1)}")
            logging.info(f"Testing target distribution: Legitimate={sum(y_test==0)}, Fraud={sum(y_test==1)}")
            
            # Evaluate all models
            results_df, trained_models = self.evaluate_all_models(
                X_train, y_train, X_test, y_test
            )
            
            # Select best model
            best_model, best_model_name, best_model_metrics = self.select_best_model(
                results_df, trained_models
            )
            
            # Save model report
            self.save_model_report(results_df, best_model_name, best_model_metrics)
            
            # Save the best model
            model_path = self.model_trainer_config.trained_model_path
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            self.utils.save_object(file_path=model_path, obj=best_model)
            self.utils.save_object(file_path=model_path, obj=best_model)
            logging.info(f"Best model saved at: {model_path}")

            # Generate Visualizations and HTML Report
            self.generate_visualizations_and_report(
                trained_models, best_model, best_model_name, 
                X_train, y_train, X_test, y_test, results_df
            )
            
            # Return F1 score as the primary metric
            logging.info("=" * 80)
            logging.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
            logging.info("=" * 80)
            
            return best_model_metrics['f1_score']
            
        except Exception as e:
            logging.error("Error occurred during model training.")
            raise CustomException(e, sys)

    def generate_visualizations_and_report(self, trained_models, best_model, best_model_name, 
                                          X_train, y_train, X_test, y_test, results_df):
        """
        Generate comprehensive visualizations and HTML report.
        """
        try:
            logging.info("Generating visualizations and HTML report...")
            viz_dir = os.path.join(artifact_folder, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # 1. ROC Curves for all models
            roc_data = {}
            for name, model in trained_models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        auc = roc_auc_score(y_test, y_proba)
                        roc_data[name] = (fpr, tpr, auc)
                except Exception as e:
                    logging.warning(f"Could not generate ROC for {name}: {e}")
            
            VisualizationUtils.plot_multiple_roc_curves(
                roc_data=roc_data,
                save_path=os.path.join(viz_dir, "roc_curves.png")
            )
            
            # 2. Confusion Matrix for Best Model
            y_pred = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            VisualizationUtils.plot_confusion_matrix(
                cm=cm,
                labels=['Legitimate', 'Fraud'],
                title=f"Confusion Matrix - {best_model_name}",
                save_path=os.path.join(viz_dir, "confusion_matrix_best.png")
            )
            
            # 3. Feature Importance (if applicable)
            if hasattr(best_model, 'feature_importances_'):
                # Assuming feature names are generic if not provided
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
                # Try to get real feature names if possible (requires passing column names, skipping for now)
                
                VisualizationUtils.plot_feature_importance(
                    importances=best_model.feature_importances_,
                    feature_names=feature_names,
                    save_path=os.path.join(viz_dir, "feature_importance.png")
                )
            
            # 4. Create HTML Report
            self.create_html_report(results_df, best_model_name, viz_dir)
            
        except Exception as e:
            logging.error(f"Error generating visualizations: {str(e)}")
            # Don't raise exception here to avoid failing the whole pipeline if viz fails

    def create_html_report(self, results_df, best_model_name, viz_dir):
        """
        Create a comprehensive HTML report with embedded visualizations.
        """
        try:
            html_path = self.model_trainer_config.model_report_html_path
            
            # Convert images to base64 or relative paths
            # For simplicity, we'll use relative paths assuming the HTML is in artifacts/
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Fraud Detection Model Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
                    h1, h2 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .highlight {{ background-color: #e8f8f5; border-left: 5px solid #2ecc71; padding: 15px; }}
                    .viz-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
                    .viz-item {{ flex: 1; min-width: 45%; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <h1>Credit Card Fraud Detection - Model Evaluation Report</h1>
                
                <div class="highlight">
                    <h2>Best Performing Model: {best_model_name}</h2>
                    <p>Selected based on F1-Score and Recall balance.</p>
                </div>
                
                <h2>Model Comparison Metrics</h2>
                {results_df.to_html(index=False, classes='table')}
                
                <h2>Visualizations</h2>
                <div class="viz-container">
                    <div class="viz-item">
                        <h3>ROC Curves Comparison</h3>
                        <img src="visualizations/roc_curves.png" alt="ROC Curves">
                    </div>
                    <div class="viz-item">
                        <h3>Confusion Matrix ({best_model_name})</h3>
                        <img src="visualizations/confusion_matrix_best.png" alt="Confusion Matrix">
                    </div>
                </div>
                
                <div class="viz-container">
                    <div class="viz-item">
                        <h3>Feature Importance</h3>
                        <img src="visualizations/feature_importance.png" alt="Feature Importance" onerror="this.style.display='none'">
                    </div>
                    <div class="viz-item">
                        <h3>Class Distribution (Training)</h3>
                        <img src="visualizations/class_distribution_smote_tomek.png" alt="Class Distribution" onerror="this.style.display='none'">
                    </div>
                </div>
                
                <p>Generated on: {pd.Timestamp.now()}</p>
            </body>
            </html>
            """
            
            with open(html_path, 'w') as f:
                f.write(html_content)
                
            logging.info(f"HTML report saved at: {html_path}")
            
        except Exception as e:
            logging.error(f"Error creating HTML report: {str(e)}")


print("Model Trainer class initialized for Fraud Detection with SMOTE")