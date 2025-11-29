import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logging
from src.exception import CustomException

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class VisualizationUtils:
    """
    Utility class for creating visualizations for fraud detection model
    """
    
    @staticmethod
    def ensure_dir(file_path: str):
        """Ensure directory exists for file path"""
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def plot_class_distribution(
        y_data: np.ndarray,
        title: str,
        save_path: str,
        labels: List[str] = None
    ) -> str:
        """
        Create bar chart for class distribution
        
        Args:
            y_data: Target variable array
            title: Plot title
            save_path: Path to save the plot
            labels: Custom labels for classes
            
        Returns:
            Path to saved plot
        """
        try:
            VisualizationUtils.ensure_dir(save_path)
            
            # Count classes
            unique, counts = np.unique(y_data, return_counts=True)
            
            if labels is None:
                labels = [f'Class {int(u)}' for u in unique]
            
            # Create figure
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(unique)), counts, color=['#2ecc71', '#e74c3c'])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.xlabel('Class', fontsize=12, fontweight='bold')
            plt.ylabel('Count', fontsize=12, fontweight='bold')
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xticks(range(len(unique)), labels)
            plt.grid(axis='y', alpha=0.3)
            
            # Add percentage labels
            total = sum(counts)
            for i, count in enumerate(counts):
                percentage = (count / total) * 100
                plt.text(i, count/2, f'{percentage:.2f}%',
                        ha='center', va='center', fontsize=11,
                        color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Class distribution plot saved at: {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating class distribution plot: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def plot_sampling_comparison(
        original_counts: Dict[str, int],
        smote_tomek_counts: Dict[str, int],
        knn_smote_counts: Dict[str, int],
        save_path: str
    ) -> str:
        """
        Compare different sampling strategies
        
        Args:
            original_counts: Original class distribution
            smote_tomek_counts: After SMOTE-Tomek
            knn_smote_counts: After KNN-SMOTE
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            VisualizationUtils.ensure_dir(save_path)
            
            # Prepare data
            strategies = ['Original', 'SMOTE-Tomek', 'KNN-SMOTE']
            legitimate = [
                original_counts.get(0, 0),
                smote_tomek_counts.get(0, 0),
                knn_smote_counts.get(0, 0)
            ]
            fraud = [
                original_counts.get(1, 0),
                smote_tomek_counts.get(1, 0),
                knn_smote_counts.get(1, 0)
            ]
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Grouped bar chart
            x = np.arange(len(strategies))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, legitimate, width, label='Legitimate', color='#2ecc71')
            bars2 = ax1.bar(x + width/2, fraud, width, label='Fraud', color='#e74c3c')
            
            ax1.set_xlabel('Sampling Strategy', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax1.set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(strategies)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height):,}',
                            ha='center', va='bottom', fontsize=9)
            
            # Imbalance ratio chart
            ratios = []
            for leg, fr in zip(legitimate, fraud):
                ratio = leg / fr if fr > 0 else 0
                ratios.append(ratio)
            
            bars = ax2.bar(strategies, ratios, color=['#3498db', '#9b59b6', '#f39c12'])
            ax2.set_xlabel('Sampling Strategy', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Imbalance Ratio (Legitimate/Fraud)', fontsize=12, fontweight='bold')
            ax2.set_title('Imbalance Ratio Comparison', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Sampling comparison plot saved at: {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating sampling comparison plot: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def plot_roc_curve(
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        model_name: str,
        save_path: str
    ) -> str:
        """
        Plot ROC curve for a single model
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            auc_score: AUC score
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            VisualizationUtils.ensure_dir(save_path)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='#e74c3c', lw=2,
                    label=f'{model_name} (AUC = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--',
                    label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"ROC curve saved at: {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating ROC curve: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def plot_multiple_roc_curves(
        roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
        save_path: str
    ) -> str:
        """
        Plot ROC curves for multiple models on same plot
        
        Args:
            roc_data: Dictionary with model_name: (fpr, tpr, auc_score)
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            VisualizationUtils.ensure_dir(save_path)
            
            plt.figure(figsize=(12, 8))
            
            # Color palette for different models
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
            
            for idx, (model_name, (fpr, tpr, auc_score)) in enumerate(roc_data.items()):
                color = colors[idx % len(colors)]
                plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{model_name} (AUC = {auc_score:.4f})')
            
            plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--',
                    label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Multiple ROC curves saved at: {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating multiple ROC curves: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        labels: List[str],
        title: str,
        save_path: str,
        normalize: bool = False
    ) -> str:
        """
        Plot confusion matrix heatmap
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            title: Plot title
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            Path to saved plot
        """
        try:
            VisualizationUtils.ensure_dir(save_path)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2%'
            else:
                fmt = 'd'
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='RdYlGn_r',
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Count' if not normalize else 'Percentage'},
                       linewidths=1, linecolor='gray')
            
            plt.ylabel('True Label', fontsize=12, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
            plt.title(title, fontsize=14, fontweight='bold')
            
            # Add text annotations for better clarity
            plt.text(0.5, -0.1, 'TN: True Negative | FP: False Positive',
                    ha='center', transform=plt.gca().transAxes, fontsize=9)
            plt.text(0.5, -0.15, 'FN: False Negative | TP: True Positive',
                    ha='center', transform=plt.gca().transAxes, fontsize=9)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Confusion matrix saved at: {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating confusion matrix: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def plot_feature_importance(
        importances: np.ndarray,
        feature_names: List[str],
        save_path: str,
        top_n: int = 20
    ) -> str:
        """
        Plot feature importance bar chart
        
        Args:
            importances: Feature importance values
            feature_names: Names of features
            save_path: Path to save the plot
            top_n: Number of top features to display
            
        Returns:
            Path to saved plot
        """
        try:
            VisualizationUtils.ensure_dir(save_path)
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1][:top_n]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            
            plt.figure(figsize=(12, 8))
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
            bars = plt.barh(range(len(top_features)), top_importances, color=colors)
            
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
            plt.ylabel('Features', fontsize=12, fontweight='bold')
            plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, top_importances)):
                plt.text(importance, i, f' {importance:.4f}',
                        va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Feature importance plot saved at: {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating feature importance plot: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def plot_fraud_distribution(
        predictions: np.ndarray,
        save_path: str
    ) -> str:
        """
        Create pie chart for fraud distribution
        
        Args:
            predictions: Array of predictions (0=legitimate, 1=fraud)
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            VisualizationUtils.ensure_dir(save_path)
            
            # Count predictions
            unique, counts = np.unique(predictions, return_counts=True)
            labels = ['Legitimate', 'Fraudulent']
            colors = ['#2ecc71', '#e74c3c']
            
            # Handle case where only one class is predicted
            if len(unique) == 1:
                if unique[0] == 0:
                    counts = [counts[0], 0]
                else:
                    counts = [0, counts[0]]
            
            plt.figure(figsize=(10, 8))
            wedges, texts, autotexts = plt.pie(counts, labels=labels, colors=colors,
                                               autopct='%1.1f%%', startangle=90,
                                               explode=(0, 0.1), shadow=True)
            
            # Enhance text
            for text in texts:
                text.set_fontsize(12)
                text.set_fontweight('bold')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(11)
                autotext.set_fontweight('bold')
            
            plt.title('Fraud Detection Results', fontsize=14, fontweight='bold')
            
            # Add legend with counts
            legend_labels = [f'{label}: {count:,}' for label, count in zip(labels, counts)]
            plt.legend(legend_labels, loc='upper right', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Fraud distribution plot saved at: {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating fraud distribution plot: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def plot_amount_distribution(
        df: pd.DataFrame,
        predictions: np.ndarray,
        save_path: str,
        amount_col: str = 'Amount'
    ) -> str:
        """
        Plot amount distribution for frauds vs legitimate transactions
        
        Args:
            df: DataFrame with transaction data
            predictions: Array of predictions
            save_path: Path to save the plot
            amount_col: Name of amount column
            
        Returns:
            Path to saved plot
        """
        try:
            VisualizationUtils.ensure_dir(save_path)
            
            if amount_col not in df.columns:
                logging.warning(f"Column '{amount_col}' not found. Skipping amount distribution plot.")
                return None
            
            # Separate amounts by prediction
            df_copy = df.copy()
            df_copy['Prediction'] = predictions
            
            legitimate_amounts = df_copy[df_copy['Prediction'] == 0][amount_col]
            fraud_amounts = df_copy[df_copy['Prediction'] == 1][amount_col]
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Histogram comparison
            axes[0, 0].hist(legitimate_amounts, bins=50, alpha=0.7, color='#2ecc71',
                          label='Legitimate', edgecolor='black')
            axes[0, 0].hist(fraud_amounts, bins=50, alpha=0.7, color='#e74c3c',
                          label='Fraudulent', edgecolor='black')
            axes[0, 0].set_xlabel('Transaction Amount', fontsize=11, fontweight='bold')
            axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
            axes[0, 0].set_title('Amount Distribution Comparison', fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # Box plot
            data_to_plot = [legitimate_amounts, fraud_amounts]
            bp = axes[0, 1].boxplot(data_to_plot, labels=['Legitimate', 'Fraudulent'],
                                   patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
                patch.set_facecolor(color)
            axes[0, 1].set_ylabel('Transaction Amount', fontsize=11, fontweight='bold')
            axes[0, 1].set_title('Amount Distribution - Box Plot', fontsize=12, fontweight='bold')
            axes[0, 1].grid(alpha=0.3)
            
            # Statistics table
            stats_data = {
                'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Legitimate': [
                    len(legitimate_amounts),
                    legitimate_amounts.mean() if len(legitimate_amounts) > 0 else 0,
                    legitimate_amounts.median() if len(legitimate_amounts) > 0 else 0,
                    legitimate_amounts.std() if len(legitimate_amounts) > 0 else 0,
                    legitimate_amounts.min() if len(legitimate_amounts) > 0 else 0,
                    legitimate_amounts.max() if len(legitimate_amounts) > 0 else 0
                ],
                'Fraudulent': [
                    len(fraud_amounts),
                    fraud_amounts.mean() if len(fraud_amounts) > 0 else 0,
                    fraud_amounts.median() if len(fraud_amounts) > 0 else 0,
                    fraud_amounts.std() if len(fraud_amounts) > 0 else 0,
                    fraud_amounts.min() if len(fraud_amounts) > 0 else 0,
                    fraud_amounts.max() if len(fraud_amounts) > 0 else 0
                ]
            }
            
            axes[1, 0].axis('tight')
            axes[1, 0].axis('off')
            table = axes[1, 0].table(cellText=[[f'{v:.2f}' if isinstance(v, float) else v 
                                               for v in row] 
                                              for row in zip(*stats_data.values())],
                                    colLabels=stats_data.keys(),
                                    cellLoc='center',
                                    loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            axes[1, 0].set_title('Statistical Summary', fontsize=12, fontweight='bold')
            
            # Cumulative distribution
            if len(legitimate_amounts) > 0:
                axes[1, 1].hist(legitimate_amounts, bins=50, cumulative=True,
                              alpha=0.7, color='#2ecc71', label='Legitimate',
                              density=True, histtype='step', linewidth=2)
            if len(fraud_amounts) > 0:
                axes[1, 1].hist(fraud_amounts, bins=50, cumulative=True,
                              alpha=0.7, color='#e74c3c', label='Fraudulent',
                              density=True, histtype='step', linewidth=2)
            axes[1, 1].set_xlabel('Transaction Amount', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
            axes[1, 1].set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Amount distribution plot saved at: {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating amount distribution plot: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def plot_fraud_timeline(
        df: pd.DataFrame,
        predictions: np.ndarray,
        save_path: str,
        time_col: str = 'Time'
    ) -> str:
        """
        Plot fraud distribution over time
        
        Args:
            df: DataFrame with transaction data
            predictions: Array of predictions
            save_path: Path to save the plot
            time_col: Name of time column
            
        Returns:
            Path to saved plot or None if time column not found
        """
        try:
            VisualizationUtils.ensure_dir(save_path)
            
            if time_col not in df.columns:
                logging.warning(f"Column '{time_col}' not found. Skipping timeline plot.")
                return None
            
            df_copy = df.copy()
            df_copy['Prediction'] = predictions
            
            # Convert time to hours if it's in seconds
            df_copy['Time_Hours'] = df_copy[time_col] / 3600
            
            # Separate by prediction
            legitimate = df_copy[df_copy['Prediction'] == 0]
            fraudulent = df_copy[df_copy['Prediction'] == 1]
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Scatter plot
            axes[0].scatter(legitimate['Time_Hours'], legitimate.index,
                          alpha=0.3, s=10, color='#2ecc71', label='Legitimate')
            axes[0].scatter(fraudulent['Time_Hours'], fraudulent.index,
                          alpha=0.7, s=30, color='#e74c3c', label='Fraudulent', marker='x')
            axes[0].set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Transaction Index', fontsize=11, fontweight='bold')
            axes[0].set_title('Fraud Detection Timeline', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # Histogram over time
            bins = 50
            axes[1].hist(legitimate['Time_Hours'], bins=bins, alpha=0.7,
                        color='#2ecc71', label='Legitimate', edgecolor='black')
            axes[1].hist(fraudulent['Time_Hours'], bins=bins, alpha=0.7,
                        color='#e74c3c', label='Fraudulent', edgecolor='black')
            axes[1].set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('Number of Transactions', fontsize=11, fontweight='bold')
            axes[1].set_title('Transaction Distribution Over Time', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Fraud timeline plot saved at: {save_path}")
            return save_path
            
        except Exception as e:
            logging.error(f"Error creating fraud timeline plot: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("Visualization Utils module loaded successfully!")
    print("Available visualization functions:")
    print("  - plot_class_distribution")
    print("  - plot_sampling_comparison")
    print("  - plot_roc_curve")
    print("  - plot_multiple_roc_curves")
    print("  - plot_confusion_matrix")
    print("  - plot_feature_importance")
    print("  - plot_fraud_distribution")
    print("  - plot_amount_distribution")
    print("  - plot_fraud_timeline")
