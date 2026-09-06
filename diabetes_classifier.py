"""
Diabetes Classifier

A modular machine learning pipeline to classify diabetic patients 
using clinical health metrics. This pipeline addresses class imbalances using 
SMOTE (Synthetic Minority Over-sampling Technique) and evaluates the models under 
clinical risk constraints where model Recall is prioritized over raw Accuracy.

Author: LeoFlores0
Date: August 2026
"""

import os
import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
matplotlib.use('Agg')  # Headless mode for server environments

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DiabetesPipeline:
    """
    A machine learning pipeline for diabetic screening and diagnostic classification.
    
    Initializes data loading, preprocessing, feature scaling, and synthetic 
    oversampling (SMOTE) immediately upon instantiation to guarantee state.
    """

    def __init__(self, filepath: str, test_size: float = 0.2, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=self.random_state)
        
        # Models
        self.model_orig = LogisticRegression(solver='liblinear', max_iter=200, random_state=self.random_state)
        self.model_smote = LogisticRegression(solver='liblinear', max_iter=200, random_state=self.random_state)
        
        # Load Data
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at: {filepath}")
        self.df: pd.DataFrame = pd.read_csv(filepath)
        logger.info("Successfully loaded dataset. Shape: %s", self.df.shape)
        
        # Preprocess and Split
        X = self.df.drop('Outcome', axis=1)
        y = self.df['Outcome']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        logger.info("Data split complete. Train shape: %s, Test shape: %s", self.X_train.shape, self.X_test.shape)

        # Scale Features
        self.X_train_scaled: np.ndarray = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled: np.ndarray = self.scaler.transform(self.X_test)
        logger.info("Feature scaling completed using StandardScaler.")

        # Apply SMOTE
        resampled = self.smote.fit_resample(self.X_train_scaled, self.y_train)
        self.X_train_smote, self.y_train_smote = resampled[0], resampled[1]
        logger.info("SMOTE oversampling applied.")
        
        # Evaluation metrics storage
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def train_and_evaluate(self) -> None:
        """Trains the original and SMOTE-augmented models and computes metrics."""
        logger.info("Training baseline model on original imbalanced training data...")
        self.model_orig.fit(self.X_train_scaled, self.y_train)
        
        logger.info("Training balanced model on SMOTE-augmented training data...")
        self.model_smote.fit(self.X_train_smote, self.y_train_smote)
        
        # Predictions
        y_pred_orig = self.model_orig.predict(self.X_test_scaled)
        y_prob_orig = self.model_orig.predict_proba(self.X_test_scaled)[:, 1]
        
        y_pred_smote = self.model_smote.predict(self.X_test_scaled)
        y_prob_smote = self.model_smote.predict_proba(self.X_test_scaled)[:, 1]
        
        # Save metrics
        self.metrics['Original Data'] = {
            'accuracy': accuracy_score(self.y_test, y_pred_orig),
            'precision': precision_score(self.y_test, y_pred_orig),
            'recall': recall_score(self.y_test, y_pred_orig),
            'auc': roc_auc_score(self.y_test, y_prob_orig),
            'y_prob': y_prob_orig,
            'y_pred': y_pred_orig,
            'report': classification_report(self.y_test, y_pred_orig)
        }
        
        self.metrics['SMOTE Augmented Data'] = {
            'accuracy': accuracy_score(self.y_test, y_pred_smote),
            'precision': precision_score(self.y_test, y_pred_smote),
            'recall': recall_score(self.y_test, y_pred_smote),
            'auc': roc_auc_score(self.y_test, y_prob_smote),
            'y_prob': y_prob_smote,
            'y_pred': y_pred_smote,
            'report': classification_report(self.y_test, y_pred_smote)
        }
        
        for name, results in self.metrics.items():
            print(f"\nPerformance on {name}:\n")
            print(f"Accuracy:  {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall:    {results['recall']:.4f}")
            print(f"ROC-AUC:   {results['auc']:.4f}")
            print(f"\nClassification Report:\n{results['report']}")

    def plot_roc_curves(self, save_path: str) -> None:
        """Generates an ROC curve comparison plot."""
        if not self.metrics:
            raise ValueError("Models have not been evaluated. Call train_and_evaluate() first.")
            
        plt.figure(figsize=(9, 7), dpi=300)
        
        # Custom professional styles
        colors = {'Original Data': '#2c3e50', 'SMOTE Augmented Data': '#e74c3c'}
        linestyles = {'Original Data': '--', 'SMOTE Augmented Data': '-'}
        linewidths = {'Original Data': 1.5, 'SMOTE Augmented Data': 2.2}
        
        for name, results in self.metrics.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_prob'])
            plt.plot(
                fpr, tpr, 
                label=f"{name} (AUC = {results['auc']:.4f})", 
                color=colors[name], 
                linestyle=linestyles[name],
                linewidth=linewidths[name]
            )
            
        # Draw the random-chance diagonal reference line
        plt.plot([0, 1], [0, 1], color='#7f8c8d', linestyle=':', linewidth=1.2, label='Random Classifier')
        
        # Labeling and polish
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold', labelpad=10)
        plt.title('Diabetes Diagnostic Pipeline\nROC Curve Comparison', fontsize=14, fontweight='bold', pad=15)
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#bdc3c7', fontsize=10)
        
        # Clean background and layout
        plt.tight_layout()
        
        # Save output
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info("Successfully generated and saved ROC plot to: %s", save_path)


if __name__ == "__main__":
    dataset_path = "diabetes.csv"
    plot_output_path = "roc_comparison.png"
    
    # Run the eager pipeline
    pipeline = DiabetesPipeline(filepath=dataset_path)
    pipeline.train_and_evaluate()
    pipeline.plot_roc_curves(plot_output_path)
