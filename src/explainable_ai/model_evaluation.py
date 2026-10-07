# src/model_evaluation.py
"""
Model Evaluation Module
Unit I Concept: Model Evaluation Metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model Evaluation Implementation
    Unit I: Model Evaluation Metrics
    """
    
    def __init__(self):
        self.metrics = {}
        self.class_names = ['Non-Criminal', 'Criminal']
    
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None):
        """
        Comprehensive classification evaluation
        """
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Print results
        print("\n" + "="*70)
        print("CLASSIFICATION EVALUATION METRICS")
        print("="*70)
        print(f"Accuracy:               {metrics['accuracy']:.4f}")
        print(f"Precision (weighted):   {metrics['precision']:.4f}")
        print(f"Recall (weighted):      {metrics['recall']:.4f}")
        print(f"F1 Score (weighted):    {metrics['f1_score']:.4f}")
        
        # Classification Report
        print("\n" + "="*70)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion Matrix
        self.plot_confusion_matrix(metrics['confusion_matrix'])
        
        self.metrics = metrics
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix with heatmap
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   ax=ax,
                   annot_kws={'size': 14, 'weight': 'bold'})
        
        ax.set_title('Confusion Matrix - Criminal Face Recognition', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Confusion matrix saved to: {save_path}")
        
        plt.show()
