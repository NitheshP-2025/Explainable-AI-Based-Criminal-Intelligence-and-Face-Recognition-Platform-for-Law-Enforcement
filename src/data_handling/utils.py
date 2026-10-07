# src/utils.py
"""
Utility Functions for Criminal Face Recognition System
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def save_model(model, path='models', name=None):
    """Save trained model to disk"""
    os.makedirs(path, exist_ok=True)
    
    if name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_{timestamp}.pkl"
    else:
        filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.pkl"
    
    filepath = os.path.join(path, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"✅ Model saved to: {filepath}")
    return filepath


def load_model(filepath):
    """Load saved model from disk"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"✅ Model loaded from: {filepath}")
    return model


def save_results(results, path='results', name=None):
    """Save results dictionary to JSON"""
    os.makedirs(path, exist_ok=True)
    
    if name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results_{timestamp}.json"
    else:
        filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.json"
    
    filepath = os.path.join(path, filename)
    
    # Convert numpy arrays to lists for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    serializable = {}
    for key, value in results.items():
        serializable[key] = convert(value)
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    logger.info(f"✅ Results saved to: {filepath}")
    return filepath


def plot_training_history(history, save_path=None):
    """Plot training history"""
    # Simplified version
    print("Training history plotting...")
    # Implementation here if needed


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot confusion matrix"""
    # Simplified version
    print("Confusion matrix plotting...")


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot ROC curve"""
    print("ROC curve plotting...")


def generate_performance_report(metrics, save_path=None):
    """Generate performance report"""
    report = []
    report.append("="*70)
    report.append("📊 CRIMINAL FACE RECOGNITION - PERFORMANCE REPORT")
    report.append("="*70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*70)
    
    if 'accuracy' in metrics:
        report.append(f"\nAccuracy: {metrics['accuracy']:.2%}")
    if 'precision' in metrics:
        report.append(f"Precision: {metrics['precision']:.2%}")
    if 'recall' in metrics:
        report.append(f"Recall: {metrics['recall']:.2%}")
    if 'f1_score' in metrics:
        report.append(f"F1-Score: {metrics['f1_score']:.2%}")
    
    report_str = "\n".join(report)
    print(report_str)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report_str)
        logger.info(f"✅ Report saved to: {save_path}")
    
    return report_str
