# src/cross_validation.py
"""
Cross Validation Module
Unit I Concept: Cross Validation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Cross Validation Implementation for Criminal Face Recognition
    
    Implements:
    1. K-Fold Cross Validation
    2. Stratified K-Fold Cross Validation
    
    Unit I Concept: Cross Validation
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize cross validator
        
        Args:
            X: Feature matrix
            y: Labels
        """
        self.X = X
        self.y = y
        self.cv_results = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        
        logger.info(f"CrossValidator initialized with {len(X)} samples")
    
    def k_fold_cross_validation(self, model, n_splits: int = 5, shuffle: bool = True, 
                                random_state: int = 42, scoring: str = 'accuracy') -> np.ndarray:
        """
        K-Fold Cross Validation
        Splits data into k folds, trains on k-1 folds, tests on the remaining fold
        
        Args:
            model: Machine learning model
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed
            scoring: Scoring metric
            
        Returns:
            Array of scores for each fold
        """
        if shuffle:
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            kfold = KFold(n_splits=n_splits, shuffle=False)
        
        scores = cross_val_score(model, self.X, self.y, cv=kfold, scoring=scoring)
        
        print("\n" + "="*70)
        print(f"📊 K-FOLD CROSS VALIDATION ({n_splits} folds)")
        print("="*70)
        print(f"Model: {model.__class__.__name__}")
        print(f"Scoring Metric: {scoring}")
        print(f"\nIndividual Fold Scores:")
        for i, score in enumerate(scores, 1):
            print(f"  Fold {i}: {score:.4f}")
        print(f"\nSummary Statistics:")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std Dev: {scores.std():.4f}")
        print(f"  95% CI: [{scores.mean() - 1.96*scores.std():.4f}, {scores.mean() + 1.96*scores.std():.4f}]")
        print(f"  Min: {scores.min():.4f}")
        print(f"  Max: {scores.max():.4f}")
        
        return scores
    
    def stratified_k_fold(self, model, n_splits: int = 5, random_state: int = 42, 
                         scoring: str = 'accuracy') -> np.ndarray:
        """
        Stratified K-Fold Cross Validation
        Preserves class distribution in each fold
        
        Args:
            model: Machine learning model
            n_splits: Number of folds
            random_state: Random seed
            scoring: Scoring metric
            
        Returns:
            Array of scores for each fold
        """
        skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, self.X, self.y, cv=skfold, scoring=scoring)
        
        print("\n" + "="*70)
        print(f"📊 STRATIFIED K-FOLD CROSS VALIDATION ({n_splits} folds)")
        print("="*70)
        print(f"Model: {model.__class__.__name__}")
        print(f"Scoring Metric: {scoring}")
        print(f"\nIndividual Fold Scores:")
        for i, score in enumerate(scores, 1):
            print(f"  Fold {i}: {score:.4f}")
        print(f"\nSummary Statistics:")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std Dev: {scores.std():.4f}")
        print(f"  95% CI: [{scores.mean() - 1.96*scores.std():.4f}, {scores.mean() + 1.96*scores.std():.4f}]")
        
        return scores
    
    def compare_models(self, models: Dict[str, Any], cv_type: str = 'kfold', 
                      n_splits: int = 5, scoring: str = 'accuracy') -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models using cross validation
        
        Args:
            models: Dictionary of model name to model object
            cv_type: 'kfold' or 'stratified'
            n_splits: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary of results for each model
        """
        print("\n" + "="*70)
        print("🔍 MODEL COMPARISON USING CROSS VALIDATION")
        print("="*70)
        print(f"CV Type: {cv_type.upper()}")
        print(f"Number of Folds: {n_splits}")
        print(f"Scoring Metric: {scoring}")
        
        if cv_type == 'stratified':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            try:
                # Perform cross validation
                scores = cross_val_score(model, self.X, self.y, cv=cv, scoring=scoring)
                results[name] = {
                    'scores': scores,
                    'mean': scores.mean(),
                    'std': scores.std()
                }
                
                print(f"\n{name}:")
                print(f"  Mean Accuracy: {scores.mean():.4f}")
                print(f"  Std Deviation: {scores.std():.4f}")
                print(f"  All Scores: {scores}")
                
                # Track best model
                if scores.mean() > self.best_score:
                    self.best_score = scores.mean()
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"\n{name}: ❌ Error - {e}")
                results[name] = {'error': str(e)}
        
        # Print best model
        if self.best_model_name:
            print("\n" + "="*70)
            print(f"🏆 BEST MODEL: {self.best_model_name}")
            print(f"   Mean Accuracy: {self.best_score:.4f}")
            print("="*70)
        
        self.cv_results = results
        
        # Plot results
        self.plot_cv_comparison()
        
        return results
    
    def plot_cv_comparison(self, save_path: Optional[str] = None):
        """
        Plot cross validation results for comparison
        """
        if not self.cv_results:
            print("No CV results to plot")
            return
        
        # Filter out errors
        valid_results = {k: v for k, v in self.cv_results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = list(valid_results.keys())
        means = [valid_results[name]['mean'] for name in names]
        stds = [valid_results[name]['std'] for name in names]
        
        # Create bar plot with error bars
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        bars = ax.bar(names, means, yerr=stds, capsize=8, 
                      color=colors[:len(names)],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Cross Validation Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison using Cross Validation', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-labels if needed
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ CV comparison saved to: {save_path}")
        
        plt.show()
