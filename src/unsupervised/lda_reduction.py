# src/unsupervised/lda_reduction.py
"""
LDA (Linear Discriminant Analysis) Module
Unit IV Concept: Linear Discriminant Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class LDAReduction:
    """
    LDA Implementation for Dimensionality Reduction
    
    Implements:
    1. LDA for Supervised Dimensionality Reduction
    2. Visualization of LDA Components
    """
    
    def __init__(self, n_components=2, random_state=42):
        """
        Initialize LDA
        
        Args:
            n_components: Number of components to keep (max = n_classes - 1)
            random_state: Random seed
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.explained_variance_ratio_ = None
        self.scaler = StandardScaler()
        
        logger.info(f"LDAReduction initialized with {n_components} components")
    
    def fit(self, X, y):
        """
        Fit LDA on data
        
        Args:
            X: Feature matrix
            y: Target labels (required for supervised LDA)
            
        Returns:
            Self
        """
        logger.info(f"Fitting LDA with {self.n_components} components...")
        
        # Scale data first
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = SklearnLDA(
            n_components=self.n_components
        )
        
        self.model.fit(X_scaled, y)
        
        # Get explained variance ratio
        if hasattr(self.model, 'explained_variance_ratio_'):
            self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        
        logger.info(f"✅ LDA fitting complete")
        logger.info(f"   Components: {self.model.components_.shape[0]}")
        logger.info(f"   Features reduced from {X.shape[1]} to {self.n_components}")
        
        return self
    
    def transform(self, X):
        """
        Transform data to lower dimension
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed data
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.transform(X_scaled)
    
    def fit_transform(self, X, y):
        """
        Fit and transform in one step
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Transformed data
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_components(self):
        """Get LDA components"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.components_
    
    def plot_components(self, X, y, save_path=None):
        """
        Plot data in LDA space (2D)
        
        Args:
            X: Feature matrix
            y: Target labels
            save_path: Optional save path
        """
        if self.n_components < 2:
            print("Cannot plot 1D data. Use n_components=2.")
            return
        
        # Transform data
        X_lda = self.transform(X)
        
        # Plot
        plt.figure(figsize=(10, 8))
        classes = np.unique(y)
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, class_label in enumerate(classes):
            mask = y == class_label
            plt.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                       color=colors[i % len(colors)], 
                       label=f'Class {class_label}', alpha=0.6)
        
        plt.xlabel('LDA Component 1')
        plt.ylabel('LDA Component 2')
        plt.title('LDA Projection (2D)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ LDA plot saved to: {save_path}")
        
        plt.show()
    
    def get_summary(self):
        """Get LDA summary"""
        if self.model is None:
            return {"status": "Not fitted"}
        
        summary = {
            'n_components': self.n_components,
            'n_features_original': self.model.components_.shape[1],
            'n_components_kept': self.model.components_.shape[0],
            'classes': self.model.classes_
        }
        
        print("\n" + "="*60)
        print("📊 LDA SUMMARY")
        print("="*60)
        print(f"Original Features: {summary['n_features_original']}")
        print(f"Reduced Components: {summary['n_components_kept']}")
        print(f"Classes: {summary['classes']}")
        
        return summary
