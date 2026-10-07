# src/unsupervised/pca_reduction.py
"""
PCA Dimensionality Reduction Module
Unit IV Concept: Principal Component Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class PCAReduction:
    """
    PCA Implementation for Dimensionality Reduction
    
    Reduces 16,384 features to lower dimensions
    """
    
    def __init__(self, n_components=100, random_state=42):
        """
        Initialize PCA
        
        Args:
            n_components: Number of components to keep
            random_state: Random seed
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.explained_variance_ratio_ = None
        self.components_ = None
        
        logger.info(f"PCAReduction initialized with {n_components} components")
    
    def fit(self, X):
        """
        Fit PCA on data
        
        Args:
            X: Feature matrix
            
        Returns:
            Self
        """
        logger.info(f"Fitting PCA with {self.n_components} components...")
        
        self.model = PCA(n_components=self.n_components, random_state=self.random_state)
        self.model.fit(X)
        
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        self.components_ = self.model.components_
        
        logger.info(f"✅ PCA fitting complete")
        logger.info(f"   Total explained variance: {np.sum(self.explained_variance_ratio_):.4f}")
        
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
        
        return self.model.transform(X)
    
    def fit_transform(self, X):
        """
        Fit and transform in one step
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Inverse transform to original space
        
        Args:
            X_transformed: Transformed data
            
        Returns:
            Original space data
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.inverse_transform(X_transformed)
    
    def get_explained_variance(self):
        """Get explained variance ratio per component"""
        return self.explained_variance_ratio_
    
    def get_components(self):
        """Get principal components"""
        return self.components_
    
    def plot_variance(self, save_path=None):
        """
        Plot explained variance ratio
        
        Args:
            save_path: Optional save path
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cumulative variance
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        
        # Individual variance
        axes[0].bar(range(1, len(self.explained_variance_ratio_) + 1), 
                    self.explained_variance_ratio_, alpha=0.7)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Individual Variance')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative variance
        axes[1].plot(range(1, len(cumulative_variance) + 1), 
                     cumulative_variance, 'b-', linewidth=2)
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Variance')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%')
        axes[1].legend()
        
        # Annotate 95% variance
        n_95 = np.argmax(cumulative_variance >= 0.95) + 1
        axes[1].axvline(x=n_95, color='g', linestyle='--', alpha=0.5)
        axes[1].text(n_95, 0.1, f'95% at {n_95} components', color='green')
        
        plt.suptitle(f'PCA Explained Variance (Total: {np.sum(self.explained_variance_ratio_):.4f})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Variance plot saved to: {save_path}")
        
        plt.show()
    
    def get_summary(self):
        """Get PCA summary"""
        if self.model is None:
            return {"status": "Not fitted"}
        
        summary = {
            'n_components': self.n_components,
            'total_explained_variance': np.sum(self.explained_variance_ratio_),
            'n_features_original': self.model.components_.shape[1],
            'n_components_kept': self.model.components_.shape[0]
        }
        
        print("\n" + "="*60)
        print("📊 PCA SUMMARY")
        print("="*60)
        print(f"Original Features: {summary['n_features_original']}")
        print(f"Reduced Components: {summary['n_components_kept']}")
        print(f"Total Explained Variance: {summary['total_explained_variance']:.4f}")
        print(f"95% Variance at: {np.argmax(np.cumsum(self.explained_variance_ratio_) >= 0.95) + 1} components")
        
        return summary
