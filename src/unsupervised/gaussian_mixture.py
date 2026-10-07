# src/unsupervised/gaussian_mixture.py
"""
Gaussian Mixture Models Module
Unit IV Concept: Gaussian Mixture Models
"""

import numpy as np
from sklearn.mixture import GaussianMixture as SklearnGMM
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class GaussianMixtureModel:
    """
    Gaussian Mixture Model Implementation
    
    Implements:
    1. GMM Clustering
    2. Component Selection
    3. Evaluation
    """
    
    def __init__(self, n_components=4, random_state=42, max_iter=200):
        """
        Initialize Gaussian Mixture Model
        
        Args:
            n_components: Number of components (clusters)
            random_state: Random seed
            max_iter: Maximum iterations
        """
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.labels = None
        self.means = None
        self.covariances = None
        self.weights = None
        
        logger.info(f"GaussianMixtureModel initialized with {n_components} components")
    
    def fit(self, X):
        """
        Fit GMM clustering
        
        Args:
            X: Feature matrix
            
        Returns:
            Self
        """
        logger.info(f"Fitting GMM with {self.n_components} components...")
        
        self.model = SklearnGMM(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        
        self.labels = self.model.fit_predict(X)
        self.means = self.model.means_
        self.covariances = self.model.covariances_
        self.weights = self.model.weights_
        
        logger.info(f"✅ GMM clustering complete")
        logger.info(f"   Cluster distribution: {np.bincount(self.labels)}")
        
        return self
    
    def predict(self, X):
        """Predict cluster for new samples"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability of belonging to each cluster"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_cluster_distribution(self):
        """Get distribution of samples across clusters"""
        if self.labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return dict(zip(range(self.n_components), np.bincount(self.labels)))
    
    def evaluate(self, X):
        """
        Evaluate clustering performance
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        metrics = {
            'silhouette_score': silhouette_score(X, self.labels),
            'n_components': self.n_components,
            'bic': self.model.bic(X),
            'aic': self.model.aic(X)
        }
        
        print("\n" + "="*60)
        print("📊 GAUSSIAN MIXTURE MODEL EVALUATION")
        print("="*60)
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"BIC: {metrics['bic']:.4f}")
        print(f"AIC: {metrics['aic']:.4f}")
        print(f"Cluster Distribution: {self.get_cluster_distribution()}")
        
        return metrics
    
    def find_optimal_components(self, X, max_components=10):
        """
        Find optimal number of components using BIC/AIC
        
        Args:
            X: Feature matrix
            max_components: Maximum components to test
            
        Returns:
            Dictionary of results
        """
        bic_scores = []
        aic_scores = []
        k_range = range(1, max_components + 1)
        
        for k in k_range:
            gmm = SklearnGMM(n_components=k, random_state=self.random_state)
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, bic_scores, 'b-', label='BIC', linewidth=2)
        plt.plot(k_range, aic_scores, 'r-', label='AIC', linewidth=2)
        plt.xlabel('Number of Components')
        plt.ylabel('Score')
        plt.title('Optimal Components Selection (BIC/AIC)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Find optimal
        optimal_bic = k_range[np.argmin(bic_scores)]
        optimal_aic = k_range[np.argmin(aic_scores)]
        
        plt.axvline(x=optimal_bic, color='b', linestyle='--', alpha=0.5)
        plt.axvline(x=optimal_aic, color='r', linestyle='--', alpha=0.5)
        plt.text(optimal_bic, max(bic_scores)*0.9, f'Optimal BIC={optimal_bic}', color='blue')
        plt.text(optimal_aic, max(bic_scores)*0.8, f'Optimal AIC={optimal_aic}', color='red')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'k_range': list(k_range),
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'optimal_bic': optimal_bic,
            'optimal_aic': optimal_aic
        }
