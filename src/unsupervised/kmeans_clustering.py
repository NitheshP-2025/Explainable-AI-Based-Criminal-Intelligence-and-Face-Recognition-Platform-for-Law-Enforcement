# src/unsupervised/kmeans_clustering.py
"""
K-Means Clustering Module
Unit IV Concept: K-Means Clustering
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class KMeansClustering:
    """
    K-Means Clustering Implementation for Criminal Grouping
    
    Groups criminals with similar facial characteristics
    """
    
    def __init__(self, n_clusters=4, random_state=42, max_iter=300):
        """
        Initialize K-Means clustering
        
        Args:
            n_clusters: Number of clusters (crime categories)
            random_state: Random seed
            max_iter: Maximum iterations
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.labels = None
        self.centroids = None
        self.inertia = None
        
        logger.info(f"KMeansClustering initialized with {n_clusters} clusters")
    
    def fit(self, X):
        """
        Fit K-Means clustering
        
        Args:
            X: Feature matrix (criminal face embeddings)
            
        Returns:
            Self
        """
        logger.info(f"Fitting K-Means with {self.n_clusters} clusters...")
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=10
        )
        
        self.labels = self.model.fit_predict(X)
        self.centroids = self.model.cluster_centers_
        self.inertia = self.model.inertia_
        
        logger.info(f"✅ K-Means clustering complete")
        logger.info(f"   Inertia: {self.inertia:.4f}")
        logger.info(f"   Cluster distribution: {np.bincount(self.labels)}")
        
        return self
    
    def predict(self, X):
        """Predict cluster for new samples"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def get_cluster_labels(self):
        """Get cluster labels for training data"""
        return self.labels
    
    def get_cluster_centers(self):
        """Get cluster centers"""
        return self.centroids
    
    def get_cluster_distribution(self):
        """Get distribution of samples across clusters"""
        return dict(zip(range(self.n_clusters), np.bincount(self.labels)))
    
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
            'calinski_harabasz_score': calinski_harabasz_score(X, self.labels),
            'davies_bouldin_score': davies_bouldin_score(X, self.labels),
            'inertia': self.inertia,
            'n_clusters': self.n_clusters
        }
        
        print("\n" + "="*60)
        print("📊 K-MEANS CLUSTERING EVALUATION")
        print("="*60)
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")
        print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
        print(f"Inertia: {metrics['inertia']:.4f}")
        print(f"Cluster Distribution: {self.get_cluster_distribution()}")
        
        return metrics
    
    def elbow_method(self, X, max_clusters=10):
        """
        Find optimal number of clusters using elbow method
        
        Args:
            X: Feature matrix
            max_clusters: Maximum clusters to test
            
        Returns:
            Dictionary of inertia values
        """
        inertias = []
        k_range = range(1, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, inertias, 'b-', linewidth=2)
        plt.plot(k_range, inertias, 'ro', markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True, alpha=0.3)
        
        # Annotate optimal k
        if len(inertias) > 2:
            # Find elbow point (maximum curvature)
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow = np.argmax(diffs2) + 2
            plt.axvline(x=elbow, color='r', linestyle='--', alpha=0.5)
            plt.text(elbow, max(inertias)*0.9, f'Optimal k={elbow}', color='red', ha='center')
        
        plt.tight_layout()
        plt.show()
        
        return {'k_range': list(k_range), 'inertias': inertias}
    
    def plot_clusters_2d(self, X, save_path=None):
        """
        Plot clusters in 2D using first two principal components
        
        Args:
            X: Feature matrix
            save_path: Optional save path
        """
        from sklearn.decomposition import PCA
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.labels, cmap='viridis', alpha=0.6)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'K-Means Clusters (k={self.n_clusters})')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Cluster plot saved to: {save_path}")
        
        plt.show()
