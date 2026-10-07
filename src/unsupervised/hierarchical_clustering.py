# src/unsupervised/hierarchical_clustering.py
"""
Hierarchical Clustering Module
Unit IV Concept: Hierarchical Clustering
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


class HierarchicalClustering:
    """
    Hierarchical Clustering Implementation
    
    Implements:
    1. Agglomerative Clustering
    2. Dendrogram Visualization
    3. Cluster Extraction
    """
    
    def __init__(self, method='ward', metric='euclidean'):
        """
        Initialize hierarchical clustering
        
        Args:
            method: Linkage method ('ward', 'complete', 'average', 'single')
            metric: Distance metric
        """
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
        self.labels = None
        self.n_clusters = None
        
        logger.info(f"HierarchicalClustering initialized with method={method}")
    
    def fit(self, X, n_clusters=None):
        """
        Fit hierarchical clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters (if None, use all)
            
        Returns:
            Self
        """
        logger.info(f"Fitting hierarchical clustering with method={self.method}...")
        
        # Compute linkage matrix
        self.linkage_matrix = linkage(X, method=self.method, metric=self.metric)
        
        if n_clusters is not None:
            self.n_clusters = n_clusters
            self.labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
            logger.info(f"✅ Extracted {n_clusters} clusters")
        
        logger.info("✅ Hierarchical clustering complete")
        
        return self
    
    def extract_clusters(self, n_clusters):
        """
        Extract clusters from linkage matrix
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels
        """
        self.n_clusters = n_clusters
        self.labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        return self.labels
    
    def get_cluster_distribution(self):
        """Get distribution of samples across clusters"""
        if self.labels is None:
            raise ValueError("No clusters extracted. Call extract_clusters() first.")
        
        return dict(zip(range(1, self.n_clusters + 1), 
                       np.bincount(self.labels)[1:]))
    
    def evaluate(self, X):
        """
        Evaluate clustering performance
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.labels is None:
            raise ValueError("No clusters extracted. Call extract_clusters() first.")
        
        metrics = {
            'silhouette_score': silhouette_score(X, self.labels),
            'n_clusters': self.n_clusters
        }
        
        print("\n" + "="*60)
        print("📊 HIERARCHICAL CLUSTERING EVALUATION")
        print("="*60)
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"Number of Clusters: {metrics['n_clusters']}")
        print(f"Cluster Distribution: {self.get_cluster_distribution()}")
        
        return metrics
    
    def plot_dendrogram(self, max_depth=None, save_path=None):
        """
        Plot dendrogram
        
        Args:
            max_depth: Maximum depth to display
            save_path: Optional save path
        """
        if self.linkage_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        plt.figure(figsize=(12, 8))
        
        dendrogram(
            self.linkage_matrix,
            truncate_mode='lastp' if max_depth else None,
            p=max_depth if max_depth else None,
            leaf_rotation=90,
            leaf_font_size=8
        )
        
        plt.title(f'Dendrogram ({self.method} linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Dendrogram saved to: {save_path}")
        
        plt.show()
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using silhouette score
        
        Args:
            X: Feature matrix
            max_clusters: Maximum clusters to test
            
        Returns:
            Dictionary of silhouette scores
        """
        # Compute linkage first if not done
        if self.linkage_matrix is None:
            self.fit(X)
        
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            labels = fcluster(self.linkage_matrix, k, criterion='maxclust')
            try:
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(-1)
        
        # Plot silhouette scores
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, silhouette_scores, 'b-', linewidth=2)
        plt.plot(k_range, silhouette_scores, 'ro', markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Optimal Clusters by Silhouette Score')
        plt.grid(True, alpha=0.3)
        
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
        plt.text(optimal_k, max(silhouette_scores)*0.9, 
                f'Optimal k={optimal_k}', color='red', ha='center')
        
        plt.tight_layout()
        plt.show()
        
        return {'k_range': list(k_range), 'silhouette_scores': silhouette_scores, 'optimal_k': optimal_k}
