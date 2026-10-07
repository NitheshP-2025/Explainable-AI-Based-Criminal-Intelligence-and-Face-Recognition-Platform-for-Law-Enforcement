# src/unsupervised/cluster_evaluation.py
"""
Cluster Evaluation Module
Unit IV Concept: Cluster Evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)


class ClusterEvaluation:
    """
    Cluster Evaluation Implementation
    
    Implements:
    1. Silhouette Score
    2. Calinski-Harabasz Score
    3. Davies-Bouldin Score
    4. Elbow Method
    """
    
    @staticmethod
    def silhouette_analysis(X, labels):
        """
        Calculate silhouette score
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Silhouette score
        """
        try:
            score = silhouette_score(X, labels)
            print(f"\nSilhouette Score: {score:.4f}")
            print("  - Range: -1 to 1")
            print("  - Higher is better")
            print("  - 0.5+ = Good clustering")
            print("  - 0.25-0.5 = Reasonable")
            print("  - <0.25 = Poor")
            return score
        except:
            print("⚠️ Could not calculate silhouette score (only one cluster?)")
            return None
    
    @staticmethod
    def calinski_harabasz(X, labels):
        """
        Calculate Calinski-Harabasz score
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Calinski-Harabasz score
        """
        try:
            score = calinski_harabasz_score(X, labels)
            print(f"\nCalinski-Harabasz Score: {score:.4f}")
            print("  - Higher is better")
            print("  - Ratio of between-cluster to within-cluster variance")
            return score
        except:
            print("⚠️ Could not calculate Calinski-Harabasz score")
            return None
    
    @staticmethod
    def davies_bouldin(X, labels):
        """
        Calculate Davies-Bouldin score
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Davies-Bouldin score
        """
        try:
            score = davies_bouldin_score(X, labels)
            print(f"\nDavies-Bouldin Score: {score:.4f}")
            print("  - Lower is better")
            print("  - 0 = Perfect clustering")
            print("  - <1 = Good clustering")
            return score
        except:
            print("⚠️ Could not calculate Davies-Bouldin score")
            return None
    
    @staticmethod
    def comprehensive_evaluation(X, labels):
        """
        Comprehensive cluster evaluation
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of all scores
        """
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE CLUSTER EVALUATION")
        print("="*60)
        
        results = {}
        
        # Silhouette Score
        silhouette = ClusterEvaluation.silhouette_analysis(X, labels)
        results['silhouette_score'] = silhouette
        
        # Calinski-Harabasz
        calinski = ClusterEvaluation.calinski_harabasz(X, labels)
        results['calinski_harabasz_score'] = calinski
        
        # Davies-Bouldin
        davies = ClusterEvaluation.davies_bouldin(X, labels)
        results['davies_bouldin_score'] = davies
        
        print("\n" + "="*60)
        print("📋 INTERPRETATION")
        print("="*60)
        
        # Overall assessment
        if silhouette is not None:
            if silhouette > 0.5:
                print("✅ Excellent clustering - Well separated clusters")
            elif silhouette > 0.25:
                print("✅ Good clustering - Reasonable cluster separation")
            elif silhouette > 0:
                print("⚠️ Fair clustering - Some overlap between clusters")
            else:
                print("❌ Poor clustering - Clusters are not well separated")
        
        return results
    
    @staticmethod
    def elbow_method(X, max_clusters=10):
        """
        Elbow method for finding optimal k
        
        Args:
            X: Feature matrix
            max_clusters: Maximum clusters to test
            
        Returns:
            Dictionary of results
        """
        from sklearn.cluster import KMeans
        
        inertias = []
        k_range = range(1, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
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
        
        # Highlight optimal k
        # Find elbow point (maximum curvature)
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow = np.argmax(diffs2) + 2
            plt.axvline(x=elbow, color='r', linestyle='--', alpha=0.5)
            plt.text(elbow, max(inertias)*0.9, f'Optimal k={elbow}', color='red', ha='center')
            print(f"📌 Suggested optimal k: {elbow}")
        
        plt.tight_layout()
        plt.show()
        
        return {'k_range': list(k_range), 'inertias': inertias}
