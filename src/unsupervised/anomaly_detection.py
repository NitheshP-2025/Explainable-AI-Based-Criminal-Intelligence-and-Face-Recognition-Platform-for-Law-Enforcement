# src/unsupervised/anomaly_detection.py
"""
Anomaly Detection Module
Unit IV Concept: Anomaly Detection Techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, accuracy_score
import logging

logger = logging.getLogger(__name__)


class AnomalyDetection:
    """
    Anomaly Detection Implementation
    
    Implements:
    1. Isolation Forest
    2. Local Outlier Factor
    3. One-Class SVM
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize anomaly detection
        
        Args:
            contamination: Expected proportion of outliers
            random_state: Random seed
        """
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        logger.info(f"AnomalyDetection initialized with contamination={contamination}")
    
    def isolation_forest(self, X, random_state=42):
        """
        Isolation Forest for anomaly detection
        
        Args:
            X: Feature matrix
            random_state: Random seed
            
        Returns:
            Trained model
        """
        logger.info("Training Isolation Forest...")
        
        model = IsolationForest(
            contamination=self.contamination,
            random_state=random_state,
            n_estimators=100
        )
        
        model.fit(X)
        self.models['isolation_forest'] = model
        
        logger.info("✅ Isolation Forest trained")
        
        return model
    
    def local_outlier_factor(self, X):
        """
        Local Outlier Factor for anomaly detection
        
        Args:
            X: Feature matrix
            
        Returns:
            Trained model
        """
        logger.info("Training Local Outlier Factor...")
        
        model = LocalOutlierFactor(
            contamination=self.contamination,
            novelty=True
        )
        
        model.fit(X)
        self.models['local_outlier_factor'] = model
        
        logger.info("✅ Local Outlier Factor trained")
        
        return model
    
    def one_class_svm(self, X, nu=0.1):
        """
        One-Class SVM for anomaly detection
        
        Args:
            X: Feature matrix
            nu: Upper bound on fraction of training errors
            
        Returns:
            Trained model
        """
        logger.info("Training One-Class SVM...")
        
        model = OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='auto'
        )
        
        model.fit(X)
        self.models['one_class_svm'] = model
        
        logger.info("✅ One-Class SVM trained")
        
        return model
    
    def detect_anomalies(self, X, method='isolation_forest'):
        """
        Detect anomalies using specified method
        
        Args:
            X: Feature matrix
            method: 'isolation_forest', 'local_outlier_factor', 'one_class_svm'
            
        Returns:
            Predictions (-1 for anomaly, 1 for normal)
        """
        if method not in self.models:
            raise ValueError(f"Method {method} not trained. Call train method first.")
        
        model = self.models[method]
        
        # Local Outlier Factor uses negative values for anomalies
        if method == 'local_outlier_factor':
            predictions = model.predict(X)
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def compare_methods(self, X):
        """
        Compare all anomaly detection methods
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of results
        """
        print("\n" + "="*60)
        print("🔍 ANOMALY DETECTION COMPARISON")
        print("="*60)
        
        results = {}
        
        # 1. Isolation Forest
        print("\n1. Isolation Forest...")
        self.isolation_forest(X)
        pred1 = self.detect_anomalies(X, 'isolation_forest')
        anomaly_count1 = np.sum(pred1 == -1)
        results['Isolation Forest'] = {
            'anomalies': anomaly_count1,
            'anomaly_percentage': anomaly_count1 / len(X) * 100
        }
        print(f"   Anomalies: {anomaly_count1} ({anomaly_count1/len(X)*100:.2f}%)")
        
        # 2. Local Outlier Factor
        print("\n2. Local Outlier Factor...")
        self.local_outlier_factor(X)
        pred2 = self.detect_anomalies(X, 'local_outlier_factor')
        anomaly_count2 = np.sum(pred2 == -1)
        results['Local Outlier Factor'] = {
            'anomalies': anomaly_count2,
            'anomaly_percentage': anomaly_count2 / len(X) * 100
        }
        print(f"   Anomalies: {anomaly_count2} ({anomaly_count2/len(X)*100:.2f}%)")
        
        # 3. One-Class SVM
        print("\n3. One-Class SVM...")
        self.one_class_svm(X)
        pred3 = self.detect_anomalies(X, 'one_class_svm')
        anomaly_count3 = np.sum(pred3 == -1)
        results['One-Class SVM'] = {
            'anomalies': anomaly_count3,
            'anomaly_percentage': anomaly_count3 / len(X) * 100
        }
        print(f"   Anomalies: {anomaly_count3} ({anomaly_count3/len(X)*100:.2f}%)")
        
        # Find best method (closest to expected contamination)
        expected_count = int(len(X) * self.contamination)
        print("\n" + "="*60)
        print("🏆 BEST METHOD")
        print("="*60)
        print(f"Expected Anomalies ({self.contamination:.1%}): {expected_count}")
        print("\nMethod Performance:")
        for method, result in results.items():
            diff = abs(result['anomalies'] - expected_count)
            print(f"  {method}: {result['anomalies']} anomalies (diff: {diff})")
        
        best_method = min(results.keys(), key=lambda k: abs(results[k]['anomalies'] - expected_count))
        print(f"\nBest Method: {best_method}")
        
        self.best_model_name = best_method
        self.best_model = self.models[best_method]
        
        return results
    
    def plot_anomalies(self, X, y=None, save_path=None):
        """
        Visualize anomalies in 2D
        
        Args:
            X: Feature matrix
            y: True labels (optional)
            save_path: Optional save path
        """
        from sklearn.decomposition import PCA
        
        # Reduce to 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        if self.best_model is None:
            print("No model selected. Run compare_methods() first.")
            return
        
        # Get predictions
        predictions = self.best_model.predict(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot with anomaly coloring
        normal = predictions == 1
        anomaly = predictions == -1
        
        axes[0].scatter(X_2d[normal, 0], X_2d[normal, 1], 
                       c='blue', alpha=0.6, label='Normal')
        axes[0].scatter(X_2d[anomaly, 0], X_2d[anomaly, 1], 
                       c='red', alpha=0.8, label='Anomaly', s=100)
        axes[0].set_title(f'Anomalies Detected ({self.best_model_name})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # If true labels available, show comparison
        if y is not None:
            axes[1].scatter(X_2d[y==0, 0], X_2d[y==0, 1], 
                           c='blue', alpha=0.6, label='Class 0')
            axes[1].scatter(X_2d[y==1, 0], X_2d[y==1, 1], 
                           c='green', alpha=0.6, label='Class 1')
            axes[1].set_title('True Labels (for reference)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Anomaly Detection: {self.best_model_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Anomaly plot saved to: {save_path}")
        
        plt.show()
