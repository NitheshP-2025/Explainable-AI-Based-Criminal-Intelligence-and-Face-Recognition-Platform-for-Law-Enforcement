# src/feature_scaling.py (Updated Version)

"""
Feature Scaling Module
Unit I Concept: Feature Scaling and Normalization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Feature Scaling Implementation for Face Recognition
    Implements multiple scaling techniques
    """
    
    def __init__(self):
        self.scalers = {}
        self.scaled_data = {}
        self.scaler_objects = {}
    
    def standard_scaling(self, X_train, X_test=None):
        """Standard Scaling (Z-score normalization)"""
        logger.info("Applying Standard Scaling...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scaler_objects['standard'] = scaler
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            self.scaled_data['standard'] = (X_train_scaled, X_test_scaled)
            return X_train_scaled, X_test_scaled
        
        self.scaled_data['standard'] = X_train_scaled
        return X_train_scaled
    
    def minmax_scaling(self, X_train, X_test=None):
        """Min-Max Scaling"""
        logger.info("Applying Min-Max Scaling...")
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scaler_objects['minmax'] = scaler
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            self.scaled_data['minmax'] = (X_train_scaled, X_test_scaled)
            return X_train_scaled, X_test_scaled
        
        self.scaled_data['minmax'] = X_train_scaled
        return X_train_scaled
    
    def robust_scaling(self, X_train, X_test=None):
        """Robust Scaling (median & IQR)"""
        logger.info("Applying Robust Scaling...")
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scaler_objects['robust'] = scaler
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            self.scaled_data['robust'] = (X_train_scaled, X_test_scaled)
            return X_train_scaled, X_test_scaled
        
        self.scaled_data['robust'] = X_train_scaled
        return X_train_scaled
    
    def compare_scaling_effects(self, X_original, save_path=None):
        """Compare different scaling techniques"""
        # Create directory if it doesn't exist
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Sample data for visualization
        data_to_plot = X_original.flatten()[:5000]
        data_2d = data_to_plot.reshape(-1, 1)
        
        # Original
        axes[0, 0].hist(data_to_plot, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Original Data', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard
        X_standard = StandardScaler().fit_transform(data_2d).flatten()
        axes[0, 1].hist(X_standard, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Standard Scaling\n(Mean=0, Std=1)', fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Min-Max
        X_minmax = MinMaxScaler().fit_transform(data_2d).flatten()
        axes[1, 0].hist(X_minmax, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_title('Min-Max Scaling\n(Range [0,1])', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Robust
        X_robust = RobustScaler().fit_transform(data_2d).flatten()
        axes[1, 1].hist(X_robust, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('Robust Scaling\n(Median & IQR)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Comparison of Feature Scaling Techniques for Face Recognition', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ Scaling comparison saved to: {save_path}")
            except Exception as e:
                logger.warning(f"Could not save figure: {e}")
                print(f"⚠️ Could not save figure to {save_path}, but showing it anyway...")
        
        plt.show()
        
        # Print statistics
        print("\n📊 Scaling Statistics:")
        print(f"  Original - Mean: {np.mean(data_to_plot):.4f}, Std: {np.std(data_to_plot):.4f}")
        print(f"  Standard - Mean: {np.mean(X_standard):.4f}, Std: {np.std(X_standard):.4f}")
        print(f"  MinMax   - Mean: {np.mean(X_minmax):.4f}, Std: {np.std(X_minmax):.4f}")
        print(f"  Robust   - Mean: {np.mean(X_robust):.4f}, Std: {np.std(X_robust):.4f}")
        
        return {'standard': X_standard, 'minmax': X_minmax, 'robust': X_robust}
    
    def get_scaling_recommendation(self, X):
        """Get recommendation for best scaling technique"""
        print("\n" + "="*60)
        print("🔍 SCALING TECHNIQUE RECOMMENDATIONS")
        print("="*60)
        
        # Check for outliers
        q1 = np.percentile(X, 25)
        q3 = np.percentile(X, 75)
        iqr = q3 - q1
        
        print("\n📊 Data Characteristics:")
        print(f"  Mean: {np.mean(X):.4f}")
        print(f"  Std Dev: {np.std(X):.4f}")
        print(f"  Range: [{np.min(X):.4f}, {np.max(X):.4f}]")
        print(f"  IQR: {iqr:.4f}")
        
        print("\n💡 Recommended Techniques:")
        print("  1. Standard Scaling - For normally distributed data")
        print("  2. Min-Max Scaling - For data with known bounds")
        print("  3. Robust Scaling - For data with outliers")
        
        return {
            'data_characteristics': {
                'mean': np.mean(X),
                'std': np.std(X),
                'min': np.min(X),
                'max': np.max(X),
                'iqr': iqr
            }
        }
