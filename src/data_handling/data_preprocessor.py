# src/data_handling/data_preprocessor.py
"""
Data Preprocessor Module
Combines data loading and feature scaling into a single pipeline
"""

import numpy as np
from .data_loader import CriminalFaceDataset
from .feature_scaling import FeatureScaler
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Complete data preprocessing pipeline
    """
    
    def __init__(self, data_path='data/criminal_faces', img_size=(128, 128)):
        self.data_path = data_path
        self.img_size = img_size
        self.dataset = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
    
    def load_and_split(self, test_size=0.2, random_state=42):
        """Load dataset and split into train/test"""
        self.dataset = CriminalFaceDataset(self.data_path, self.img_size)
        X, y = self.dataset.load_dataset()
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.split_train_test(
            test_size=test_size, random_state=random_state
        )
        logger.info(f"Data loaded and split: Train={len(self.X_train)}, Test={len(self.X_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def apply_scaling(self, method='standard'):
        """Apply feature scaling to train and test sets"""
        self.scaler = FeatureScaler()
        
        if method == 'standard':
            self.X_train_scaled, self.X_test_scaled = self.scaler.standard_scaling(
                self.X_train, self.X_test
            )
        elif method == 'minmax':
            self.X_train_scaled, self.X_test_scaled = self.scaler.minmax_scaling(
                self.X_train, self.X_test
            )
        elif method == 'robust':
            self.X_train_scaled, self.X_test_scaled = self.scaler.robust_scaling(
                self.X_train, self.X_test
            )
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        logger.info(f"Applied {method} scaling: Train={self.X_train_scaled.shape}, Test={self.X_test_scaled.shape}")
        return self.X_train_scaled, self.X_test_scaled
    
    def get_preprocessed_data(self):
        """Get fully preprocessed data"""
        return {
            'X_train': self.X_train_scaled,
            'X_test': self.X_test_scaled,
            'y_train': self.y_train,
            'y_test': self.y_test
        }
    
    def get_summary(self):
        """Get preprocessing summary"""
        if self.dataset:
            return self.dataset.get_data_summary()
        return None
