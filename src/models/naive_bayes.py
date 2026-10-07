# src/models/naive_bayes.py
"""
Naïve Bayes Classifier Module
Unit II Concept: Naïve Bayes Classifier
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class NaiveBayesModel:
    """
    Naïve Bayes Classifier Implementation
    
    Implements Gaussian Naïve Bayes for continuous features
    """
    
    def __init__(self):
        """Initialize Naïve Bayes classifier"""
        self.classes = None
        self.class_priors = {}
        self.mean = {}
        self.var = {}
        
        logger.info("NaiveBayesModel initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayesModel':
        """
        Train the Naïve Bayes classifier
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self
        """
        logger.info("Training Naïve Bayes classifier...")
        
        self.classes = np.unique(y)
        
        for class_label in self.classes:
            # Get samples for this class
            X_class = X[y == class_label]
            
            # Calculate prior probability
            self.class_priors[class_label] = len(X_class) / len(X)
            
            # Calculate mean and variance for each feature
            self.mean[class_label] = np.mean(X_class, axis=0)
            self.var[class_label] = np.var(X_class, axis=0) + 1e-9  # Add small value to avoid division by zero
        
        logger.info(f"✅ Naïve Bayes trained with {len(self.classes)} classes")
        
        return self
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """
        Calculate Gaussian probability density function
        
        Args:
            x: Feature values
            mean: Mean of distribution
            var: Variance of distribution
            
        Returns:
            Probability density
        """
        exponent = np.exp(-0.5 * ((x - mean) ** 2 / var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features
            
        Returns:
            Probability for each class
        """
        if self.classes is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, class_label in enumerate(self.classes):
            # Calculate posterior probability
            prior = np.log(self.class_priors[class_label])
            
            # Calculate likelihood using Gaussian PDF
            mean = self.mean[class_label]
            var = self.var[class_label]
            
            # Calculate log likelihood for all features
            log_likelihood = np.sum(
                -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X - mean) ** 2 / var),
                axis=1
            )
            
            # Posterior = prior + log_likelihood
            probabilities[:, i] = prior + log_likelihood
        
        # Normalize to get probabilities
        exp_probs = np.exp(probabilities - np.max(probabilities, axis=1, keepdims=True))
        probabilities = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance based on variance ratio
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of feature importance
        """
        importance = {}
        
        for class_label in self.classes:
            # Higher variance means less importance
            # Use inverse of variance as importance measure
            var = self.var[class_label]
            importance_class = 1 / (var + 1e-9)
            
            # Normalize
            importance_class = importance_class / np.sum(importance_class)
            
            if feature_names is not None:
                for i, name in enumerate(feature_names):
                    importance[f"{name}_class_{class_label}"] = importance_class[i]
            else:
                for i, val in enumerate(importance_class):
                    importance[f"feature_{i}_class_{class_label}"] = val
        
        return importance
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get model summary
        
        Returns:
            Dictionary with model parameters
        """
        if self.classes is None:
            return {"status": "Not trained"}
        
        summary = {
            'classes': self.classes.tolist(),
            'class_priors': self.class_priors,
            'mean_shape': self.mean[self.classes[0]].shape,
            'var_shape': self.var[self.classes[0]].shape
        }
        
        print("\n" + "="*50)
        print("NAÏVE BAYES SUMMARY")
        print("="*50)
        print(f"Classes: {self.classes}")
        print(f"Class Priors: {self.class_priors}")
        print(f"Feature Mean Shape: {self.mean[self.classes[0]].shape}")
        print(f"Feature Variance Shape: {self.var[self.classes[0]].shape}")
        print("="*50)
        
        return summary
