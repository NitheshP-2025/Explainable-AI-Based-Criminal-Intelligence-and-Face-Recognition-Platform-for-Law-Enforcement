# src/models/perceptron.py
"""
Perceptron Algorithm Module
Unit II Concept: Perceptron Algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class PerceptronModel:
    """
    Perceptron Algorithm Implementation
    
    The Perceptron is a simple binary classifier that learns a linear decision boundary
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 random_state: int = 42):
        """
        Initialize Perceptron
        
        Args:
            learning_rate: Learning rate for weight updates
            max_iter: Maximum number of training iterations
            random_state: Random seed for weight initialization
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors = []
        self.classes = None
        
        np.random.seed(random_state)
        logger.info("PerceptronModel initialized")
    
    def _initialize_weights(self, n_features: int) -> None:
        """Initialize weights randomly"""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    def _unit_step(self, z: np.ndarray) -> np.ndarray:
        """Unit step activation function"""
        return np.where(z >= 0, 1, 0)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PerceptronModel':
        """
        Train the Perceptron using the perceptron learning algorithm
        
        Args:
            X: Training features
            y: Training labels (must be 0 or 1)
            
        Returns:
            Self
        """
        logger.info("Training Perceptron...")
        
        # Convert labels to -1 and 1 for Perceptron
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("Perceptron is a binary classifier. Only 2 classes allowed.")
        
        # Convert to -1, 1
        y_transformed = np.where(y == self.classes[0], -1, 1)
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        self.errors = []
        
        for iteration in range(self.max_iter):
            n_errors = 0
            
            # Shuffle training data for better convergence
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_transformed[indices]
            
            for i in range(n_samples):
                # Forward pass
                z = np.dot(X_shuffled[i], self.weights) + self.bias
                prediction = self._unit_step(z)
                
                # Convert prediction to -1, 1
                prediction_transformed = 1 if prediction == 1 else -1
                
                # Check if prediction is correct
                if prediction_transformed != y_shuffled[i]:
                    # Update weights and bias
                    update = self.learning_rate * y_shuffled[i]
                    self.weights += update * X_shuffled[i]
                    self.bias += update
                    n_errors += 1
            
            # Track errors
            self.errors.append(n_errors)
            
            # Early stopping if no errors
            if n_errors == 0:
                logger.info(f"Converged at iteration {iteration + 1}")
                break
            
            if (iteration + 1) % 100 == 0:
                logger.info(f"Iteration {iteration + 1}: {n_errors} errors")
        
        logger.info(f"✅ Perceptron trained: {len(self.errors)} iterations, Final errors: {self.errors[-1]}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        z = np.dot(X, self.weights) + self.bias
        predictions = self._unit_step(z)
        
        # Map back to original class labels
        return np.where(predictions == 1, self.classes[1], self.classes[0])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (using sigmoid approximation)
        
        Args:
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        z = np.dot(X, self.weights) + self.bias
        # Use sigmoid to approximate probabilities
        probabilities = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        return np.column_stack([1 - probabilities, probabilities])
    
    def plot_errors(self, save_path: Optional[str] = None) -> None:
        """
        Plot training errors over iterations
        
        Args:
            save_path: Optional path to save the figure
        """
        if not self.errors:
            print("No errors to plot. Train the model first.")
            return
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.errors) + 1), self.errors, 'b-', linewidth=2)
        plt.xlabel('Iterations')
        plt.ylabel('Number of Misclassifications')
        plt.title('Perceptron Training Progress')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Error plot saved to: {save_path}")
        
        plt.show()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get model summary
        
        Returns:
            Dictionary with model parameters
        """
        if self.weights is None:
            return {"status": "Not trained"}
        
        summary = {
            'classes': self.classes.tolist(),
            'n_features': len(self.weights),
            'n_iterations': len(self.errors),
            'final_errors': self.errors[-1] if self.errors else 0,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter
        }
        
        print("\n" + "="*50)
        print("PERCEPTRON SUMMARY")
        print("="*50)
        print(f"Classes: {self.classes}")
        print(f"Features: {summary['n_features']}")
        print(f"Training Iterations: {summary['n_iterations']}")
        print(f"Final Misclassifications: {summary['final_errors']}")
        print(f"Learning Rate: {self.learning_rate}")
        print("="*50)
        
        return summary
