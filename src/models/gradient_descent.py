# src/models/gradient_descent.py
"""
Gradient Descent Module
Unit II Concept: Gradient Descent
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class GradientDescent:
    """
    Gradient Descent Implementation
    
    Implements:
    1. Batch Gradient Descent
    2. Stochastic Gradient Descent
    3. Mini-Batch Gradient Descent
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 tolerance: float = 1e-6, random_state: int = 42):
        """
        Initialize gradient descent
        
        Args:
            learning_rate: Learning rate for updates
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            random_state: Random seed
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.X = None
        self.y = None
        
        np.random.seed(random_state)
        logger.info("GradientDescent initialized")
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def _initialize_weights(self, n_features: int) -> None:
        """Initialize weights randomly"""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    def _compute_gradients(self, X: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute gradients for weights and bias"""
        y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)
        error = y_pred - y_true
        d_weights = np.dot(X.T, error) / len(X)
        d_bias = np.mean(error)
        return d_weights, d_bias
    
    def batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Batch Gradient Descent - Uses entire dataset for each update
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary with training results
        """
        logger.info("Running Batch Gradient Descent...")
        self.X = X
        self.y = y
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        self.loss_history = []
        
        for iteration in range(self.max_iter):
            # Forward pass
            y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)
            
            # Compute loss
            loss = self._compute_loss(y_pred, y)
            self.loss_history.append(loss)
            
            # Compute gradients
            d_weights, d_bias = self._compute_gradients(X, y)
            
            # Update parameters
            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias
            
            # Check convergence
            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                logger.info(f"Converged at iteration {iteration}")
                break
        
        logger.info(f"Batch GD completed: {len(self.loss_history)} iterations, Final loss: {self.loss_history[-1]:.6f}")
        
        return {
            'weights': self.weights,
            'bias': self.bias,
            'loss_history': self.loss_history,
            'iterations': len(self.loss_history),
            'final_loss': self.loss_history[-1],
            'method': 'Batch'
        }
    
    def stochastic_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Stochastic Gradient Descent - Uses one sample for each update
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary with training results
        """
        logger.info("Running Stochastic Gradient Descent...")
        self.X = X
        self.y = y
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        self.loss_history = []
        
        for iteration in range(self.max_iter):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            
            for i in range(n_samples):
                # Single sample
                x_i = X_shuffled[i:i+1]
                y_i = y_shuffled[i:i+1]
                
                # Forward pass
                y_pred = self._sigmoid(np.dot(x_i, self.weights) + self.bias)
                
                # Compute gradients
                error = y_pred - y_i
                d_weights = np.dot(x_i.T, error)
                d_bias = error[0]
                
                # Update parameters
                self.weights -= self.learning_rate * d_weights.flatten()
                self.bias -= self.learning_rate * d_bias
            
            # Compute epoch loss
            y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)
            loss = self._compute_loss(y_pred, y)
            self.loss_history.append(loss)
            
            # Check convergence
            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                logger.info(f"Converged at iteration {iteration}")
                break
        
        logger.info(f"SGD completed: {len(self.loss_history)} iterations, Final loss: {self.loss_history[-1]:.6f}")
        
        return {
            'weights': self.weights,
            'bias': self.bias,
            'loss_history': self.loss_history,
            'iterations': len(self.loss_history),
            'final_loss': self.loss_history[-1],
            'method': 'Stochastic'
        }
    
    def mini_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray, 
                                   batch_size: int = 32) -> Dict[str, Any]:
        """
        Mini-Batch Gradient Descent - Uses batch_size samples for each update
        
        Args:
            X: Features
            y: Labels
            batch_size: Number of samples per batch
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Running Mini-Batch Gradient Descent (batch_size={batch_size})...")
        self.X = X
        self.y = y
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        self.loss_history = []
        
        for iteration in range(self.max_iter):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self._sigmoid(np.dot(X_batch, self.weights) + self.bias)
                
                # Compute gradients
                error = y_pred - y_batch
                d_weights = np.dot(X_batch.T, error) / len(X_batch)
                d_bias = np.mean(error)
                
                # Update parameters
                self.weights -= self.learning_rate * d_weights
                self.bias -= self.learning_rate * d_bias
            
            # Compute epoch loss
            y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)
            loss = self._compute_loss(y_pred, y)
            self.loss_history.append(loss)
            
            # Check convergence
            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                logger.info(f"Converged at iteration {iteration}")
                break
        
        logger.info(f"Mini-Batch GD completed: {len(self.loss_history)} iterations, Final loss: {self.loss_history[-1]:.6f}")
        
        return {
            'weights': self.weights,
            'bias': self.bias,
            'loss_history': self.loss_history,
            'iterations': len(self.loss_history),
            'final_loss': self.loss_history[-1],
            'method': 'Mini-Batch',
            'batch_size': batch_size
        }
    
    def compare_optimizers(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compare all three gradient descent variants
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "="*70)
        print("📉 GRADIENT DESCENT COMPARISON")
        print("="*70)
        
        results = {}
        
        # Batch GD
        print("\n1. Running Batch Gradient Descent...")
        batch_results = self.batch_gradient_descent(X, y)
        results['Batch'] = batch_results
        print(f"   Iterations: {batch_results['iterations']}")
        print(f"   Final Loss: {batch_results['final_loss']:.6f}")
        
        # SGD
        print("\n2. Running Stochastic Gradient Descent...")
        sgd_results = self.stochastic_gradient_descent(X, y)
        results['Stochastic'] = sgd_results
        print(f"   Iterations: {sgd_results['iterations']}")
        print(f"   Final Loss: {sgd_results['final_loss']:.6f}")
        
        # Mini-Batch GD
        print("\n3. Running Mini-Batch Gradient Descent...")
        mb_results = self.mini_batch_gradient_descent(X, y)
        results['Mini-Batch'] = mb_results
        print(f"   Iterations: {mb_results['iterations']}")
        print(f"   Final Loss: {mb_results['final_loss']:.6f}")
        
        # Find best
        best_method = min(results.keys(), key=lambda k: results[k]['final_loss'])
        print("\n" + "="*70)
        print(f"🏆 BEST METHOD: {best_method}")
        print(f"   Final Loss: {results[best_method]['final_loss']:.6f}")
        print("="*70)
        
        # Plot comparison
        self._plot_loss_comparison(results)
        
        return results
    
    def _plot_loss_comparison(self, results: Dict[str, Any]) -> None:
        """Plot loss history for all methods"""
        plt.figure(figsize=(10, 6))
        
        for method, result in results.items():
            plt.plot(result['loss_history'], label=method)
        
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Gradient Descent Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            X: Features
            
        Returns:
            Predictions (0 or 1)
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call one of the GD methods first.")
        
        probabilities = self._sigmoid(np.dot(X, self.weights) + self.bias)
        return (probabilities >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call one of the GD methods first.")
        
        probabilities = self._sigmoid(np.dot(X, self.weights) + self.bias)
        return np.column_stack([1 - probabilities, probabilities])
