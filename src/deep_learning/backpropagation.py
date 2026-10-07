# src/deep_learning/backpropagation.py
"""
Backpropagation Algorithm Module
Unit III Concept: Backpropagation Algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class Backpropagation:
    """
    Backpropagation Algorithm Implementation
    
    Implements:
    1. Forward Pass
    2. Backward Pass (Error Propagation)
    3. Weight Updates
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, activation='sigmoid'):
        """
        Initialize neural network with backpropagation
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            learning_rate: Learning rate for weight updates
            activation: Activation function to use
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        
        self.weights = []
        self.biases = []
        self.loss_history = []
        
        # Initialize weights and biases
        self._initialize_weights()
        
        logger.info(f"Backpropagation initialized with layers: {layer_sizes}")
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/He initialization"""
        np.random.seed(42)
        
        for i in range(len(self.layer_sizes) - 1):
            # Xavier initialization for better convergence
            limit = np.sqrt(6 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit, (self.layer_sizes[i], self.layer_sizes[i + 1]))
            b = np.zeros((1, self.layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _activation(self, x):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            return x
    
    def _activation_derivative(self, x):
        """Derivative of activation function"""
        if self.activation == 'sigmoid':
            s = self._activation(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        else:
            return 1
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Args:
            X: Input data
            
        Returns:
            Predictions and layer outputs
        """
        self.layer_outputs = [X]
        current_input = X
        
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            current_input = self._activation(z)
            self.layer_outputs.append(current_input)
        
        return current_input
    
    def backward(self, y_true):
        """
        Backward pass (backpropagation)
        
        Args:
            y_true: True labels
        """
        n_samples = y_true.shape[0]
        
        # Convert labels to one-hot if needed
        if len(y_true.shape) == 1:
            y_one_hot = np.zeros((n_samples, self.layer_sizes[-1]))
            y_one_hot[np.arange(n_samples), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        # Calculate output layer error
        output = self.layer_outputs[-1]
        error = output - y_true
        
        # Backpropagate through layers
        deltas = [error]
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta = deltas[-1]
            weight = self.weights[i]
            activation_deriv = self._activation_derivative(self.layer_outputs[i])
            
            error = np.dot(delta, weight.T) * activation_deriv
            deltas.append(error)
        
        deltas = deltas[::-1]
        
        # Update weights and biases
        for i in range(len(self.weights)):
            layer_input = self.layer_outputs[i]
            delta = deltas[i]
            
            # Compute gradients
            dW = np.dot(layer_input.T, delta) / n_samples
            db = np.mean(delta, axis=0, keepdims=True)
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
    
    def fit(self, X, y, epochs=100, verbose=True):
        """
        Train the neural network using backpropagation
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            verbose: Print progress
        """
        logger.info(f"Training neural network for {epochs} epochs...")
        
        self.loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Compute loss
            loss = np.mean((predictions - y) ** 2)
            self.loss_history.append(loss)
            
            # Backward pass
            self.backward(y)
            
            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {loss:.6f}")
        
        logger.info(f"✅ Training complete: Final loss = {self.loss_history[-1]:.6f}")
        
        return self.loss_history
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)
    
    def predict_binary(self, X, threshold=0.5):
        """Binary predictions (0 or 1)"""
        predictions = self.forward(X)
        return (predictions >= threshold).astype(int)
    
    def plot_loss(self, save_path=None):
        """Plot training loss"""
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss (Backpropagation)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Loss plot saved to: {save_path}")
        
        plt.show()
        logger.info("✅ Loss plot displayed")
    
    def get_weights_summary(self):
        """Get summary of weights"""
        print("\n" + "="*50)
        print("NEURAL NETWORK WEIGHTS SUMMARY")
        print("="*50)
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"\nLayer {i+1}:")
            print(f"  Weights Shape: {w.shape}")
            print(f"  Weights Mean: {np.mean(w):.6f}")
            print(f"  Weights Std: {np.std(w):.6f}")
            print(f"  Biases Shape: {b.shape}")
            print(f"  Biases Mean: {np.mean(b):.6f}")
