# src/deep_learning/neural_network.py
"""
Artificial Neural Network Module
Unit III Concept: Artificial Neural Networks
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import logging

from .activation_functions import ActivationFunctions
from .backpropagation import Backpropagation

logger = logging.getLogger(__name__)


class NeuralNetwork:
    """
    Artificial Neural Network Implementation
    
    Implements:
    1. Multi-layer Neural Network
    2. Forward Propagation
    3. Backpropagation
    4. Training and Prediction
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, activation='relu'):
        """
        Initialize neural network
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            learning_rate: Learning rate
            activation: Activation function ('sigmoid', 'tanh', 'relu')
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights = []
        self.biases = []
        self.loss_history = []
        
        self._initialize_weights()
        
        logger.info(f"NeuralNetwork initialized with layers: {layer_sizes}")
    
    def _initialize_weights(self):
        """Initialize weights and biases"""
        np.random.seed(42)
        
        for i in range(len(self.layer_sizes) - 1):
            # He initialization for ReLU, Xavier for others
            if self.activation == 'relu':
                limit = np.sqrt(2 / self.layer_sizes[i])
            else:
                limit = np.sqrt(6 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            
            w = np.random.uniform(-limit, limit, (self.layer_sizes[i], self.layer_sizes[i + 1]))
            b = np.zeros((1, self.layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _activation_func(self, x):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            return ActivationFunctions.sigmoid(x)
        elif self.activation == 'tanh':
            return ActivationFunctions.tanh(x)
        elif self.activation == 'relu':
            return ActivationFunctions.relu(x)
        else:
            return x
    
    def _activation_derivative(self, x):
        """Derivative of activation function"""
        if self.activation == 'sigmoid':
            return ActivationFunctions.sigmoid_derivative(x)
        elif self.activation == 'tanh':
            return ActivationFunctions.tanh_derivative(x)
        elif self.activation == 'relu':
            return ActivationFunctions.relu_derivative(x)
        else:
            return 1
    
    def _forward_propagate(self, X):
        """Forward propagation through the network"""
        self.layer_outputs = [X]
        current_input = X
        
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            current_input = self._activation_func(z)
            self.layer_outputs.append(current_input)
        
        return current_input
    
    def _backward_propagate(self, y_true):
        """Backward propagation"""
        n_samples = y_true.shape[0]
        
        # Convert labels to one-hot if needed
        if len(y_true.shape) == 1:
            y_one_hot = np.zeros((n_samples, self.layer_sizes[-1]))
            y_one_hot[np.arange(n_samples), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        # Output layer error
        output = self.layer_outputs[-1]
        error = output - y_true
        
        # Backpropagate error
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
            
            dW = np.dot(layer_input.T, delta) / n_samples
            db = np.mean(delta, axis=0, keepdims=True)
            
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
    
    def fit(self, X, y, epochs=100, verbose=True):
        """
        Train the neural network
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of epochs
            verbose: Print progress
        """
        logger.info(f"Training Neural Network for {epochs} epochs...")
        
        # Use subset if dataset is too large
        if len(X) > 5000:
            logger.info(f"Using subset of 5000 samples for training")
            indices = np.random.choice(len(X), 5000, replace=False)
            X = X[indices]
            y = y[indices]
        
        self.loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self._forward_propagate(X)
            
            # Compute loss (MSE)
            if len(y.shape) == 1:
                y_one_hot = np.zeros((len(y), self.layer_sizes[-1]))
                y_one_hot[np.arange(len(y)), y.astype(int)] = 1
                y_true = y_one_hot
            else:
                y_true = y
            
            loss = np.mean((predictions - y_true) ** 2)
            self.loss_history.append(loss)
            
            # Backward pass
            self._backward_propagate(y)
            
            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {loss:.6f}")
        
        logger.info(f"✅ Training complete! Final loss: {self.loss_history[-1]:.6f}")
        
        return self.loss_history
    
    def predict(self, X):
        """Make predictions"""
        return self._forward_propagate(X)
    
    def predict_binary(self, X, threshold=0.5):
        """Binary predictions"""
        predictions = self.predict(X)
        return (predictions >= threshold).astype(int)
    
    def predict_class(self, X):
        """Multi-class predictions"""
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        # Use subset if too large
        if len(X_test) > 1000:
            indices = np.random.choice(len(X_test), 1000, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]
        
        y_pred = self.predict_binary(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*60)
        print("NEURAL NETWORK EVALUATION")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Criminal', 'Criminal']))
        
        return {
            'accuracy': accuracy,
            'y_pred': y_pred
        }
    
    def plot_loss(self, save_path=None):
        """Plot training loss"""
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Neural Network Training Loss')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_summary(self):
        """Get network summary"""
        print("\n" + "="*60)
        print("NEURAL NETWORK SUMMARY")
        print("="*60)
        print(f"Layer Configuration: {self.layer_sizes}")
        print(f"Activation Function: {self.activation}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Total Parameters: {sum([w.size + b.size for w, b in zip(self.weights, self.biases)])}")
        print(f"Final Loss: {self.loss_history[-1]:.6f}" if self.loss_history else "Not trained yet")
