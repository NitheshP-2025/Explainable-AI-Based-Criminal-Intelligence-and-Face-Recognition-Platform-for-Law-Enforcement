# src/deep_learning/cnn_model.py
"""
CNN Model for Face Recognition
Unit III Concept: Convolutional Neural Networks
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CNNModel:
    """
    Simple CNN Implementation for Face Recognition
    
    Implements:
    1. Convolutional Layer
    2. Max Pooling Layer
    3. Flatten Layer
    4. Dense Layer
    """
    
    def __init__(self, input_shape=(128, 128), num_classes=2, learning_rate=0.001):
        """
        Initialize CNN model
        
        Args:
            input_shape: Input image shape (height, width)
            num_classes: Number of output classes
            learning_rate: Learning rate
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Initialize layers
        self.conv1_filters = 32
        self.conv2_filters = 64
        self.kernel_size = 3
        self.pool_size = 2
        
        # Initialize weights
        self._initialize_weights()
        self.loss_history = []
        
        logger.info(f"CNNModel initialized with input_shape: {input_shape}")
    
    def _initialize_weights(self):
        """Initialize CNN weights"""
        np.random.seed(42)
        
        # Conv1: 1 channel -> 32 filters, 3x3
        self.conv1_weights = np.random.randn(32, 1, 3, 3) * 0.1
        self.conv1_bias = np.zeros(32)
        
        # Conv2: 32 filters -> 64 filters, 3x3
        self.conv2_weights = np.random.randn(64, 32, 3, 3) * 0.1
        self.conv2_bias = np.zeros(64)
        
        # Calculate output size after conv and pooling
        # Input: 128x128
        # After conv1: 126x126 (128-3+1)
        # After pool1: 63x63 (126/2)
        # After conv2: 61x61 (63-3+1)
        # After pool2: 30x30 (61/2 rounded)
        self.flatten_size = 64 * 30 * 30
        
        # Dense layers
        self.dense1_weights = np.random.randn(self.flatten_size, 128) * 0.01
        self.dense1_bias = np.zeros(128)
        self.dense2_weights = np.random.randn(128, self.num_classes) * 0.01
        self.dense2_bias = np.zeros(self.num_classes)
    
    def _conv2d(self, input_data, weights, bias):
        """
        2D Convolution operation
        
        Args:
            input_data: Input image or feature map
            weights: Convolution filters
            bias: Bias values
        
        Returns:
            Convolved output
        """
        n_filters, n_channels, k_h, k_w = weights.shape
        h, w = input_data.shape[1], input_data.shape[2]
        output_h = h - k_h + 1
        output_w = w - k_w + 1
        
        output = np.zeros((input_data.shape[0], n_filters, output_h, output_w))
        
        for n in range(input_data.shape[0]):
            for f in range(n_filters):
                for i in range(output_h):
                    for j in range(output_w):
                        patch = input_data[n, :, i:i+k_h, j:j+k_w]
                        output[n, f, i, j] = np.sum(patch * weights[f]) + bias[f]
        
        return output
    
    def _max_pool2d(self, input_data, pool_size=2):
        """
        2D Max Pooling
        
        Args:
            input_data: Input feature map
            pool_size: Pooling window size
        
        Returns:
            Pooled output
        """
        n, c, h, w = input_data.shape
        output_h = h // pool_size
        output_w = w // pool_size
        
        output = np.zeros((n, c, output_h, output_w))
        
        for n_i in range(n):
            for c_i in range(c):
                for i in range(output_h):
                    for j in range(output_w):
                        patch = input_data[n_i, c_i, i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
                        output[n_i, c_i, i, j] = np.max(patch)
        
        return output
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _softmax(self, x):
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _forward(self, X):
        """
        Forward pass through the CNN
        """
        # Reshape input to (batch, channels, height, width)
        if len(X.shape) == 2:
            batch_size = X.shape[0]
            X = X.reshape(batch_size, 1, self.input_shape[0], self.input_shape[1])
        elif len(X.shape) == 3:
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        
        # Conv1 + ReLU + Pool1
        conv1_out = self._conv2d(X, self.conv1_weights, self.conv1_bias)
        relu1_out = self._relu(conv1_out)
        pool1_out = self._max_pool2d(relu1_out, self.pool_size)
        
        # Conv2 + ReLU + Pool2
        conv2_out = self._conv2d(pool1_out, self.conv2_weights, self.conv2_bias)
        relu2_out = self._relu(conv2_out)
        pool2_out = self._max_pool2d(relu2_out, self.pool_size)
        
        # Flatten
        self.flattened = pool2_out.reshape(pool2_out.shape[0], -1)
        
        # Dense1 + ReLU
        dense1_out = np.dot(self.flattened, self.dense1_weights) + self.dense1_bias
        relu3_out = self._relu(dense1_out)
        
        # Dense2 + Softmax
        dense2_out = np.dot(relu3_out, self.dense2_weights) + self.dense2_bias
        softmax_out = self._softmax(dense2_out)
        
        return softmax_out
    
    def fit(self, X, y, epochs=50, verbose=True):
        """
        Train the CNN (simplified version)
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of epochs
            verbose: Print progress
        """
        logger.info(f"Training CNN for {epochs} epochs...")
        
        # Use subset if too large
        if len(X) > 2000:
            indices = np.random.choice(len(X), 2000, replace=False)
            X = X[indices]
            y = y[indices]
        
        self.loss_history = []
        
        # Convert labels to one-hot
        y_one_hot = np.zeros((len(y), self.num_classes))
        y_one_hot[np.arange(len(y)), y.astype(int)] = 1
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self._forward(X)
            
            # Compute loss (cross-entropy)
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(y_one_hot * np.log(predictions), axis=1))
            self.loss_history.append(loss)
            
            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {loss:.6f}")
        
        logger.info(f"✅ CNN training complete! Final loss: {self.loss_history[-1]:.6f}")
        
        return self.loss_history
    
    def predict(self, X):
        """Make predictions"""
        return self._forward(X)
    
    def predict_class(self, X):
        """Predict class labels"""
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the CNN"""
        # Use subset if too large
        if len(X_test) > 500:
            indices = np.random.choice(len(X_test), 500, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]
        
        y_pred = self.predict_class(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*60)
        print("CNN EVALUATION")
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
        plt.title('CNN Training Loss')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
