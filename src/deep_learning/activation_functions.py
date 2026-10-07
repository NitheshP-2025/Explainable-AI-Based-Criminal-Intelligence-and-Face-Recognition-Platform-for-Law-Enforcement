# src/deep_learning/activation_functions.py
"""
Activation Functions Module
Unit III Concept: Activation Functions
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class ActivationFunctions:
    """
    Implementation of Common Activation Functions
    
    Functions:
    1. Sigmoid
    2. Tanh
    3. ReLU
    4. Leaky ReLU
    5. ELU
    6. Softmax
    """
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function: 1 / (1 + exp(-x))"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        """Tanh activation function: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x):
        """ReLU activation function: max(0, x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU"""
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU: max(alpha*x, x)"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """Derivative of Leaky ReLU"""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def elu(x, alpha=1.0):
        """ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise"""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def elu_derivative(x, alpha=1.0):
        """Derivative of ELU"""
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    @staticmethod
    def softmax(x):
        """Softmax activation for multi-class classification"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def plot_activation_functions(save_path=None):
        """
        Visualize all activation functions
        """
        x = np.linspace(-5, 5, 1000)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Sigmoid
        axes[0, 0].plot(x, ActivationFunctions.sigmoid(x), 'b-', linewidth=2)
        axes[0, 0].set_title('Sigmoid')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
        
        # 2. Tanh
        axes[0, 1].plot(x, ActivationFunctions.tanh(x), 'r-', linewidth=2)
        axes[0, 1].set_title('Tanh')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 3. ReLU
        axes[0, 2].plot(x, ActivationFunctions.relu(x), 'g-', linewidth=2)
        axes[0, 2].set_title('ReLU')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Leaky ReLU
        axes[1, 0].plot(x, ActivationFunctions.leaky_relu(x, 0.1), 'orange', linewidth=2)
        axes[1, 0].set_title('Leaky ReLU (alpha=0.1)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. ELU
        axes[1, 1].plot(x, ActivationFunctions.elu(x), 'purple', linewidth=2)
        axes[1, 1].set_title('ELU (alpha=1.0)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Softmax (multi-class)
        x_softmax = np.array([[1, 2, 3], [3, 2, 1]])
        y_softmax = ActivationFunctions.softmax(x_softmax)
        axes[1, 2].bar(['Class 1', 'Class 2', 'Class 3'], y_softmax[0])
        axes[1, 2].set_title('Softmax (Multi-class)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Activation Functions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Activation functions saved to: {save_path}")
        
        plt.show()
        logger.info("✅ Activation functions plotted")
