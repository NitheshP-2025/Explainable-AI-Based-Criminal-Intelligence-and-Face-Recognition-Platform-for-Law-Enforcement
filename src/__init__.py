# src/__init__.py
"""
Criminal Face Recognition System
"""

# Data Handling (Unit I)
from src.data_handling import CriminalFaceDataset, FeatureScaler

# Core
from src.core import CrossValidator

# Explainable AI (Unit V)
from src.explainable_ai import (
    HypothesisAnalyzer,
    ModelEvaluator,
    SHAPExplainer,
    LIMEExplainer,
    FairnessAnalysis,
    MLOpsPipeline
)

# Models (Unit II)
from src.models import (
    EnsembleLearning,
    GradientDescent,
    NaiveBayesModel,
    PerceptronModel
)

# Deep Learning (Unit III)
from src.deep_learning import (
    NeuralNetwork,
    CNNModel,
    ActivationFunctions,
    Backpropagation
)

# Unsupervised Learning (Unit IV)
from src.unsupervised import (
    KMeansClustering,
    HierarchicalClustering,
    GaussianMixtureModel,
    PCAReduction,
    LDAReduction,
    TSNEVisualization,
    AnomalyDetection,
    ClusterEvaluation
)

# Deployment
from src.deployment import CriminalDashboard

__version__ = "3.0.0"

__all__ = [
    # Data Handling
    'CriminalFaceDataset',
    'FeatureScaler',
    # Core
    'CrossValidator',
    # Explainable AI
    'HypothesisAnalyzer',
    'ModelEvaluator',
    'SHAPExplainer',
    'LIMEExplainer',
    'FairnessAnalysis',
    'MLOpsPipeline',
    # Models
    'EnsembleLearning',
    'GradientDescent',
    'NaiveBayesModel',
    'PerceptronModel',
    # Deep Learning
    'NeuralNetwork',
    'CNNModel',
    'ActivationFunctions',
    'Backpropagation',
    # Unsupervised
    'KMeansClustering',
    'HierarchicalClustering',
    'GaussianMixtureModel',
    'PCAReduction',
    'LDAReduction',
    'TSNEVisualization',
    'AnomalyDetection',
    'ClusterEvaluation',
    # Deployment
    'CriminalDashboard'
]