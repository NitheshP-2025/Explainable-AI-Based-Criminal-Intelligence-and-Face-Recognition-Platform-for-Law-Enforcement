# src/hypothesis_analysis.py
"""
Hypothesis Space and Inductive Bias Analysis Module
Unit I Concept: Hypothesis Space and Inductive Bias
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class HypothesisAnalyzer:
    """
    Hypothesis Space and Inductive Bias Analysis for Criminal Face Recognition
    Unit I Concept: Hypothesis Space and Inductive Bias
    """
    
    def __init__(self):
        """Initialize the hypothesis analyzer"""
        self.hypothesis_spaces = {}
        self.inductive_biases = {}
        
        logger.info("HypothesisAnalyzer initialized")
    
    def analyze_hypothesis_space(self, X, y):
        """
        Analyze the hypothesis space for criminal face recognition
        """
        n_features = X.shape[1]
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        print("\n" + "="*70)
        print("🧠 HYPOTHESIS SPACE ANALYSIS")
        print("="*70)
        
        print(f"\n📐 FEATURE SPACE:")
        print(f"  • Number of features: {n_features:,}")
        print(f"  • Feature space dimension: {n_features}D")
        
        print(f"\n🎯 TARGET SPACE:")
        print(f"  • Number of classes: {n_classes}")
        print(f"  • Binary classification: Criminal vs Non-Criminal")
        
        print(f"\n🌌 THEORETICAL HYPOTHESIS SPACE:")
        print(f"  • For continuous features, hypothesis space is INFINITE")
        print(f"  • Each hypothesis h: R^{n_features} → {{0,1}}")
        
        print("\n" + "="*70)
        print("📊 ALGORITHM-SPECIFIC HYPOTHESIS SPACES")
        print("="*70)
        
        algorithms = {
            'Logistic Regression': {
                'type': 'Linear Classifier',
                'hypothesis': f'h(x) = σ(w·x + b)',
                'parameters': f'{n_features+1} weights',
                'bias': 'Assumes linear decision boundary'
            },
            'KNN': {
                'type': 'Instance-Based Learner',
                'hypothesis': 'h(x) = majority vote of k nearest neighbors',
                'parameters': f'k=5 neighbors',
                'bias': 'Assumes local similarity in feature space'
            },
            'SVM': {
                'type': 'Kernel Method',
                'hypothesis': 'h(x) = sign(ΣαᵢyᵢK(x, xᵢ) + b)',
                'parameters': f'Support vectors',
                'bias': 'Maximum margin separation'
            },
            'Decision Tree': {
                'type': 'Tree-Based Classifier',
                'hypothesis': 'Hierarchical axis-aligned partitions',
                'parameters': 'Tree depth',
                'bias': 'Occam\'s razor'
            }
        }
        
        for algo, info in algorithms.items():
            print(f"\n🔹 {algo}:")
            print(f"   Type: {info['type']}")
            print(f"   Hypothesis: {info['hypothesis']}")
            print(f"   Parameters: {info['parameters']}")
            print(f"   Inductive Bias: {info['bias']}")
        
        self.hypothesis_spaces = algorithms
        return algorithms
    
    def analyze_inductive_bias(self):
        """Analyze inductive biases of different algorithms"""
        print("\n" + "="*70)
        print("🎯 INDUCTIVE BIAS ANALYSIS")
        print("="*70)
        
        biases = {
            'Logistic Regression': {
                'bias_type': 'Preference Bias',
                'description': 'Prefers linear decision boundaries',
                'assumption': 'Classes are linearly separable',
                'police_use': 'Good for quick, initial screening'
            },
            'KNN': {
                'bias_type': 'Restriction Bias',
                'description': 'Assumes similarity implies same class',
                'assumption': 'Similar looking faces are likely same class',
                'police_use': 'Good for finding similar-looking criminals'
            },
            'SVM': {
                'bias_type': 'Preference Bias',
                'description': 'Prefers maximum margin hyperplanes',
                'assumption': 'There exists a separating hyperplane',
                'police_use': 'Best for high-stakes identification'
            },
            'Decision Tree': {
                'bias_type': 'Restriction Bias',
                'description': 'Axis-aligned decision boundaries',
                'assumption': 'Features can be represented hierarchically',
                'police_use': 'Great for explaining WHY'
            }
        }
        
        for algo, bias_info in biases.items():
            print(f"\n🔹 {algo}:")
            print(f"   Bias Type: {bias_info['bias_type']}")
            print(f"   Description: {bias_info['description']}")
            print(f"   Assumption: {bias_info['assumption']}")
            print(f"   Police Use: {bias_info['police_use']}")
        
        self.inductive_biases = biases
        return biases
    
    def get_hypothesis_summary(self):
        """Get summary of all analyses"""
        summary = []
        summary.append("="*70)
        summary.append("📋 HYPOTHESIS SPACE AND INDUCTIVE BIAS SUMMARY")
        summary.append("="*70)
        
        if self.hypothesis_spaces:
            summary.append("\n🔬 HYPOTHESIS SPACES:")
            for algo, info in self.hypothesis_spaces.items():
                summary.append(f"  • {algo}: {info['type']}")
        
        if self.inductive_biases:
            summary.append("\n🎯 INDUCTIVE BIASES:")
            for algo, info in self.inductive_biases.items():
                summary.append(f"  • {algo}: {info['bias_type']}")
        
        return "\n".join(summary)
    
    def analyze_model_decision(self, model, X_sample):
        """Analyze how a model makes decisions"""
        print("\n" + "="*70)
        print("🔍 MODEL DECISION ANALYSIS")
        print("="*70)
        print(f"Model: {model.__class__.__name__}")
        
        # Get prediction
        if hasattr(model, 'predict'):
            prediction = model.predict(X_sample)
            print(f"Prediction: {prediction[0] if len(prediction) > 0 else prediction}")
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_sample)
            print(f"Probabilities: {probabilities[0] if len(probabilities) > 0 else probabilities}")
