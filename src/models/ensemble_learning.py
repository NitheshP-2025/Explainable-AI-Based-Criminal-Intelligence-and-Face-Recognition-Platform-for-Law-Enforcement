# src/models/ensemble_learning.py
"""
Ensemble Learning Module
Unit II Concept: Ensemble Learning
Techniques: Hard Voting, Soft Voting, Bagging, Boosting
"""

import numpy as np
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class EnsembleLearning:
    """
    Ensemble Learning Implementation for Criminal Face Recognition
    
    Implements:
    1. Hard Voting Ensemble
    2. Soft Voting Ensemble
    3. Bagging Ensemble
    4. Boosting Ensemble (AdaBoost)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ensemble learning
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_ensemble = None
        self.best_ensemble_name = None
        self.best_accuracy = 0
        
        logger.info("EnsembleLearning initialized")
    
    def create_voting_ensemble(self, estimators: List[Tuple[str, object]], voting: str = 'hard') -> VotingClassifier:
        """
        Create a voting ensemble classifier
        
        Args:
            estimators: List of (name, model) tuples
            voting: 'hard' for majority voting, 'soft' for weighted voting
            
        Returns:
            VotingClassifier
        """
        logger.info(f"Creating {voting} voting ensemble with {len(estimators)} estimators")
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        
        return ensemble
    
    def create_bagging_ensemble(self, base_estimator=None, n_estimators: int = 10) -> BaggingClassifier:
        """
        Create a bagging ensemble
        
        Args:
            base_estimator: Base estimator (default: DecisionTree)
            n_estimators: Number of base estimators
            
        Returns:
            BaggingClassifier
        """
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(random_state=self.random_state)
        
        logger.info(f"Creating bagging ensemble with {n_estimators} estimators")
        
        ensemble = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        return ensemble
    
    def create_boosting_ensemble(self, n_estimators: int = 50) -> AdaBoostClassifier:
        """
        Create a boosting ensemble (AdaBoost)
        
        Args:
            n_estimators: Number of boosting rounds
            
        Returns:
            AdaBoostClassifier
        """
        logger.info(f"Creating boosting ensemble with {n_estimators} estimators")
        
        ensemble = AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state
        )
        
        return ensemble
    
    def compare_ensembles(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare different ensemble techniques
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of results for each ensemble
        """
        print("\n" + "="*70)
        print("🔀 ENSEMBLE LEARNING COMPARISON")
        print("="*70)
        
        results = {}
        
        # 1. Hard Voting Ensemble
        print("\n1. Hard Voting Ensemble...")
        estimators = [
            ('lr', LogisticRegression(max_iter=1000, random_state=self.random_state)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('svm', SVC(kernel='rbf', random_state=self.random_state, probability=True)),
            ('dt', DecisionTreeClassifier(random_state=self.random_state)),
        ]
        
        hard_voting = self.create_voting_ensemble(estimators, voting='hard')
        hard_voting.fit(X_train, y_train)
        y_pred = hard_voting.predict(X_test)
        
        results['Hard Voting'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        print(f"   Accuracy: {results['Hard Voting']['accuracy']:.4f}")
        print(f"   F1-Score: {results['Hard Voting']['f1']:.4f}")
        
        # 2. Soft Voting Ensemble
        print("\n2. Soft Voting Ensemble...")
        soft_voting = self.create_voting_ensemble(estimators, voting='soft')
        soft_voting.fit(X_train, y_train)
        y_pred = soft_voting.predict(X_test)
        
        results['Soft Voting'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        print(f"   Accuracy: {results['Soft Voting']['accuracy']:.4f}")
        print(f"   F1-Score: {results['Soft Voting']['f1']:.4f}")
        
        # 3. Bagging Ensemble
        print("\n3. Bagging Ensemble...")
        bagging = self.create_bagging_ensemble(n_estimators=10)
        bagging.fit(X_train, y_train)
        y_pred = bagging.predict(X_test)
        
        results['Bagging'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        print(f"   Accuracy: {results['Bagging']['accuracy']:.4f}")
        print(f"   F1-Score: {results['Bagging']['f1']:.4f}")
        
        # 4. Boosting Ensemble (AdaBoost)
        print("\n4. Boosting Ensemble (AdaBoost)...")
        boosting = self.create_boosting_ensemble(n_estimators=50)
        boosting.fit(X_train, y_train)
        y_pred = boosting.predict(X_test)
        
        results['Boosting'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        print(f"   Accuracy: {results['Boosting']['accuracy']:.4f}")
        print(f"   F1-Score: {results['Boosting']['f1']:.4f}")
        
        # Find best ensemble
        for name, metrics in results.items():
            if metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = metrics['accuracy']
                self.best_ensemble_name = name
                if name == 'Hard Voting':
                    self.best_ensemble = hard_voting
                elif name == 'Soft Voting':
                    self.best_ensemble = soft_voting
                elif name == 'Bagging':
                    self.best_ensemble = bagging
                elif name == 'Boosting':
                    self.best_ensemble = boosting
        
        print("\n" + "="*70)
        print(f"🏆 BEST ENSEMBLE: {self.best_ensemble_name}")
        print(f"   Accuracy: {self.best_accuracy:.4f}")
        print("="*70)
        
        self.results = results
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the best ensemble
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.best_ensemble is None:
            raise ValueError("No ensemble trained yet. Call compare_ensembles() first.")
        
        return self.best_ensemble.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities using the best ensemble
        
        Args:
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if self.best_ensemble is None:
            raise ValueError("No ensemble trained yet. Call compare_ensembles() first.")
        
        if hasattr(self.best_ensemble, 'predict_proba'):
            return self.best_ensemble.predict_proba(X)
        else:
            return None
