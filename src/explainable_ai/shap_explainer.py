# src/explainable_ai/shap_explainer.py
"""
SHAP (SHapley Additive exPlanations) Module
Unit V Concept: SHAP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP Implementation for Model Explainability
    
    Implements:
    1. SHAP Feature Importance
    2. SHAP Summary Plot
    3. SHAP Dependence Plot
    4. SHAP Force Plot
    """
    
    def __init__(self, model=None, data=None):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model
            data: Data to explain
        """
        self.model = model
        self.data = data
        self.shap_values = None
        self.explainer = None
        
        logger.info("SHAPExplainer initialized")
    
    def create_explainer(self, model, X_train, model_type='sklearn'):
        """
        Create SHAP explainer for the model
        
        Args:
            model: Trained model
            X_train: Training data
            model_type: 'sklearn', 'tree', 'linear', 'deep'
            
        Returns:
            SHAP explainer object
        """
        self.model = model
        
        try:
            import shap
            
            if model_type == 'tree':
                self.explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                self.explainer = shap.LinearExplainer(model, X_train)
            elif model_type == 'deep':
                self.explainer = shap.DeepExplainer(model, X_train[:100])
            else:
                self.explainer = shap.KernelExplainer(model.predict, X_train[:100])
            
            logger.info(f"✅ {model_type.capitalize()} SHAP explainer created")
            return self.explainer
            
        except ImportError:
            print("❌ SHAP not installed. Run: pip install shap")
            return None
    
    def explain_prediction(self, X_sample, feature_names=None):
        """
        Explain a single prediction
        
        Args:
            X_sample: Single sample to explain
            feature_names: List of feature names
            
        Returns:
            SHAP values for the prediction
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        try:
            import shap
            
            # Get SHAP values
            self.shap_values = self.explainer.shap_values(X_sample)
            
            # Create force plot
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(X_sample.shape[1])]
            
            # Force plot for single prediction
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values,
                X_sample,
                feature_names=feature_names,
                matplotlib=True
            )
            
            return self.shap_values
            
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {e}")
            return None
    
    def summary_plot(self, X, feature_names=None, max_display=20, save_path=None):
        """
        Create SHAP summary plot
        
        Args:
            X: Data to explain
            feature_names: List of feature names
            max_display: Maximum features to display
            save_path: Optional save path
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        try:
            import shap
            
            # Get SHAP values
            self.shap_values = self.explainer.shap_values(X[:500])
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            
            # Summary plot
            shap.summary_plot(
                self.shap_values,
                X[:500],
                feature_names=feature_names,
                max_display=max_display,
                show=True
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ SHAP summary plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in SHAP summary plot: {e}")
    
    def dependence_plot(self, X, feature_idx, feature_names=None, save_path=None):
        """
        Create SHAP dependence plot
        
        Args:
            X: Data to explain
            feature_idx: Index of feature to plot
            feature_names: List of feature names
            save_path: Optional save path
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        try:
            import shap
            
            # Get SHAP values if not already computed
            if self.shap_values is None:
                self.shap_values = self.explainer.shap_values(X[:500])
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            
            # Dependence plot
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                X[:500],
                feature_names=feature_names,
                show=True
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ SHAP dependence plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in SHAP dependence plot: {e}")
    
    def feature_importance(self, X, feature_names=None, save_path=None):
        """
        Get feature importance from SHAP values
        
        Args:
            X: Data to explain
            feature_names: List of feature names
            save_path: Optional save path
            
        Returns:
            DataFrame of feature importance
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(X[:500])
            
            # Calculate mean absolute SHAP values
            importance = np.abs(shap_values).mean(axis=0)
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importance)],
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Feature Importance (SHAP)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ Feature importance saved to: {save_path}")
            
            plt.show()
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error in feature importance: {e}")
            return None
    
    def get_explanation_text(self, X_sample, prediction=None, feature_names=None):
        """
        Get human-readable explanation text
        
        Args:
            X_sample: Sample to explain
            prediction: Prediction value (optional)
            feature_names: List of feature names
            
        Returns:
            Explanation text
        """
        if self.explainer is None:
            return "⚠️ SHAP explainer not initialized"
        
        try:
            import shap
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(X_sample)
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(X_sample.shape[1])]
            
            # Get top contributing features
            shap_vals = shap_values[0] if len(shap_values.shape) == 1 else shap_values
            
            # Get positive and negative contributors
            positive_indices = np.argsort(shap_vals)[-3:][::-1]
            negative_indices = np.argsort(shap_vals)[:3]
            
            explanation = []
            explanation.append("🔍 EXPLANATION SUMMARY")
            explanation.append("="*50)
            
            if prediction is not None:
                explanation.append(f"Prediction: {'Criminal' if prediction == 1 else 'Non-Criminal'}")
            
            explanation.append("\n📈 Top features supporting this prediction:")
            for idx in positive_indices:
                explanation.append(f"  • {feature_names[idx]}: {shap_vals[idx]:.4f}")
            
            explanation.append("\n📉 Top features opposing this prediction:")
            for idx in negative_indices:
                explanation.append(f"  • {feature_names[idx]}: {shap_vals[idx]:.4f}")
            
            return "\n".join(explanation)
            
        except Exception as e:
            return f"⚠️ Could not generate explanation: {e}"
