# src/explainable_ai/lime_explainer.py
"""
LIME (Local Interpretable Model-agnostic Explanations) Module
Unit V Concept: LIME
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    LIME Implementation for Local Explanations
    
    Implements:
    1. LIME Explainer for Classification
    2. LIME Explainer for Regression
    3. Feature Importance Visualization
    4. Local Explanation Text
    """
    
    def __init__(self, model=None, training_data=None, feature_names=None, class_names=None):
        """
        Initialize LIME explainer
        
        Args:
            model: Trained model
            training_data: Training data
            feature_names: List of feature names
            class_names: List of class names
        """
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names or ['Non-Criminal', 'Criminal']
        self.explainer = None
        self.explanation = None
        
        logger.info("LIMEExplainer initialized")
    
    def create_explainer(self, model, X_train, mode='classification'):
        """
        Create LIME explainer
        
        Args:
            model: Trained model
            X_train: Training data
            mode: 'classification' or 'regression'
            
        Returns:
            LIME explainer object
        """
        self.model = model
        self.training_data = X_train
        
        try:
            import lime
            from lime.lime_tabular import LimeTabularExplainer
            
            if self.feature_names is None:
                self.feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
            
            self.explainer = LimeTabularExplainer(
                training_data=X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode=mode,
                discretize_continuous=True,
                discretize_continuous_method='quantile'
            )
            
            logger.info("✅ LIME explainer created")
            return self.explainer
            
        except ImportError:
            print("❌ LIME not installed. Run: pip install lime")
            return None
    
    def explain_instance(self, X_sample, predict_fn=None, num_features=10):
        """
        Explain a single instance
        
        Args:
            X_sample: Sample to explain
            predict_fn: Prediction function
            num_features: Number of features to show
            
        Returns:
            LIME explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        try:
            if predict_fn is None:
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = self.model.predict_proba
                elif hasattr(self.model, 'predict'):
                    predict_fn = self.model.predict
                else:
                    raise ValueError("Model must have predict or predict_proba method")
            
            # Get explanation
            self.explanation = self.explainer.explain_instance(
                data_row=X_sample,
                predict_fn=predict_fn,
                num_features=num_features,
                top_labels=2
            )
            
            return self.explanation
            
        except Exception as e:
            logger.error(f"Error in LIME explanation: {e}")
            return None
    
    def show_explanation(self, X_sample, predict_fn=None, save_path=None):
        """
        Display LIME explanation as visualization
        
        Args:
            X_sample: Sample to explain
            predict_fn: Prediction function
            save_path: Optional save path
        """
        explanation = self.explain_instance(X_sample, predict_fn)
        
        if explanation is None:
            print("No explanation generated")
            return
        
        # Show prediction
        if hasattr(self.model, 'predict'):
            pred = self.model.predict([X_sample])[0]
            print(f"\nPrediction: {self.class_names[pred]}")
        
        # Show explanation
        explanation.show_in_notebook()
        
        if save_path:
            explanation.save_to_file(save_path)
            logger.info(f"✅ LIME explanation saved to: {save_path}")
    
    def get_explanation_text(self, X_sample, predict_fn=None):
        """
        Get text explanation
        
        Args:
            X_sample: Sample to explain
            predict_fn: Prediction function
            
        Returns:
            Explanation text
        """
        explanation = self.explain_instance(X_sample, predict_fn)
        
        if explanation is None:
            return "No explanation generated"
        
        # Get feature weights
        feature_weights = explanation.as_list()
        
        # Get prediction
        if hasattr(self.model, 'predict'):
            pred = self.model.predict([X_sample])[0]
            prediction_label = self.class_names[pred]
        
        text = []
        text.append("🔍 LIME EXPLANATION")
        text.append("="*50)
        text.append(f"\nPrediction: {prediction_label}")
        text.append("\n📊 Feature Contributions:")
        text.append("   Feature → Contribution (Supporting)")
        
        # Split positive and negative contributions
        positives = [(f, w) for f, w in feature_weights if w > 0][:5]
        negatives = [(f, w) for f, w in feature_weights if w < 0][:5]
        
        if positives:
            text.append("\n✅ Supporting features:")
            for f, w in positives:
                text.append(f"  • {f}: {w:.4f}")
        
        if negatives:
            text.append("\n❌ Opposing features:")
            for f, w in negatives:
                text.append(f"  • {f}: {w:.4f}")
        
        return "\n".join(text)
    
    def feature_importance_plot(self, X_sample, predict_fn=None, save_path=None):
        """
        Plot feature importance for a single instance
        
        Args:
            X_sample: Sample to explain
            predict_fn: Prediction function
            save_path: Optional save path
        """
        explanation = self.explain_instance(X_sample, predict_fn)
        
        if explanation is None:
            return
        
        # Get feature weights
        feature_weights = explanation.as_list()
        
        if not feature_weights:
            print("No feature weights available")
            return
        
        # Create DataFrame
        df = pd.DataFrame(feature_weights, columns=['Feature', 'Weight'])
        
        # Plot
        plt.figure(figsize=(10, 6))
        colors = ['green' if w > 0 else 'red' for w in df['Weight']]
        plt.barh(df['Feature'], df['Weight'], color=colors, alpha=0.7)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Weight (Contribution to Prediction)')
        plt.title('LIME Feature Importance for this Instance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ LIME feature importance saved to: {save_path}")
        
        plt.show()
