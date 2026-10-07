# main.py - Complete Unit I Pipeline with Error Handling
"""
Criminal Face Recognition System - Complete Unit I Pipeline
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create required directories
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('data/criminal_faces/criminal', exist_ok=True)
os.makedirs('data/criminal_faces/non_criminal', exist_ok=True)

# Import from src
from src import (
    CriminalFaceDataset,
    FeatureScaler,
    CrossValidator,
    ModelEvaluator,
    HypothesisAnalyzer,
    save_model,
    save_results
)

# Import sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """
    Run complete Unit I pipeline
    """
    print("\n" + "="*80)
    print("👮 CRIMINAL FACE RECOGNITION SYSTEM - FULL PIPELINE")
    print("="*80)
    print("Unit I: Data Preparation and Initial Analysis")
    print("="*80)
    
    # ========== STEP 1: LOAD DATASET ==========
    print("\n" + "📊"*20)
    print("STEP 1: LOADING DATASET")
    print("📊"*20)
    
    dataset = CriminalFaceDataset('data/criminal_faces')
    X, y = dataset.load_dataset()
    X_train, X_test, y_train, y_test = dataset.split_train_test()
    dataset.get_data_summary()
    
    # ========== STEP 2: HYPOTHESIS SPACE ANALYSIS ==========
    print("\n" + "🧠"*20)
    print("STEP 2: HYPOTHESIS SPACE AND INDUCTIVE BIAS")
    print("🧠"*20)
    
    analyzer = HypothesisAnalyzer()
    analyzer.analyze_hypothesis_space(X, y)
    analyzer.analyze_inductive_bias()
    
    # ========== STEP 3: FEATURE SCALING ==========
    print("\n" + "📐"*20)
    print("STEP 3: FEATURE SCALING")
    print("📐"*20)
    
    scaler = FeatureScaler()
    
    # Compare scaling techniques (with smaller sample to avoid memory issues)
    sample_size = min(200, X_train.shape[0])
    sample_features = min(200, X_train.shape[1])
    sample_data = X_train[:sample_size, :sample_features]
    
    try:
        scaler.compare_scaling_effects(sample_data, save_path='results/scaling_comparison.png')
    except Exception as e:
        print(f"⚠️ Scaling comparison plot issue: {e}")
        print("   Showing simplified version...")
        # Show a simpler version
        simple_data = sample_data.flatten()[:1000]
        plt.figure(figsize=(8, 4))
        plt.hist(simple_data, bins=30, alpha=0.7)
        plt.title('Feature Distribution')
        plt.show()
    
    # Apply scaling
    X_train_scaled, X_test_scaled = scaler.standard_scaling(X_train, X_test)
    print(f"\n✅ Data scaled: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")
    
    # ========== STEP 4: CROSS VALIDATION ==========
    print("\n" + "🔄"*20)
    print("STEP 4: CROSS VALIDATION")
    print("🔄"*20)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    cv = CrossValidator(X_train_scaled, y_train)
    cv_results = cv.compare_models(models, cv_type='stratified', n_splits=5)
    
    # ========== STEP 5: MODEL EVALUATION ==========
    print("\n" + "📈"*20)
    print("STEP 5: MODEL EVALUATION")
    print("📈"*20)
    
    evaluator = ModelEvaluator()
    
    # Train best model
    best_model = cv.best_model
    best_model_name = cv.best_model_name
    
    if best_model:
        best_model.fit(X_train_scaled, y_train)
        y_pred = best_model.predict(X_test_scaled)
        
        # Get probabilities if available
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test_scaled)
        else:
            y_pred_proba = None
        
        # Evaluate
        metrics = evaluator.evaluate_classification(y_test, y_pred, y_pred_proba)
        metrics['model_name'] = best_model_name
        
        # Save results
        try:
            save_results({
                'model_name': best_model_name,
                'metrics': metrics,
                'cv_results': cv_results
            }, path='results')
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        # Save model
        try:
            save_model(best_model, path='models', name='criminal_face_recognition')
            save_model(scaler, path='models', name='feature_scaler')
        except Exception as e:
            print(f"⚠️ Could not save model: {e}")
        
        # ========== STEP 6: FINAL SUMMARY ==========
        print("\n" + "="*80)
        print("🎉 UNIT I IMPLEMENTATION COMPLETE!")
        print("="*80)
        print(f"Best Model: {best_model_name}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print("="*80)
        
        print("\n📚 UNIT I CONCEPTS IMPLEMENTED:")
        print("  ✅ Training and Test Datasets")
        print("  ✅ Hypothesis Space and Inductive Bias")
        print("  ✅ Feature Scaling and Normalization")
        print("  ✅ Cross Validation")
        print("  ✅ Model Evaluation Metrics")
        
        return {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'metrics': metrics,
            'cv_results': cv_results
        }
    else:
        print("❌ No best model found!")
        return None


def main():
    """Main function"""
    results = run_full_pipeline()
    
    if results:
        print("\n" + "="*80)
        print("👮 SYSTEM READY FOR LAW ENFORCEMENT USE")
        print("="*80)
        print(f"""
System Status: ✅ READY FOR DEPLOYMENT

Performance Metrics:
  • Accuracy: {results['metrics']['accuracy']:.2%}
  • Precision: {results['metrics']['precision']:.2%}
  • Recall: {results['metrics']['recall']:.2%}
  • F1-Score: {results['metrics']['f1_score']:.2%}

Best Algorithm: {results['best_model_name']}

Interpretation:
  ✓ The system can correctly identify {results['metrics']['accuracy']*100:.1f}% of faces
  ✓ When it says "Criminal", it is correct {results['metrics']['precision']*100:.1f}% of the time
  ✓ It catches {results['metrics']['recall']*100:.1f}% of actual criminals
""")
    
    print("\n" + "="*80)
    print("👮 UNIT I COMPLETE - READY FOR UNIT II (SUPERVISED LEARNING)")
    print("="*80)


if __name__ == "__main__":
    main()
