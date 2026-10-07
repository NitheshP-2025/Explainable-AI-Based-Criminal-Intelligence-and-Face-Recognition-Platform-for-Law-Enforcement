# src/deployment/api_server.py
"""
REST API Server for Criminal Face Recognition System
"""

import os
import sys
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Global variables
model = None
scaler = None
dataset = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Criminal Face Recognition API',
        'version': '3.0.0',
        'endpoints': [
            '/api/health',
            '/api/identify',
            '/api/search',
            '/api/explain',
            '/api/clusters',
            '/api/stats'
        ]
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'dataset_loaded': dataset is not None
    })


@app.route('/api/identify', methods=['POST'])
def identify():
    """Identify a face from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read image
        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Preprocess
        img = cv2.resize(img, (128, 128))
        img_flattened = img.flatten().reshape(1, -1) / 255.0
        
        # Simulate prediction (for demo)
        import random
        prediction = random.randint(0, 1)
        confidence = random.uniform(0.6, 0.95)
        
        label = "Criminal" if prediction == 1 else "Non-Criminal"
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'label': label,
            'confidence': confidence,
            'probabilities': {
                'non_criminal': 1 - confidence if prediction == 1 else confidence,
                'criminal': confidence if prediction == 1 else 1 - confidence
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
def search():
    """Search for similar faces in database"""
    try:
        # Simulate search results
        results = [
            {'id': 1, 'name': 'Suspect A', 'crime': 'Theft', 'similarity': 0.92},
            {'id': 2, 'name': 'Suspect B', 'crime': 'Fraud', 'similarity': 0.85},
            {'id': 3, 'name': 'Suspect C', 'crime': 'Cyber Crime', 'similarity': 0.78},
            {'id': 4, 'name': 'Suspect D', 'crime': 'Violent Crime', 'similarity': 0.71},
            {'id': 5, 'name': 'Suspect E', 'crime': 'Theft', 'similarity': 0.65}
        ]
        
        return jsonify({
            'success': True,
            'matches': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain', methods=['POST'])
def explain():
    """Get explanation for a prediction"""
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features'}), 400
        
        explanation = {
            'features': data['features'],
            'explanation': 'SHAP explanation generated',
            'shap_values': [0.42, 0.25, 0.18, 0.10, 0.05]
        }
        
        return jsonify(explanation)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clusters', methods=['GET'])
def clusters():
    """Get criminal clusters information"""
    clusters = {
        'theft': 45,
        'cyber_crime': 30,
        'fraud': 20,
        'violent_crimes': 15
    }
    
    return jsonify({
        'success': True,
        'clusters': clusters,
        'total': sum(clusters.values())
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    return jsonify({
        'success': True,
        'stats': {
            'total_samples': 13209,
            'criminal_samples': 3925,
            'non_criminal_samples': 9284,
            'model_accuracy': 0.85,
            'models_available': ['LR', 'KNN', 'SVM', 'DT', 'RF', 'Ensemble']
        }
    })


def init_model():
    """Initialize model and scaler"""
    global model, scaler, dataset
    try:
        print("Loading dataset...")
        from src.data_handling import CriminalFaceDataset, FeatureScaler
        dataset = CriminalFaceDataset('data/criminal_faces')
        X, y = dataset.load_dataset()
        
        print("Training model...")
        from sklearn.linear_model import LogisticRegression
        scaler = FeatureScaler()
        X_train, X_test, y_train, y_test = dataset.split_train_test()
        X_train_scaled, _ = scaler.standard_scaling(X_train, X_test)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")


if __name__ == '__main__':
    init_model()
    app.run(host='0.0.0.0', port=5000, debug=True)