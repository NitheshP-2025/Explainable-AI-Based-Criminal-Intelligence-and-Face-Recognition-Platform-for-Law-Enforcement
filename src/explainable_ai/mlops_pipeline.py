# src/explainable_ai/mlops_pipeline.py
"""
MLOps Pipeline Module
Unit V Concept: MLOps and Federated Learning
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MLOpsPipeline:
    """
    MLOps Pipeline Implementation
    
    Implements:
    1. Model Versioning
    2. Model Registry
    3. Pipeline Tracking
    4. Model Monitoring
    5. Automated Retraining
    """
    
    def __init__(self, model_path='models', results_path='results', log_path='logs'):
        """
        Initialize MLOps pipeline
        
        Args:
            model_path: Path to save models
            results_path: Path to save results
            log_path: Path to save logs
        """
        self.model_path = model_path
        self.results_path = results_path
        self.log_path = log_path
        self.pipeline_run_id = None
        
        # Create directories
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        
        logger.info("MLOpsPipeline initialized")
    
    def start_pipeline_run(self):
        """Start a new pipeline run"""
        self.pipeline_run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Pipeline run started: {self.pipeline_run_id}")
        return self.pipeline_run_id
    
    def save_model_artifact(self, model, model_name, metrics=None, version=None):
        """
        Save model with versioning
        
        Args:
            model: Trained model
            model_name: Name of the model
            metrics: Model performance metrics
            version: Version number
            
        Returns:
            Model path
        """
        if version is None:
            version = datetime.now().strftime('%Y%m%d')
        
        model_dir = os.path.join(self.model_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_dir, f'{model_name}_v{version}.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'pipeline_run_id': self.pipeline_run_id,
            'metrics': metrics or {}
        }
        
        metadata_file = os.path.join(model_dir, f'{model_name}_v{version}_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model saved: {model_file}")
        return model_file
    
    def load_model_artifact(self, model_name, version='latest'):
        """
        Load model artifact
        
        Args:
            model_name: Name of the model
            version: Version number or 'latest'
            
        Returns:
            Loaded model
        """
        model_dir = os.path.join(self.model_path, model_name)
        
        if version == 'latest':
            # Get latest version
            files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            if not files:
                raise ValueError(f"No models found for {model_name}")
            files.sort()
            model_file = os.path.join(model_dir, files[-1])
        else:
            model_file = os.path.join(model_dir, f'{model_name}_v{version}.pkl')
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"✅ Model loaded: {model_file}")
        return model
    
    def get_model_metadata(self, model_name):
        """
        Get model metadata
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of metadata dictionaries
        """
        model_dir = os.path.join(self.model_path, model_name)
        
        if not os.path.exists(model_dir):
            return []
        
        metadata_files = [f for f in os.listdir(model_dir) if f.endswith('_metadata.json')]
        metadata_files.sort()
        
        metadata_list = []
        for file in metadata_files:
            with open(os.path.join(model_dir, file), 'r') as f:
                metadata_list.append(json.load(f))
        
        return metadata_list
    
    def save_pipeline_results(self, results, pipeline_name='criminal_face_recognition'):
        """
        Save pipeline results
        
        Args:
            results: Dictionary of results
            pipeline_name: Name of the pipeline
            
        Returns:
            File path
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(
            self.results_path,
            f'{pipeline_name}_{timestamp}.json'
        )
        
        # Convert numpy arrays to lists
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        serializable = {}
        for key, value in results.items():
            try:
                serializable[key] = convert(value)
            except:
                serializable[key] = str(value)
        
        serializable['timestamp'] = datetime.now().isoformat()
        serializable['pipeline_run_id'] = self.pipeline_run_id
        
        with open(results_file, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        
        logger.info(f"✅ Results saved: {results_file}")
        return results_file
    
    def get_pipeline_history(self, pipeline_name='criminal_face_recognition'):
        """
        Get pipeline run history
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            List of results files
        """
        results_files = [f for f in os.listdir(self.results_path) 
                        if f.startswith(pipeline_name) and f.endswith('.json')]
        results_files.sort()
        
        history = []
        for file in results_files:
            with open(os.path.join(self.results_path, file), 'r') as f:
                data = json.load(f)
                history.append({
                    'file': file,
                    'timestamp': data.get('timestamp', ''),
                    'pipeline_run_id': data.get('pipeline_run_id', ''),
                    'metrics': data.get('metrics', {})
                })
        
        return history
    
    def log_message(self, message, level='INFO'):
        """
        Log a message
        
        Args:
            message: Message to log
            level: Log level
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'[{timestamp}] [{level}] {message}'
        
        if self.pipeline_run_id:
            log_file = os.path.join(self.log_path, f'pipeline_{self.pipeline_run_id}.log')
        else:
            log_file = os.path.join(self.log_path, 'pipeline.log')
        
        with open(log_file, 'a') as f:
            f.write(log_entry + '\n')
        
        print(log_entry)
    
    def get_pipeline_summary(self):
        """
        Get pipeline summary
        
        Returns:
            Summary report
        """
        summary = []
        summary.append("="*60)
        summary.append("🔧 MLOPS PIPELINE SUMMARY")
        summary.append("="*60)
        
        if self.pipeline_run_id:
            summary.append(f"Current Run ID: {self.pipeline_run_id}")
        
        summary.append(f"Model Path: {self.model_path}")
        summary.append(f"Results Path: {self.results_path}")
        summary.append(f"Log Path: {self.log_path}")
        
        # Count models
        model_dirs = [d for d in os.listdir(self.model_path) 
                     if os.path.isdir(os.path.join(self.model_path, d))]
        summary.append(f"\nModels Available: {len(model_dirs)}")
        
        for model_dir in model_dirs:
            model_metadata = self.get_model_metadata(model_dir)
            versions = len(model_metadata)
            summary.append(f"  • {model_dir}: {versions} version(s)")
        
        # Count results
        result_files = [f for f in os.listdir(self.results_path) if f.endswith('.json')]
        summary.append(f"\nResults Available: {len(result_files)}")
        
        summary.append("="*60)
        
        return "\n".join(summary)
