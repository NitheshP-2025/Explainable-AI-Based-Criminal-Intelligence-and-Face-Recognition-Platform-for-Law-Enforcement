# src/data_handling/data_loader.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class CriminalFaceDataset:
    def __init__(self, data_path='data/criminal_faces', img_size=(128, 128)):
        self.data_path = data_path
        self.img_size = img_size
        self.X = None
        self.y = None
        self.class_names = ['non_criminal', 'criminal']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_dataset(self, sample_size=None):
        features = []
        labels = []
        
        # class_id: 0 = non_criminal, 1 = criminal
        for class_id, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                continue
            
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(root, file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, self.img_size)
                            img = img / 255.0
                            features.append(img.flatten())
                            labels.append(class_id)
                            if sample_size and len(features) >= sample_size:
                                break
                if sample_size and len(features) >= sample_size:
                    break
        
        if len(features) == 0:
            return self._create_sample_data()
        
        self.X = np.array(features, dtype=np.float32)
        self.y = np.array(labels, dtype=np.int32)
        
        print(f"✅ Loaded: {np.sum(self.y == 1)} Criminal, {np.sum(self.y == 0)} Non-Criminal")
        return self.X, self.y
    
    def _create_sample_data(self):
        np.random.seed(42)
        n_samples = 200
        n_features = self.img_size[0] * self.img_size[1]
        self.X = np.random.randn(n_samples, n_features) * 0.5 + 0.5
        self.X = np.clip(self.X, 0, 1)
        self.y = np.array([1 if i < n_samples//2 else 0 for i in range(n_samples)])
        return self.X, self.y
    
    def split_train_test(self, test_size=0.2, random_state=42):
        if self.X is None or self.y is None:
            raise ValueError("Dataset not loaded.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
