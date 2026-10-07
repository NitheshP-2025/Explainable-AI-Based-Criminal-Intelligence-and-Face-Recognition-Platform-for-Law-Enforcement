# src/data_handling/__init__.py
"""
Data Handling Module
"""

from .data_loader import CriminalFaceDataset
from .feature_scaling import FeatureScaler

__all__ = ['CriminalFaceDataset', 'FeatureScaler']
