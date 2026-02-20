"""
Feature Extraction module for AirPointer
- Converts landmarks to physical features
- Outputs structured feature vector per frame
"""

from .extractor import FeatureExtractor, HandFeatures

__all__ = ['FeatureExtractor', 'HandFeatures']
