"""
Temporal Analysis module for AirPointer
- Maintains sliding window of features
- Computes deltas, stability, and temporal features
"""

from .analyzer import TemporalAnalyzer, TemporalFeatures

__all__ = ['TemporalAnalyzer', 'TemporalFeatures']
