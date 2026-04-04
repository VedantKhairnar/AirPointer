import numpy as np

from feature_extraction.extractor import HandFeatures
from temporal_analysis.analyzer import TemporalAnalyzer


def _features(ts: float, pinch_distance: float) -> HandFeatures:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    return HandFeatures(
        palm_x=0.5,
        palm_y=0.5,
        palm_z=0.0,
        palm_velocity=0.01,
        palm_acceleration=0.0,
        palm_stability=0.9,
        index_extended=True,
        middle_extended=True,
        ring_extended=True,
        pinky_extended=True,
        thumb_extended=True,
        thumb_index_distance=pinch_distance,
        finger_spread=0.1,
        timestamp=ts,
        landmarks=landmarks,
    )


def test_pinch_hysteresis_reduces_chatter():
    analyzer = TemporalAnalyzer(window_size=5)

    # Enter pinch below threshold.
    t1 = analyzer.update(_features(1.0, 0.28))
    assert t1.is_pinched is True

    # Slightly above base threshold, but below exit threshold -> should stay pinched.
    t2 = analyzer.update(_features(1.1, 0.33))
    assert t2.is_pinched is True

    # Above exit threshold -> release pinch.
    t3 = analyzer.update(_features(1.2, 0.38))
    assert t3.is_pinched is False
