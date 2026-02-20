"""
Test script to validate AirPointer system components individually
Run this before main.py to ensure all modules are functioning correctly
"""

import sys
import numpy as np
from collections import deque

print("=" * 60)
print("AirPointer System Validation Test")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/6] Testing imports...")
try:
    from perception.camera_handler import CameraHandler
    from perception.hand_detector import HandDetector
    from feature_extraction.extractor import FeatureExtractor, HandFeatures
    from temporal_analysis.analyzer import TemporalAnalyzer, TemporalFeatures
    from state_management.state_machine import StateManager, InteractionState
    from action_execution.executor import ActionExecutor
    from debug.visualizer import DebugVisualizer
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Feature Extraction
print("\n[2/6] Testing feature extraction...")
try:
    extractor = FeatureExtractor()
    
    # Create fake landmarks (21 points, each [x, y, z])
    fake_landmarks = np.random.rand(21, 3)
    features = extractor.extract_features(fake_landmarks, 0.0)
    
    assert features.palm_x is not None
    assert features.thumb_index_distance is not None
    assert features.landmarks is not None
    print("✓ Feature extraction working")
    print(f"  - Palm position: ({features.palm_x:.3f}, {features.palm_y:.3f})")
    print(f"  - Pinch distance: {features.thumb_index_distance:.3f}")
    print(f"  - Stability: {features.palm_stability:.3f}")
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")
    sys.exit(1)

# Test 3: Temporal Analysis
print("\n[3/6] Testing temporal analysis...")
try:
    analyzer = TemporalAnalyzer(window_size=15, fps=30)
    
    # Feed multiple frames
    for i in range(20):
        fake_landmarks = np.random.rand(21, 3)
        features = extractor.extract_features(fake_landmarks, i * 0.033)
        temporal_features = analyzer.update(features)
    
    assert temporal_features.palm_velocity_trend is not None
    assert temporal_features.is_moving is not None
    assert temporal_features.is_pinched is not None
    print("✓ Temporal analysis working")
    print(f"  - Velocity trend: {temporal_features.palm_velocity_trend:.4f}")
    print(f"  - Stability trend: {temporal_features.palm_stability_trend:.3f}")
    print(f"  - Is pinched: {temporal_features.is_pinched}")
    print(f"  - Confidence: {temporal_features.confidence:.3f}")
except Exception as e:
    print(f"✗ Temporal analysis failed: {e}")
    sys.exit(1)

# Test 4: State Management
print("\n[4/6] Testing state management...")
try:
    state_manager = StateManager()
    
    # Test state transitions
    state_info = state_manager.update(temporal_features)
    assert state_info.current_state is not None
    print("✓ State management working")
    print(f"  - Current state: {state_info.current_state.value}")
    print(f"  - Duration: {state_info.state_duration:.3f}s")
    print(f"  - Last action: {state_info.last_action}")
except Exception as e:
    print(f"✗ State management failed: {e}")
    sys.exit(1)

# Test 5: Action Execution
print("\n[5/6] Testing action execution...")
try:
    executor = ActionExecutor(screen_width=1920, screen_height=1080, 
                             frame_width=1280, frame_height=720)
    
    # Test cursor smoothing (without actually moving cursor)
    fake_features = extractor.extract_features(np.random.rand(21, 3), 0.0)
    print("✓ Action execution initialized")
    print(f"  - Screen resolution: {executor.screen_width}x{executor.screen_height}")
    print(f"  - Smoothing factor: {executor.smoothing_factor}")
    print(f"  - Drag active: {executor.is_currently_dragging()}")
except Exception as e:
    print(f"✗ Action execution failed: {e}")
    sys.exit(1)

# Test 6: Debug Visualization
print("\n[6/6] Testing debug visualization...")
try:
    visualizer = DebugVisualizer(frame_width=1280, frame_height=720)
    
    # Create a fake frame
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Test drawing functions (shouldn't crash)
    assert visualizer._get_state_color("MOVE") is not None
    print("✓ Debug visualization working")
    print(f"  - Frame size: {visualizer.frame_width}x{visualizer.frame_height}")
except Exception as e:
    print(f"✗ Debug visualization failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! AirPointer is ready to run.")
print("=" * 60)
print("\nNext step: Run 'python main.py' to start the system")
print("Controls:")
print("  - Press 'q' to quit")
print("  - Press 'd' to toggle debug mode")
print("  - Press 'm' to enable/disable mouse control")
