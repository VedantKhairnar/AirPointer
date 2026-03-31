import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from inference_pipeline import process_frame
from model_loader import initialize_models

class TestInferencePipeline(unittest.TestCase):

    def setUp(self):
        self.models, self.device = initialize_models()
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame

    def test_process_frame(self):
        processed_frame = process_frame(self.dummy_frame, self.models, self.device)
        self.assertEqual(processed_frame.shape, self.dummy_frame.shape)

if __name__ == '__main__':
    unittest.main()