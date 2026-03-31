import sys
import os
import unittest
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_loader import initialize_models

class TestModelLoader(unittest.TestCase):

    def test_initialize_models(self):
        models, device = initialize_models()
        self.assertIn('stage_one', models)
        self.assertIn('stage_two', models)
        self.assertIsNotNone(models['stage_one'])
        self.assertIsNotNone(models['stage_two'])
        self.assertEqual(device.type, 'cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    unittest.main()