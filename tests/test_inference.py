import unittest
import numpy as np
import os
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils
from PIL import Image

class TestInference(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("Setting up TestInference...")
        cls.model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        cls.feature_extractor, cls.model, cls.device = model_utils.load_model(cls.model_name)
        
        # Create dummy image
        cls.image = Image.new('RGB', (512, 512), color = 'red')

    def test_model_loading(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.feature_extractor)

    def test_prediction_shape(self):
        mask = model_utils.predict_mask(self.image, (self.feature_extractor, self.model, self.device))
        self.assertEqual(mask.shape, (512, 512))
        
    def test_safety_mapping(self):
        # Create a dummy mask with known IDs
        # 4: floor (safe), 10: grass (safe), 17: mountain (safe)
        # 12: person (hazard), 20: car (hazard)
        # 999: unknown
        
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[0:10, :] = 4 # safe
        mask[10:20, :] = 12 # hazard
        
        # We need the real id2label to test logic, or mock it.
        # Let's use the loaded model's config
        id2label = self.model.config.id2label
        
        # Check if our assumptions about IDs hold specific to ADE20k
        # "floor" id might differ.
        # Instead, let's reverse lookup from the loaded model to be robust.
        label2id = {v: k for k, v in id2label.items()}
        
        # Find a safe label and a hazard label
        safe_lbl = "grass"
        hazard_lbl = "person"
        
        # Partial match lookup
        safe_id = -1
        hazard_id = -1
        
        for k, v in id2label.items():
            if safe_lbl in v.lower():
                safe_id = int(k)
                break
        
        for k, v in id2label.items():
            if hazard_lbl in v.lower():
                hazard_id = int(k)
                break
                
        if safe_id != -1:
            mask[0:50, :] = safe_id
        if hazard_id != -1:
            mask[50:100, :] = hazard_id
            
        safety_mask = model_utils.map_classes_to_safety(mask, id2label)
        
        # Top half should be 1 (safe), bottom 2 (hazard)
        if safe_id != -1:
            self.assertTrue(np.all(safety_mask[0:50, :] == 1))
        if hazard_id != -1:
            self.assertTrue(np.all(safety_mask[50:100, :] == 2))

    def test_stats_computation(self):
        mask = np.zeros((10, 10), dtype=np.int64)
        safety_mask = np.zeros((10, 10), dtype=np.uint8)
        
        # 50% safe, 50% hazard
        safety_mask[0:5, :] = 1
        safety_mask[5:10, :] = 2
        
        stats = model_utils.compute_stats(mask, safety_mask, self.model.config.id2label)
        
        self.assertEqual(stats["safe_pixels"], 50)
        self.assertEqual(stats["hazard_pixels"], 50)
        self.assertAlmostEqual(stats["safe_percentage"], 50.0)
        self.assertAlmostEqual(stats["hazard_percentage"], 50.0)

if __name__ == '__main__':
    unittest.main()
