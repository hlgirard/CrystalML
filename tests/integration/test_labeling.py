import unittest

import os
import cv2
import re
import numpy as np

from src.models.utils.loading_models import load_model

class TestLabelDroplets(unittest.TestCase):

    def setUp(self):

        # Load model
        self.model_name = "cnn-simple-model.json"
        self.model = load_model(self.model_name)

        # Load test images
        self.test_images_folder = "tests/test_images/labeling_images"
        test_images = []
        for test_image in sorted(os.listdir(self.test_images_folder), key = lambda x : int(re.search(r'\d+',x).group(0))):
            img = cv2.imread(os.path.join(self.test_images_folder, test_image), 0)
            img = cv2.resize(img, (150, 150))
            img = np.expand_dims(img, axis=2)
            test_images.append(img*1./255)

        self.test_images_array = np.asarray(test_images)

        # Excpected results
        self.num_crystals = 20
        self.permitted_delta = 10

    def test_crystal_clear_predictions(self):
        Y = self.model.predict_classes(self.test_images_array).flatten().tolist()

        num_crystals = Y.count(1)

        self.assertAlmostEqual(num_crystals, self.num_crystals, delta=self.permitted_delta)
