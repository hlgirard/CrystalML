''' Unittesting for droplet segmentation cases '''

import unittest

from src.data.utils import get_date_taken, crop, open_grey_scale_image

class TestOpenImage(unittest.TestCase):

    def test_open_asserts_error_if_file_not_found(self):
        with self.assertRaises(OSError):
            open_grey_scale_image('some/nonexistent/path.JPG')

    def test_open_non_image_raises_error(self):
        with self.assertRaises(OSError):
            open_grey_scale_image('LICENSE')

class TestCrop(unittest.TestCase):

    def setUp(self):
        self.empty_img = open_grey_scale_image('tests/test_images/empty_droplets.JPG')

    def test_crop(self):
        crop_box = (250,500,2350,4000)
        cropped = crop(self.empty_img, crop_box)
        self.assertEqual(cropped.shape, (2100, 3500))

class TestUtils(unittest.TestCase):

    def test_get_date_taken(self):
        from datetime import datetime
        img_path = 'tests/test_images/empty_droplets.JPG'
        self.assertEqual(get_date_taken(img_path), datetime(2019, 4, 10, 17, 50, 29))

if __name__ == '__main__':
    unittest.main()