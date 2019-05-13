import unittest

from src.data.segment_droplets import segment
from src.data.utils import open_grey_scale_image

class TestSegmentDroplets(unittest.TestCase):
    '''
    Integration testing of droplet segmentation
    '''

    def setUp(self):

        # Test images cropped around the droplet region
        self.empty_img = open_grey_scale_image('tests/test_images/empty_droplets.JPG')[250:2350, 0:4288]
        self.some_crystals_img = open_grey_scale_image('tests/test_images/some_crystals.JPG')[250:2350, 0:4288]
        self.all_crystals_img = open_grey_scale_image('tests/test_images/all_crystals.JPG')[250:2350, 0:4288]

        # Target number of droplets
        self.num_droplets = 149
        self.delta = self.num_droplets // 0.7 # 15% variation in segmentation results is allowed

    def test_segment_empty_droplets(self):
        _, num_regions = segment(self.empty_img)

        self.assertAlmostEqual(num_regions, self.num_droplets, delta=self.delta)

    def test_segment_some_crystals(self):
        _, num_regions = segment(self.some_crystals_img)

        self.assertAlmostEqual(num_regions, self.num_droplets, delta=self.delta)

    def test_segment_all_crystals(self):
        _, num_regions = segment(self.all_crystals_img)

        self.assertAlmostEqual(num_regions, self.num_droplets, delta=self.delta)

