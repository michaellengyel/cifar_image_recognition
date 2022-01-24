import unittest

import torch

from model_atomic import FeatureExtractor
from model_atomic import CnnBlock
from model_atomic import ScalePrediction
from model_atomic import Yolo


class CnnBlockTest(unittest.TestCase):

    def test_cnn_block_test_0(self):
        """Tests the cnn block"""
        # IF
        in_channels = 512
        out_channels = 512
        kernel_size = 1
        stride = 1
        padding = 0
        x = torch.randn((16, 512, 13, 13))  # (Batch, Channel, Width, Height)
        # WHEN
        model = CnnBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        y_p = model(x)
        y_p_actual = y_p.shape
        y_p_expected = (16, 512, 13, 13)
        # THEN
        self.assertEqual(y_p_expected, y_p_actual)
        print(y_p_actual)


class FeatureExtractorTest(unittest.TestCase):

    def test_feature_extractor_0(self):
        """Tests the feature extractor"""
        # IF
        x = torch.randn((16, 3, 416, 416))  # (Batch, Channel, Width, Height)
        # WHEN
        model = FeatureExtractor()
        y_p = model(x)
        y_p_actual = y_p.shape
        y_p_expected = (16, 512, 13, 13)
        # THEN
        self.assertEqual(y_p_expected, y_p_actual)
        print(y_p_actual)


class ScalePredictionTest(unittest.TestCase):

    def test_scale_prediction_test_0(self):
        """Tests the scale predictor"""
        # IF
        in_channels = 1024
        num_classes = 20
        x = torch.randn((16, 1024, 11, 11))  # (Batch, Channel, Width, Height)
        # WHEN
        model = ScalePrediction(in_channels=in_channels, num_classes=num_classes)
        y_p = model(x)
        y_p_actual = y_p.shape
        y_p_expected = (16, 512, 13, 13)
        # THEN
        self.assertEqual(y_p_expected, y_p_actual)
        print(y_p_actual)


class YoloTest(unittest.TestCase):

    def test_feature_extractor_0(self):
        """Tests the feature extractor"""
        # IF
        x = torch.randn((16, 3, 416, 416))  # (Batch, Channel, Width, Height)
        # WHEN
        model = Yolo()
        y_p = model(x)
        y_p_actual = y_p.shape
        y_p_expected = (16, 1024, 11, 11)
        # THEN
        self.assertEqual(y_p_expected, y_p_actual)
        print(y_p_actual)


if __name__ == '__main__':
    unittest.main()
