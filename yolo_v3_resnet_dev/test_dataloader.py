import unittest

import torch

from dataloader import centered_iou
from dataloader import bboxes_to_target


class CenteredIouTest(unittest.TestCase):

    def test_centered_iou_0(self):
        # IF
        boxes_target = torch.tensor([(1, 1)])
        boxes_prediction = torch.tensor([(1, 1)])
        iou_expected = torch.tensor([1])
        # WHEN
        iou = centered_iou(boxes_target, boxes_prediction)
        # THEN
        self.assertEqual(iou[0], iou_expected[0])

    def test_centered_iou_1(self):
        # IF
        boxes_target = torch.tensor([(1.0, 1.0)])
        boxes_prediction = torch.tensor([(0.5, 0.5)])
        iou_expected = torch.tensor([0.25])
        # WHEN
        iou = centered_iou(boxes_target, boxes_prediction)
        # THEN
        self.assertEqual(iou[0], iou_expected[0])

    def test_centered_iou_2(self):
        # IF
        boxes_target = torch.tensor([(1.0, 1.0)])
        boxes_prediction = torch.tensor([(1.0, 0.5)])
        iou_expected = torch.tensor([0.5])
        # WHEN
        iou = centered_iou(boxes_target, boxes_prediction)
        # THEN
        self.assertEqual(iou[0], iou_expected[0])

    def test_centered_iou_3(self):
        # IF
        boxes_target = torch.tensor([(1.0, 1.0)])
        boxes_prediction = torch.tensor([(1, 1), (1, 1), (1, 1)])
        iou_expected = torch.tensor([1.0, 1.0, 1.0])
        # WHEN
        iou = centered_iou(boxes_target, boxes_prediction)
        # THEN
        self.assertEqual(iou[0], iou_expected[0])
        self.assertEqual(iou[1], iou_expected[1])
        self.assertEqual(iou[2], iou_expected[2])

    def test_centered_iou_4(self):
        # IF
        boxes_target = torch.tensor([(1.0, 1.0)])
        boxes_prediction = torch.tensor([(1.0, 1.0), (0.5, 0.5), (0.25, 0.25)])
        iou_expected = torch.tensor([1.0, 0.25, 0.0625])
        # WHEN
        iou = centered_iou(boxes_target, boxes_prediction)
        # THEN
        self.assertEqual(iou[0], iou_expected[0])
        self.assertEqual(iou[1], iou_expected[1])
        self.assertEqual(iou[2], iou_expected[2])

    def test_centered_iou_5(self):
        # IF
        boxes_target = torch.tensor([(1, 1)])
        boxes_prediction = torch.tensor([(1, 1), (2, 2), (4, 4)])
        iou_expected = torch.tensor([1, 0.25, 0.0625])
        # WHEN
        iou = centered_iou(boxes_target, boxes_prediction)
        # THEN
        self.assertEqual(iou[0], iou_expected[0])
        self.assertEqual(iou[1], iou_expected[1])
        self.assertEqual(iou[2], iou_expected[2])

    def test_centered_iou_6(self):
        # IF
        boxes_target = torch.tensor([(1, 1)])
        boxes_prediction = torch.tensor([(0, 0)])
        iou_expected = torch.tensor([0])
        # WHEN
        iou = centered_iou(boxes_target, boxes_prediction)
        # THEN
        self.assertEqual(iou[0], iou_expected[0])

    def test_centered_iou_7(self):
        # IF
        boxes_target = torch.tensor([(0, 0)])
        boxes_prediction = torch.tensor([(1, 1)])
        iou_expected = torch.tensor([0])
        # WHEN
        iou = centered_iou(boxes_target, boxes_prediction)
        # THEN
        self.assertEqual(iou[0], iou_expected[0])


class BoxesToTargetTest(unittest.TestCase):

    def test_bboxes_to_target_0(self):
        """
        Tests case where there is no input bboxes for the image
        """
        # IF
        Scale = [13, 26, 52]
        number_of_scales = 3
        targets = [torch.zeros((number_of_scales, scale, scale, 6)) for scale in Scale]
        bboxes = torch.tensor([])  # [x, y, w, h, c]
        ignore_iou_threshold = 0.5
        num_anchors_per_scale = 3
        anchors = [
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]
        ]
        anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        targets_expected = [torch.zeros((number_of_scales, scale, scale, 6)) for scale in Scale]
        # WHEN
        targets_actual = bboxes_to_target(bboxes, targets, ignore_iou_threshold, num_anchors_per_scale, anchors, Scale)
        # THEN
        torch.testing.assert_allclose(targets_actual[0], targets_expected[0])
        torch.testing.assert_allclose(targets_actual[1], targets_expected[1])
        torch.testing.assert_allclose(targets_actual[2], targets_expected[2])

    def test_bboxes_to_target_1(self):
        """
        Tests case where there is one input bbox for the image. There are three scales of the same size.
        """
        # IF
        Scale = [4, 4, 4]
        number_of_scales = 3
        targets = [torch.zeros((number_of_scales, scale, scale, 6)) for scale in Scale]
        bboxes = torch.tensor([[0.125, 0.125, 0.25, 0.25, 1.0]])  # [x, y, w, h, c]
        ignore_iou_threshold = 0.5
        num_anchors_per_scale = 3
        anchors = [
            [(0.00, 0.00), (0.00, 0.00), (0.00, 0.00)],
            [(0.00, 0.00), (0.00, 0.00), (0.00, 0.00)],
            [(0.1, 0.1), (0.1, 0.1), (0.1, 0.1)]
        ]
        anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])

        # Create expected target
        targets_expected = [torch.zeros((number_of_scales, scale, scale, 6)) for scale in Scale]
        targets_expected[0][0, 0, 0, ...] = torch.tensor([1.0, 0.5, 0.5, 1.0, 1.0, 1.0])
        targets_expected[1][0, 0, 0, ...] = torch.tensor([1.0, 0.5, 0.5, 1.0, 1.0, 1.0])
        targets_expected[2][0, 0, 0, ...] = torch.tensor([1.0, 0.5, 0.5, 1.0, 1.0, 1.0])

        # WHEN
        targets_actual = bboxes_to_target(bboxes, targets, ignore_iou_threshold, num_anchors_per_scale, anchors, Scale)

        # THEN
        torch.testing.assert_allclose(targets_actual[0], targets_expected[0])
        torch.testing.assert_allclose(targets_actual[1], targets_expected[1])
        torch.testing.assert_allclose(targets_actual[2], targets_expected[2])

    def test_bboxes_to_target_2(self):
        """
        Tests case where there is no input bboxes for the image
        """
        # IF
        Scale = [2, 6]
        number_of_scales = 2
        num_anchors_per_scale = 3
        targets = [torch.zeros((num_anchors_per_scale, scale, scale, 6)) for scale in Scale]
        bboxes = torch.tensor([[0.25, 0.25, 0.166666667, 0.166666667, 1.0]])  # [x, y, w, h, c]
        ignore_iou_threshold = 0.5
        anchors = [
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)]
        ]
        anchors = torch.tensor(anchors[0] + anchors[1])

        # Create expected target
        targets_expected = [torch.zeros((num_anchors_per_scale, scale, scale, 6)) for scale in Scale]
        targets_expected[0][0, 0, 0, ...] = torch.tensor([1.0, 0.5, 0.5, 0.333333333, 0.333333333, 1.0])  # Scale's active anchor
        # targets_expected[0][1, 0, 0, ...] = torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # targets_expected[0][2, 0, 0, ...] = torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # targets_expected[1][0, 1, 1, ...] = torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        targets_expected[1][1, 1, 1, ...] = torch.tensor([1.0, 0.5, 0.5, 1.0, 1.0, 1.0])  # Scale's active anchor
        targets_expected[1][2, 1, 1, ...] = torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # WHEN
        targets_actual = bboxes_to_target(bboxes, targets, ignore_iou_threshold, num_anchors_per_scale, anchors, Scale)

        # THEN
        torch.testing.assert_allclose(targets_actual[0], targets_expected[0])
        torch.testing.assert_allclose(targets_actual[1], targets_expected[1])


if __name__ == '__main__':
    unittest.main()
