from utils import intersection_over_union

import unittest
import torch


class IouMidpointsTest(unittest.TestCase):

    def test_iou_midpoints_0(self):
        """Checked the iou of two coinciding squares"""
        # IF
        boxes_predictions = torch.tensor([0.5, 0.5, 0.5, 0.5])
        boxes_labels = torch.tensor([0.5, 0.5, 0.5, 0.5])
        ious_expected = torch.tensor([1.0],)
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "midpoints")
        # THEN
        self.assertAlmostEqual(ious_expected.item(), ious_actual.item(), places=5)

    def test_iou_midpoints_1(self):
        """Checked the iou of two identical, overlapping squares with some offset"""
        # IF
        boxes_predictions = torch.tensor([0.25, 0.25, 0.5, 0.5])
        boxes_labels = torch.tensor([0.5, 0.5, 0.5, 0.5])
        ious_expected = torch.tensor([(1/16)/(7/16)])
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "midpoints")
        # THEN
        self.assertAlmostEqual(ious_expected.item(), ious_actual.item(), places=5)

    def test_iou_midpoints_2(self):
        """Checked the iou of two identical, overlapping squares with some offset"""
        # IF
        boxes_predictions = torch.tensor([0.5, 0.5, 0.5, 0.5])
        boxes_labels = torch.tensor([0.25, 0.25, 0.5, 0.5])
        ious_expected = torch.tensor([(1/16)/(7/16)])
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "midpoints")
        # THEN
        self.assertAlmostEqual(ious_expected.item(), ious_actual.item(), places=5)

    def test_iou_midpoints_3(self):
        """Checked the iou of two nested squares"""
        # IF
        boxes_predictions = torch.tensor([0.5, 0.5, 1.0, 1.0])
        boxes_labels = torch.tensor([0.5, 0.5, 0.5, 0.5])
        ious_expected = torch.tensor([0.25])
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "midpoints")
        # THEN
        self.assertAlmostEqual(ious_expected.item(), ious_actual.item(), places=5)

    def test_iou_midpoints_4(self):
        """Checked the iou of multiple identical, overlapping squares with some offset"""
        # IF
        boxes_predictions = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
        boxes_labels = torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.25, 0.25, 0.5, 0.5]])
        ious_expected = torch.tensor([[(1/16)/(7/16)], [(1/16)/(7/16)]])
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "midpoints")
        # THEN
        self.assertAlmostEqual(ious_expected[0].item(), ious_actual[0].item(), places=5)
        self.assertAlmostEqual(ious_expected[0].item(), ious_actual[0].item(), places=5)


class IouCenterTest(unittest.TestCase):

    def test_iou_center_0(self):
        # IF
        boxes_predictions = torch.tensor([0.25, 0.25, 0.75, 0.75])
        boxes_labels = torch.tensor([0.25, 0.25, 0.75, 0.75])
        ious_expected = torch.tensor([1.0],)
        # WHEN
        ious_actual = intersection_over_union(boxes_predictions, boxes_labels, "corners")
        # THEN
        self.assertAlmostEqual(ious_expected.item(), ious_actual.item(), places=5)


if __name__ == '__main__':
    unittest.main()